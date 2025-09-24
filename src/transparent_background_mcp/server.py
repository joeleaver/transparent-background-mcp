"""
Transparent Background MCP Server

A local AI-powered background removal MCP server using state-of-the-art models.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    ServerCapabilities,
    ToolsCapability,
)
from pydantic import BaseModel, Field

from .utils import HardwareDetector, ImageProcessor, ModelManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
hardware_detector = HardwareDetector()
model_manager = ModelManager()
image_processor = ImageProcessor()

# Model instances cache
model_cache: Dict[str, Any] = {}


class RemoveBackgroundRequest(BaseModel):
    """Request model for background removal."""
    image_data: str = Field(description="Image file path or base64 encoded image data")
    model_name: str = Field(default="ben2-base", description="Model to use for background removal")
    output_format: str = Field(default="PNG", description="Output format (PNG, JPEG)")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for detection")


class BatchRemoveBackgroundRequest(BaseModel):
    """Request model for batch background removal."""
    images_data: List[str] = Field(description="List of base64 encoded image data")
    model_name: str = Field(default="ben2-base", description="Model to use for background removal")
    output_format: str = Field(default="PNG", description="Output format (PNG, JPEG)")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for detection")


class YOLOSegmentRequest(BaseModel):
    """Request model for YOLO object segmentation."""
    image_data: str = Field(description="Base64 encoded image data")
    model_name: str = Field(default="yolo11m-seg", description="YOLO model to use")
    target_classes: Optional[List[str]] = Field(default=None, description="Target object classes to segment")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for detection")
    combine_masks: bool = Field(default=True, description="Whether to combine multiple object masks")


# Initialize MCP server
server = Server("transparent-background-mcp")


async def get_model_instance(model_name: str):
    """Get or create model instance."""
    if model_name not in model_cache:
        logger.info(f"Creating new model instance: {model_name}")
        if model_name.startswith("ben2"):
            from .models import BEN2Model
            model_cache[model_name] = BEN2Model(model_name)
        elif model_name.startswith("yolo11"):
            from .models import YOLOModel
            model_cache[model_name] = YOLOModel(model_name)
        elif model_name.startswith("inspyrenet"):
            from .models import InSPyReNetModel
            model_cache[model_name] = InSPyReNetModel(model_name)
        else:
            logger.warning(f"Unknown model {model_name}, defaulting to ben2-base")
            from .models import BEN2Model
            model_cache[model_name] = BEN2Model("ben2-base")
    return model_cache[model_name]


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="remove_background",
            description="Remove background from a single image using AI models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "Image file path or base64 encoded image data (with or without data URL prefix)"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Model to use for background removal",
                        "enum": ["ben2-base", "inspyrenet-base", "yolo11s-seg", "yolo11l-seg"],
                        "default": "ben2-base"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output image format",
                        "enum": ["PNG", "JPEG"],
                        "default": "PNG"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Confidence threshold for detection (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5
                    }
                },
                "required": ["image_data"]
            }
        ),
        Tool(
            name="batch_remove_background",
            description="Remove background from multiple images using AI models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "images_data": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of image file paths or base64 encoded image data"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Model to use for background removal",
                        "enum": ["ben2-base", "inspyrenet-base", "yolo11s-seg", "yolo11l-seg"],
                        "default": "ben2-base"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output image format",
                        "enum": ["PNG", "JPEG"],
                        "default": "PNG"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Confidence threshold for detection (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5
                    }
                },
                "required": ["images_data"]
            }
        ),
        Tool(
            name="yolo_segment_objects",
            description="Segment specific objects using YOLO11 models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "Image file path or base64 encoded image data"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "YOLO model to use",
                        "enum": ["yolo11s-seg", "yolo11m-seg", "yolo11l-seg"],
                        "default": "yolo11m-seg"
                    },
                    "target_classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target object classes to segment (e.g., ['person', 'car']). If not specified, segments all detected objects."
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Confidence threshold for detection (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5
                    },
                    "combine_masks": {
                        "type": "boolean",
                        "description": "Whether to combine multiple object masks into one",
                        "default": True
                    }
                },
                "required": ["image_data"]
            }
        ),
        Tool(
            name="get_available_models",
            description="Get information about available models and system recommendations",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_system_info",
            description="Get system hardware information and capabilities",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "remove_background":
            return await handle_remove_background(arguments)
        elif name == "batch_remove_background":
            return await handle_batch_remove_background(arguments)
        elif name == "yolo_segment_objects":
            return await handle_yolo_segment_objects(arguments)
        elif name == "get_available_models":
            return await handle_get_available_models(arguments)
        elif name == "get_system_info":
            return await handle_get_system_info(arguments)

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_remove_background(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle single image background removal."""
    start_time = time.time()
    try:
        # Parse arguments
        image_data = arguments["image_data"]
        model_name = arguments.get("model_name", "ben2-base")
        output_format = arguments.get("output_format", "PNG")
        confidence_threshold = arguments.get("confidence_threshold", 0.5)
        
        # Load image (from file path or base64)
        image = image_processor.load_image(image_data)
        
        # Resize if too large
        image = image_processor.resize_image_if_needed(image, max_size=(2048, 2048))
        
        # Get model instance
        model = await get_model_instance(model_name)
        
        # Process image
        result_image = await model.remove_background(
            image,
            confidence_threshold=confidence_threshold
        )

        # Optionally save result to disk if input was a file path
        output_file_path = None
        try:
            if not image_data.startswith('data:') and not ImageProcessor._is_base64(image_data):
                in_path = Path(image_data)
                if in_path.exists() and in_path.is_file():
                    suffix = os.getenv("OUTPUT_SUFFIX", "-no-bg")
                    ext = ".png" if output_format.upper() == "PNG" else ".jpg"
                    out_path = in_path.with_name(f"{in_path.stem}{suffix}{ext}")

                    # Handle JPEG alpha flattening
                    save_image = result_image
                    if output_format.upper() == "JPEG" and save_image.mode == "RGBA":
                        from PIL import Image as _PIL_Image
                        bg = _PIL_Image.new("RGB", save_image.size, (255, 255, 255))
                        bg.paste(save_image, mask=save_image.split()[-1])
                        save_image = bg

                    save_kwargs = {"optimize": True}
                    if output_format.upper() == "JPEG":
                        save_kwargs.update({"quality": 95})
                    save_image.save(out_path, format=output_format.upper(), **save_kwargs)
                    output_file_path = str(out_path)
        except Exception as e:
            logger.warning(f"Failed to save output image to disk: {e}")

        # Also encode result to base64 for clients that want inline data
        result_base64 = image_processor.encode_image_to_base64(
            result_image,
            format=output_format
        )

        # Return JSON response
        response = {
            "success": True,
            "message": f"Background removed successfully using {model_name}",
            "output_image_base64": result_base64,
            "output_file_path": output_file_path,
            "model_used": model_name,
            "processing_time_seconds": time.time() - start_time,
            "output_format": output_format,
            "image_size": image.size
        }

        return [TextContent(
            type="text",
            text=json.dumps(response)
        )]

    except Exception as e:
        logger.error(f"Background removal failed: {e}")
        return [TextContent(type="text", text=f"Background removal failed: {str(e)}")]


async def handle_batch_remove_background(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle batch image background removal."""
    start_time = time.time()
    try:
        # Parse arguments
        images_data = arguments["images_data"]
        model_name = arguments.get("model_name", "ben2-base")
        output_format = arguments.get("output_format", "PNG")
        confidence_threshold = arguments.get("confidence_threshold", 0.5)

        if not images_data:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "message": "No images provided",
                "model_used": model_name
            }))]

        # Load images (from file paths or base64)
        images = []
        for i, image_data in enumerate(images_data):
            try:
                image = image_processor.load_image(image_data)
                image = image_processor.resize_image_if_needed(image, max_size=(2048, 2048))
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to load image {i}: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "message": f"Failed to load image {i}: {str(e)}",
                    "model_used": model_name
                }))]

        # Get model instance
        model = await get_model_instance(model_name)

        # Process images
        result_images = await model.batch_remove_background(
            images,
            confidence_threshold=confidence_threshold
        )

        # Save results next to input files when file paths are provided
        output_file_paths: List[Optional[str]] = []
        for idx, src in enumerate(images_data):
            saved_path = None
            try:
                if not str(src).startswith('data:') and not ImageProcessor._is_base64(str(src)):
                    in_path = Path(str(src))
                    if in_path.exists() and in_path.is_file():
                        suffix = os.getenv("OUTPUT_SUFFIX", "-no-bg")
                        ext = ".png" if output_format.upper() == "PNG" else ".jpg"
                        out_path = in_path.with_name(f"{in_path.stem}{suffix}{ext}")
                        save_img = result_images[idx]
                        if output_format.upper() == "JPEG" and save_img.mode == "RGBA":
                            from PIL import Image as _PIL_Image
                            bg = _PIL_Image.new("RGB", save_img.size, (255, 255, 255))
                            bg.paste(save_img, mask=save_img.split()[-1])
                            save_img = bg
                        save_kwargs = {"optimize": True}
                        if output_format.upper() == "JPEG":
                            save_kwargs.update({"quality": 95})
                        save_img.save(out_path, format=output_format.upper(), **save_kwargs)
                        saved_path = str(out_path)
            except Exception as e:
                logger.warning(f"Failed to save batch output {idx}: {e}")
            output_file_paths.append(saved_path)

        # Encode results to base64
        results_base64: List[str] = []
        for result_image in result_images:
            result_base64 = image_processor.encode_image_to_base64(
                result_image,
                format=output_format
            )
            results_base64.append(result_base64)

        # Build JSON response
        response = {
            "success": True,
            "message": f"Batch background removal completed using {model_name}",
            "outputs_base64": results_base64,
            "output_file_paths": output_file_paths,
            "model_used": model_name,
            "processing_time_seconds": time.time() - start_time,
            "output_format": output_format,
            "processed_count": len(results_base64),
            "image_sizes": [img.size for img in images],
        }

        return [TextContent(type="text", text=json.dumps(response))]

    except Exception as e:
        logger.error(f"Batch background removal failed: {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "message": f"Batch background removal failed: {str(e)}",
            "model_used": arguments.get("model_name", "ben2-base")
        }))]


async def handle_yolo_segment_objects(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle YOLO object segmentation."""
    start_time = time.time()
    try:
        # Parse arguments
        image_data = arguments["image_data"]
        model_name = arguments.get("model_name", "yolo11m-seg")
        target_classes = arguments.get("target_classes")
        confidence_threshold = arguments.get("confidence_threshold", 0.5)
        combine_masks = arguments.get("combine_masks", True)

        # Load image (from file path or base64)
        image = image_processor.load_image(image_data)

        # Resize if too large
        image = image_processor.resize_image_if_needed(image, max_size=(2048, 2048))

        # Get YOLO model instance
        model = await get_model_instance(model_name)

        # Process image
        result_image = await model.remove_background(
            image,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
            combine_masks=combine_masks
        )

        # Encode result
        result_base64 = image_processor.encode_image_to_base64(result_image, format="PNG")

        # Build JSON response
        classes_info = ""
        if target_classes:
            classes_info = f" for classes: {', '.join(target_classes)}"
        response = {
            "success": True,
            "message": f"Object segmentation completed using {model_name}{classes_info}",
            "output_image_base64": result_base64,
            "model_used": model_name,
            "processing_time_seconds": time.time() - start_time,
            "output_format": "PNG",
            "image_size": image.size,
            "target_classes": target_classes,
        }


        return [TextContent(type="text", text=json.dumps(response))]

    except Exception as e:
        logger.error(f"YOLO segmentation failed: {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "message": f"YOLO segmentation failed: {str(e)}",
            "model_used": arguments.get("model_name", "yolo11m-seg")
        }))]


async def handle_get_available_models(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle get available models request."""
    try:
        # Get model information
        available_models = model_manager.get_available_models()

        # Get system recommendations
        recommendations = hardware_detector.get_recommended_models()

        # Get cached models
        cached_models = model_manager.get_cached_models()

        # Format response
        response = {
            "available_models": available_models,
            "system_recommendations": recommendations,
            "cached_models": cached_models,
            "cache_info": model_manager.get_cache_size(),
        }

        import json
        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    except Exception as e:
        logger.error(f"Get available models failed: {e}")
        return [TextContent(type="text", text=f"Get available models failed: {str(e)}")]


async def handle_get_system_info(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle get system info request."""
    try:
        # Get comprehensive system information
        system_info = hardware_detector.get_system_summary()

        import json
        return [TextContent(
            type="text",
            text=json.dumps(system_info, indent=2)
        )]

    except Exception as e:
        logger.error(f"Get system info failed: {e}")
        return [TextContent(type="text", text=f"Get system info failed: {str(e)}")]


async def main():
    """Main entry point for the MCP server."""
    # Set up environment
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    logger.info("Starting Transparent Background MCP Server")
    logger.info(f"System info: {hardware_detector.system_info}")
    logger.info(f"GPU available: {hardware_detector.gpu_info['available']}")

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="transparent-background-mcp",
                server_version="0.1.0",
                capabilities=ServerCapabilities(
                    tools=ToolsCapability()
                ),
            ),
        )





def cli_main():
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
