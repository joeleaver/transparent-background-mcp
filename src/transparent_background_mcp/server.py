"""
Transparent Background MCP Server

A local AI-powered background removal MCP server using state-of-the-art models.
"""

import asyncio
import json
import logging
import os
import time
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
from .utils.dependency_manager import dependency_manager

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
    model_name: str = Field(default="inspyrenet-base", description="Model to use for background removal")
    output_format: str = Field(default="PNG", description="Output format (PNG, JPEG)")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for detection")


class BatchRemoveBackgroundRequest(BaseModel):
    """Request model for batch background removal."""
    images_data: List[str] = Field(description="List of base64 encoded image data")
    model_name: str = Field(default="inspyrenet-base", description="Model to use for background removal")
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
    """Get or create model instance with lazy imports."""
    if model_name not in model_cache:
        logger.info(f"Creating new model instance: {model_name}")

        # Lazy import models to avoid import errors at startup
        if model_name.startswith("ben2"):
            from .models.ben2_model import BEN2Model
            model_cache[model_name] = BEN2Model(model_name)
        elif model_name.startswith("yolo11"):
            from .models.yolo_model import YOLOModel
            model_cache[model_name] = YOLOModel(model_name)
        elif model_name.startswith("inspyrenet"):
            from .models.inspyrenet_model import InSPyReNetModel
            model_cache[model_name] = InSPyReNetModel(model_name)
        else:
            # Default to InSPyReNet for unknown models
            logger.warning(f"Unknown model {model_name}, defaulting to inspyrenet-base")
            from .models.inspyrenet_model import InSPyReNetModel
            model_cache[model_name] = InSPyReNetModel("inspyrenet-base")

    return model_cache[model_name]


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="remove_background",
            description="Remove background from a single image using AI models",
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
                        "enum": ["ben2-base", "inspyrenet-base", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
                        "default": "inspyrenet-base"
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
            description="Remove background from multiple images using AI models",
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
                        "enum": ["ben2-base", "inspyrenet-base", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
                        "default": "inspyrenet-base"
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
            description="Segment specific objects using YOLO11 models",
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
                        "enum": ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
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
        Tool(
            name="check_dependencies",
            description="Check dependency installation status for all models",
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
        elif name == "check_dependencies":
            return await handle_check_dependencies(arguments)
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
        model_name = arguments.get("model_name", "inspyrenet-base")
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
        
        # Encode result
        result_base64 = image_processor.encode_image_to_base64(
            result_image, 
            format=output_format
        )
        
        # Return JSON response
        response = {
            "success": True,
            "message": f"Background removed successfully using {model_name}",
            "output_image_base64": result_base64,
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
    try:
        # Parse arguments
        images_data = arguments["images_data"]
        model_name = arguments.get("model_name", "inspyrenet-base")
        output_format = arguments.get("output_format", "PNG")
        confidence_threshold = arguments.get("confidence_threshold", 0.5)
        
        if not images_data:
            return [TextContent(type="text", text="No images provided")]
        
        # Load images (from file paths or base64)
        images = []
        for i, image_data in enumerate(images_data):
            try:
                image = image_processor.load_image(image_data)
                image = image_processor.resize_image_if_needed(image, max_size=(2048, 2048))
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to load image {i}: {e}")
                return [TextContent(type="text", text=f"Failed to load image {i}: {str(e)}")]
        
        # Get model instance
        model = await get_model_instance(model_name)
        
        # Process images
        result_images = await model.batch_remove_background(
            images,
            confidence_threshold=confidence_threshold
        )
        
        # Encode results
        results_base64 = []
        for result_image in result_images:
            result_base64 = image_processor.encode_image_to_base64(
                result_image,
                format=output_format
            )
            results_base64.append(result_base64)
        
        return [TextContent(
            type="text",
            text=f"Batch background removal completed using {model_name}. Processed {len(results_base64)} images."
        )]
        
    except Exception as e:
        logger.error(f"Batch background removal failed: {e}")
        return [TextContent(type="text", text=f"Batch background removal failed: {str(e)}")]


async def handle_yolo_segment_objects(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle YOLO object segmentation."""
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

        # Get detected classes info
        classes_info = ""
        if target_classes:
            classes_info = f" for classes: {', '.join(target_classes)}"

        return [TextContent(
            type="text",
            text=f"Object segmentation completed using {model_name}{classes_info}. Result: {len(result_base64)} characters of base64 data."
        )]

    except Exception as e:
        logger.error(f"YOLO segmentation failed: {e}")
        return [TextContent(type="text", text=f"YOLO segmentation failed: {str(e)}")]


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


async def handle_check_dependencies(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle dependency status check."""
    try:
        status = dependency_manager.get_installation_status()

        response = {
            "success": True,
            "message": "Dependency status retrieved successfully",
            "dependencies": status,
            "summary": {
                "total_models": len(status),
                "ready_models": sum(1 for ready in status.values() if ready),
                "pending_models": sum(1 for ready in status.values() if not ready)
            }
        }

        return [TextContent(
            type="text",
            text=json.dumps(response)
        )]

    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return [TextContent(type="text", text=f"Dependency check failed: {str(e)}")]


def cli_main():
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
