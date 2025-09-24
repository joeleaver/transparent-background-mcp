#!/usr/bin/env python3
"""
Basic usage example for the Transparent Background MCP Server.

This script demonstrates how to use the MCP server programmatically
for background removal tasks.
"""

import asyncio
import base64
import json
from pathlib import Path
from PIL import Image
import io

# Import the server components
from transparent_background_mcp.models import BEN2Model, InSPyReNetModel, YOLOModel
from transparent_background_mcp.utils import HardwareDetector, ImageProcessor, ModelManager


async def main():
    """Demonstrate basic usage of the transparent background MCP server."""
    
    print("🎨 Transparent Background MCP Server - Basic Usage Example")
    print("=" * 60)
    
    # Initialize components
    hardware_detector = HardwareDetector()
    model_manager = ModelManager()
    image_processor = ImageProcessor()
    
    # Display system information
    print("\n📊 System Information:")
    system_info = hardware_detector.get_system_summary()
    print(f"Platform: {system_info['system']['platform']}")
    print(f"CPU Cores: {system_info['system']['cpu_count']}")
    print(f"Memory: {system_info['system']['memory_gb']:.1f} GB")
    print(f"GPU Available: {system_info['gpu']['available']}")
    
    if system_info['gpu']['available']:
        for i, device in enumerate(system_info['gpu']['devices']):
            print(f"GPU {i}: {device['name']} ({device['vram_gb']:.1f} GB VRAM)")
    
    # Display model recommendations
    print("\n🤖 Recommended Models:")
    recommendations = hardware_detector.get_recommended_models()
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec['model']} - {rec['reason']} (Performance: {rec['performance']})")
    
    # Create a sample image for demonstration
    print("\n🖼️  Creating sample image...")
    sample_image = Image.new('RGB', (400, 300), color=(70, 130, 180))  # Steel blue background
    
    # Add a simple shape (circle) to the image
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sample_image)
    draw.ellipse([100, 75, 300, 225], fill=(255, 69, 0))  # Orange circle
    
    # Save sample image
    sample_path = Path("sample_input.png")
    sample_image.save(sample_path)
    print(f"Sample image saved: {sample_path}")
    
    # Convert to base64 for processing
    buffer = io.BytesIO()
    sample_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test different models
    models_to_test = [
        ("inspyrenet-base", InSPyReNetModel),
        ("yolo11m-seg", YOLOModel),
    ]
    
    # Only test BEN2 if we have good hardware
    if system_info['gpu']['available'] and system_info['gpu']['total_vram_gb'] >= 4:
        models_to_test.insert(0, ("ben2-base", BEN2Model))
    
    print(f"\n🔄 Testing {len(models_to_test)} models...")
    
    for model_name, model_class in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            # Initialize model
            model = model_class(model_name)
            
            # Load model
            print("Loading model...")
            success = await model.load_model()
            
            if not success:
                print(f"❌ Failed to load {model_name}")
                continue
            
            print("✅ Model loaded successfully")
            
            # Get model info
            model_info = model.get_model_info()
            print(f"Model: {model_info['name']}")
            print(f"Type: {model_info['type']}")
            print(f"Performance: {model_info['performance']}")
            
            # Process image
            print("Processing image...")
            decoded_image = image_processor.decode_base64_image(image_base64)
            
            if model_name.startswith("yolo"):
                # For YOLO, we can specify target classes
                result_image = await model.remove_background(
                    decoded_image,
                    target_classes=None,  # Segment all detected objects
                    confidence_threshold=0.5
                )
            else:
                # For background removal models
                result_image = await model.remove_background(
                    decoded_image,
                    confidence_threshold=0.5
                )
            
            # Save result
            output_path = Path(f"output_{model_name}.png")
            result_image.save(output_path)
            print(f"✅ Result saved: {output_path}")
            
            # Unload model to free memory
            await model.unload_model()
            print("Model unloaded")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    # Display cache information
    print(f"\n💾 Cache Information:")
    cache_info = model_manager.get_cache_size()
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Total size: {cache_info['total_size_mb']:.2f} MB")
    print(f"Files: {cache_info['file_count']}")
    
    cached_models = model_manager.get_cached_models()
    if cached_models:
        print(f"Cached models: {', '.join(cached_models)}")
    else:
        print("No models cached yet")
    
    print(f"\n✨ Example completed! Check the output images.")
    print(f"Sample input: {sample_path}")
    
    # List output files
    output_files = list(Path(".").glob("output_*.png"))
    if output_files:
        print("Generated outputs:")
        for output_file in output_files:
            print(f"  - {output_file}")
    
    print(f"\n🔧 To use this server with your IDE, add this configuration:")
    print(json.dumps({
        "mcpServers": {
            "transparent-background-mcp": {
                "command": "uvx",
                "args": [
                    "--from", 
                    "git+https://github.com/joeleaver/transparent-background-mcp.git",
                    "transparent-background-mcp-server"
                ],
                "env": {
                    "MODEL_CACHE_DIR": str(model_manager.cache_dir)
                }
            }
        }
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
