#!/usr/bin/env python3
"""
Test script to test all available models with the improved image loading functionality.
This script demonstrates that the MCP functions can now accept file paths directly.
"""

import sys
import os
import time
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transparent_background_mcp.utils.image_utils import ImageProcessor
from transparent_background_mcp.utils.model_manager import ModelManager
from transparent_background_mcp.utils.hardware import HardwareDetector
from transparent_background_mcp.models import BEN2Model, InSPyReNetModel, YOLOModel

async def test_model(model_class, model_name, image_path, output_dir):
    """Test a specific model with the given image."""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        # Load image using the improved method
        image = ImageProcessor.load_image(image_path)
        print(f"   ✅ Loaded image: {image.size}, {image.mode}")
        
        # Initialize model
        model = model_class(model_name)
        
        # Load model
        start_time = time.time()
        success = await model.load_model()
        load_time = time.time() - start_time
        
        if not success:
            print(f"   ❌ Failed to load model")
            return None
            
        print(f"   ✅ Model loaded in {load_time:.2f}s")
        
        # Process image
        start_time = time.time()
        result_image = await model.remove_background(image, confidence_threshold=0.5)
        process_time = time.time() - start_time
        
        print(f"   ✅ Background removed in {process_time:.2f}s")
        
        # Save result
        output_path = output_dir / f"{model_name}_result.png"
        result_image.save(output_path, "PNG")
        print(f"   ✅ Result saved to: {output_path}")
        
        # Get model info
        model_info = model.get_model_info()
        
        return {
            "model_name": model_name,
            "load_time": load_time,
            "process_time": process_time,
            "output_path": str(output_path),
            "model_info": model_info,
            "success": True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "model_name": model_name,
            "error": str(e),
            "success": False
        }

async def main():
    """Main test function."""
    print("🚀 Testing all transparent-background-mcp models with file path input")
    print("=" * 70)
    
    # Test image path
    image_path = r"C:\Users\joe\OneDrive\Desktop\da9f2bb6-d701-46c6-b691-84e5f7110fb6.jpg"
    
    # Create output directory
    output_dir = Path("model_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Test models
    models_to_test = [
        (BEN2Model, "ben2-base"),
        (InSPyReNetModel, "inspyrenet-base"),
        (YOLOModel, "yolo11s-seg"),
        (YOLOModel, "yolo11l-seg"),
    ]
    
    results = []
    
    for model_class, model_name in models_to_test:
        result = await test_model(model_class, model_name, image_path, output_dir)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    successful_models = [r for r in results if r.get("success", False)]
    failed_models = [r for r in results if not r.get("success", False)]
    
    print(f"✅ Successful models: {len(successful_models)}")
    print(f"❌ Failed models: {len(failed_models)}")
    
    if successful_models:
        print("\n🏆 Successful Models:")
        for result in successful_models:
            print(f"   • {result['model_name']}: Load {result['load_time']:.2f}s, Process {result['process_time']:.2f}s")
    
    if failed_models:
        print("\n💥 Failed Models:")
        for result in failed_models:
            print(f"   • {result['model_name']}: {result['error']}")
    
    print(f"\n📁 Results saved in: {output_dir.absolute()}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(main())
