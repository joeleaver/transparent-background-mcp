#!/usr/bin/env python3
"""
Test script to verify the transparent background MCP server installation.

This script performs basic checks to ensure the server is properly installed
and can be imported and initialized.
"""

import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from transparent_background_mcp import main
        print("✅ Main module imported successfully")
        
        from transparent_background_mcp.server import server
        print("✅ Server module imported successfully")
        
        from transparent_background_mcp.utils import HardwareDetector, ImageProcessor, ModelManager
        print("✅ Utility modules imported successfully")
        
        from transparent_background_mcp.models import BaseBackgroundRemovalModel, BEN2Model, InSPyReNetModel, YOLOModel
        print("✅ Model modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        traceback.print_exc()
        return False


def test_hardware_detection():
    """Test hardware detection functionality."""
    print("\n🖥️  Testing hardware detection...")
    
    try:
        from transparent_background_mcp.utils import HardwareDetector
        
        detector = HardwareDetector()
        system_info = detector.get_system_summary()
        
        print(f"✅ System detected: {system_info['system']['platform']}")
        print(f"✅ CPU cores: {system_info['system']['cpu_count']}")
        print(f"✅ Memory: {system_info['system']['memory_gb']:.1f} GB")
        print(f"✅ GPU available: {system_info['gpu']['available']}")
        
        if system_info['gpu']['available']:
            print(f"✅ GPU devices: {len(system_info['gpu']['devices'])}")
            for i, device in enumerate(system_info['gpu']['devices']):
                print(f"   GPU {i}: {device['name']} ({device['vram_gb']:.1f} GB)")
        
        recommendations = detector.get_recommended_models()
        print(f"✅ Model recommendations: {len(recommendations)} models")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection error: {e}")
        traceback.print_exc()
        return False


def test_model_manager():
    """Test model manager functionality."""
    print("\n📦 Testing model manager...")
    
    try:
        from transparent_background_mcp.utils import ModelManager
        
        manager = ModelManager()
        
        # Test getting available models
        models = manager.get_available_models()
        print(f"✅ Available models: {len(models)}")
        
        # Test specific model info
        ben2_info = manager.get_model_info("ben2-base")
        if ben2_info:
            print(f"✅ BEN2 model info: {ben2_info['name']}")
        
        inspyrenet_info = manager.get_model_info("inspyrenet-base")
        if inspyrenet_info:
            print(f"✅ InSPyReNet model info: {inspyrenet_info['name']}")
        
        yolo_info = manager.get_model_info("yolo11m-seg")
        if yolo_info:
            print(f"✅ YOLO11 model info: {yolo_info['name']}")
        
        # Test cache info
        cache_info = manager.get_cache_size()
        print(f"✅ Cache directory: {cache_info['cache_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model manager error: {e}")
        traceback.print_exc()
        return False


def test_image_processor():
    """Test image processing functionality."""
    print("\n🖼️  Testing image processor...")
    
    try:
        from transparent_background_mcp.utils import ImageProcessor
        from PIL import Image
        import base64
        import io
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Test encoding
        base64_data = ImageProcessor.encode_image_to_base64(test_image, format="PNG")
        print(f"✅ Image encoded to base64: {len(base64_data)} characters")
        
        # Test decoding
        decoded_image = ImageProcessor.decode_base64_image(base64_data)
        print(f"✅ Image decoded from base64: {decoded_image.size}")
        
        # Test validation
        is_valid = ImageProcessor.validate_image_size(test_image, max_size=(200, 200))
        print(f"✅ Image validation: {is_valid}")
        
        # Test resizing
        resized = ImageProcessor.resize_image_if_needed(test_image, max_size=(50, 50))
        print(f"✅ Image resized: {resized.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processor error: {e}")
        traceback.print_exc()
        return False


def test_server_initialization():
    """Test MCP server initialization."""
    print("\n🚀 Testing server initialization...")
    
    try:
        from transparent_background_mcp.server import server, handle_list_tools
        
        print("✅ Server object created successfully")
        
        # Test listing tools (this should work without async)
        print("✅ Server tools can be accessed")
        
        return True
        
    except Exception as e:
        print(f"❌ Server initialization error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🎨 Transparent Background MCP Server - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Model Manager", test_model_manager),
        ("Image Processor", test_image_processor),
        ("Server Initialization", test_server_initialization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The installation appears to be working correctly.")
        print("\n🔧 You can now use the server with your MCP client:")
        print('Add this to your MCP client configuration:')
        print("""
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ]
    }
  }
}
        """)
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        print("You may need to install missing dependencies or fix configuration issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
