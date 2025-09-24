#!/usr/bin/env python3
"""
Test script to verify MCP server integration and background removal functionality.
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from PIL import Image
import io

# Import our MCP server components
from transparent_background_mcp.server import server
from transparent_background_mcp.utils.hardware import HardwareDetector
from transparent_background_mcp.utils.model_manager import ModelManager


async def test_system_info():
    """Test the get_system_info tool."""
    print("🖥️  Testing system info...")

    try:
        # Import the handler function directly
        from transparent_background_mcp.server import handle_call_tool
        result = await handle_call_tool("get_system_info", {})
        print("✅ System info retrieved successfully")
        
        # Parse the result
        if result and len(result) > 0:
            content = result[0].text
            system_info = json.loads(content)
            print(f"   Platform: {system_info.get('platform', 'Unknown')}")
            print(f"   CPU cores: {system_info.get('cpu_count', 'Unknown')}")
            print(f"   Memory: {system_info.get('memory_gb', 'Unknown')} GB")
            print(f"   GPU available: {system_info.get('gpu_available', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False


async def test_available_models():
    """Test the get_available_models tool."""
    print("\n📦 Testing available models...")

    try:
        from transparent_background_mcp.server import handle_call_tool
        result = await handle_call_tool("get_available_models", {})
        print("✅ Available models retrieved successfully")
        
        # Parse the result
        if result and len(result) > 0:
            content = result[0].text
            models_info = json.loads(content)
            print(f"   Total models: {len(models_info.get('models', []))}")
            
            for model in models_info.get('models', [])[:3]:  # Show first 3
                print(f"   - {model.get('name', 'Unknown')}: {model.get('description', 'No description')}")
        
        return True
    except Exception as e:
        print(f"❌ Available models test failed: {e}")
        return False


async def test_background_removal():
    """Test background removal with the test image."""
    print("\n🎨 Testing background removal...")
    
    # Load the base64 image data
    try:
        with open('test_image_base64.txt', 'r') as f:
            image_base64 = f.read().strip()
        
        print(f"   Loaded image data: {len(image_base64)} characters")
        
        # Test with InSPyReNet model (most stable)
        from transparent_background_mcp.server import handle_call_tool
        result = await handle_call_tool("remove_background", {
            "image_data": image_base64,
            "model_name": "inspyrenet-base",
            "output_format": "PNG"
        })
        
        print("✅ Background removal completed successfully")

        # Parse the result
        if result and len(result) > 0:
            content = result[0].text
            print(f"   Raw response: {content[:200]}...")  # Debug output

            if not content.strip():
                print("   Empty response received")
                return False

            try:
                result_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"   JSON decode error: {e}")
                print(f"   Content: {repr(content[:500])}")
                return False
            
            if "output_image_base64" in result_data:
                # Save the result image
                output_base64 = result_data["output_image_base64"]
                output_image_data = base64.b64decode(output_base64)
                
                output_path = "test_output_transparent.png"
                with open(output_path, 'wb') as f:
                    f.write(output_image_data)
                
                print(f"   Output saved to: {output_path}")
                
                # Verify the output image
                with Image.open(output_path) as img:
                    print(f"   Output size: {img.size}")
                    print(f"   Output mode: {img.mode}")
                    print(f"   Has transparency: {img.mode in ['RGBA', 'LA'] or 'transparency' in img.info}")
            
            print(f"   Processing time: {result_data.get('processing_time_seconds', 'Unknown')} seconds")
            print(f"   Model used: {result_data.get('model_used', 'Unknown')}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Test image base64 file not found. Please run the image encoding first.")
        return False
    except Exception as e:
        print(f"❌ Background removal test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Transparent Background MCP Server")
    print("=" * 50)
    
    tests = [
        ("System Info", test_system_info),
        ("Available Models", test_available_models),
        ("Background Removal", test_background_removal),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 All tests passed! The MCP server is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
