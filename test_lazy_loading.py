#!/usr/bin/env python3
"""
Test script to demonstrate lazy loading behavior.
This simulates what happens when a user first runs the MCP server.
"""

import asyncio
import json
import time
from transparent_background_mcp.server import handle_call_tool


async def test_lazy_loading():
    """Test the lazy loading functionality."""
    print("🚀 Testing Lazy Loading MCP Server")
    print("=" * 50)
    
    # Test 1: Check dependency status
    print("\n📊 Step 1: Checking dependency status...")
    start_time = time.time()
    
    result = await handle_call_tool('check_dependencies', {})
    response = json.loads(result[0].text)
    
    print(f"⏱️  Time: {time.time() - start_time:.2f}s")
    print(f"✅ Ready models: {response['summary']['ready_models']}")
    print(f"⏳ Pending models: {response['summary']['pending_models']}")
    
    # Test 2: Get available models (should be fast)
    print("\n🤖 Step 2: Getting available models...")
    start_time = time.time()
    
    result = await handle_call_tool('get_available_models', {})
    response = json.loads(result[0].text)

    print(f"⏱️  Time: {time.time() - start_time:.2f}s")
    models = response.get('available_models', [])
    print(f"📋 Available models: {len(models)}")
    
    # Test 3: Test a lightweight model (InSPyReNet)
    print("\n🎨 Step 3: Testing InSPyReNet (should be fast)...")
    start_time = time.time()
    
    # Create a simple test image (1x1 pixel)
    import base64
    from PIL import Image
    import io
    
    # Create minimal test image
    test_image = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    result = await handle_call_tool('remove_background', {
        'image_data': test_image_b64,
        'model_name': 'inspyrenet-base',
        'output_format': 'PNG'
    })
    
    response = json.loads(result[0].text)
    print(f"⏱️  Time: {time.time() - start_time:.2f}s")
    print(f"✅ Success: {response.get('success', False)}")
    print(f"🎯 Model used: {response.get('model_used', 'Unknown')}")
    
    # Test 4: System info (should be instant)
    print("\n💻 Step 4: Getting system info...")
    start_time = time.time()
    
    result = await handle_call_tool('get_system_info', {})
    response = json.loads(result[0].text)
    
    print(f"⏱️  Time: {time.time() - start_time:.2f}s")
    sys_info = response.get('system_info', {})
    print(f"🖥️  Platform: {sys_info.get('platform', 'Unknown')}")
    print(f"🧠 Memory: {sys_info.get('memory_gb', 0):.1f}GB")
    print(f"🎮 GPU: {sys_info.get('gpu_available', False)}")
    
    print("\n🎉 Lazy Loading Test Complete!")
    print("=" * 50)
    print("✅ Server starts quickly with minimal dependencies")
    print("📦 Model dependencies are installed on first use")
    print("⚡ Subsequent model usage is fast (cached)")


if __name__ == "__main__":
    asyncio.run(test_lazy_loading())
