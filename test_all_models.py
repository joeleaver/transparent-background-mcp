#!/usr/bin/env python3
"""
Test all available models in the transparent background MCP server.
"""

import asyncio
import base64
import json
from PIL import Image
import io
import time
from transparent_background_mcp.server import handle_call_tool


async def test_models():
    # Load the test image
    image_path = r'C:\Users\joe\OneDrive\Desktop\da9f2bb6-d701-46c6-b691-84e5f7110fb6.jpg'
    
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f'📸 Loaded image: {img.size}')
    
    # Test different models
    models_to_test = [
        ('inspyrenet-base', 'InSPyReNet - Stable & Reliable'),
        ('yolo11n-seg', 'YOLO11 Nano - Fastest'),
        ('yolo11s-seg', 'YOLO11 Small - Balanced'),
        ('ben2-base', 'BEN2 - State-of-the-art')
    ]
    
    results = []
    
    for model_name, description in models_to_test:
        print(f'\n🤖 Testing {description}...')
        
        try:
            result = await handle_call_tool('remove_background', {
                'image_data': image_base64,
                'model_name': model_name,
                'output_format': 'PNG'
            })
            
            if result and len(result) > 0:
                response_data = json.loads(result[0].text)
                
                if response_data.get('success'):
                    # Save the result
                    output_base64 = response_data['output_image_base64']
                    output_image_data = base64.b64decode(output_base64)
                    
                    output_path = f'test_output_{model_name.replace("-", "_")}.png'
                    with open(output_path, 'wb') as f:
                        f.write(output_image_data)
                    
                    # Verify transparency
                    with Image.open(output_path) as output_img:
                        has_transparency = output_img.mode == 'RGBA'
                        if has_transparency:
                            alpha_channel = output_img.split()[-1]
                            min_alpha = min(alpha_channel.getdata())
                            transparent_pixels = min_alpha < 255
                        else:
                            transparent_pixels = False
                    
                    results.append({
                        'model': model_name,
                        'description': description,
                        'success': True,
                        'processing_time': response_data.get('processing_time_seconds', 0),
                        'output_file': output_path,
                        'has_transparency': transparent_pixels,
                        'output_size': output_img.size
                    })
                    
                    print(f'   ✅ Success: {response_data.get("processing_time_seconds", 0):.2f}s')
                    print(f'   📁 Saved: {output_path}')
                    print(f'   🎨 Transparency: {transparent_pixels}')
                else:
                    print(f'   ❌ Failed: {response_data.get("message", "Unknown error")}')
                    results.append({
                        'model': model_name,
                        'description': description,
                        'success': False,
                        'error': response_data.get('message', 'Unknown error')
                    })
        
        except Exception as e:
            print(f'   ❌ Error: {str(e)}')
            results.append({
                'model': model_name,
                'description': description,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f'\n📊 Model Comparison Results:')
    print('=' * 60)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        # Sort by processing time
        successful_results.sort(key=lambda x: x['processing_time'])
        
        print('✅ Successful Models:')
        for result in successful_results:
            print(f'   {result["description"]}:')
            print(f'     ⏱️  Time: {result["processing_time"]:.2f}s')
            print(f'     🎨 Transparency: {result["has_transparency"]}')
            print(f'     📁 File: {result["output_file"]}')
            print()
    
    if failed_results:
        print('❌ Failed Models:')
        for result in failed_results:
            print(f'   {result["description"]}: {result["error"]}')
    
    print(f'Total: {len(successful_results)} successful, {len(failed_results)} failed')


if __name__ == "__main__":
    asyncio.run(test_models())
