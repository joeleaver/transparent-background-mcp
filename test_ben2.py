#!/usr/bin/env python3
"""
Test BEN2 model specifically after fixing dependencies.
"""

import asyncio
import base64
import json
from PIL import Image
import io
from transparent_background_mcp.server import handle_call_tool


async def test_ben2():
    # Load the test image
    image_path = r'C:\Users\joe\OneDrive\Desktop\da9f2bb6-d701-46c6-b691-84e5f7110fb6.jpg'
    
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f'📸 Loaded image: {img.size}')
    
    print('🤖 Testing BEN2 - State-of-the-art...')
    
    try:
        result = await handle_call_tool('remove_background', {
            'image_data': image_base64,
            'model_name': 'ben2-base',
            'output_format': 'PNG'
        })
        
        if result and len(result) > 0:
            response_data = json.loads(result[0].text)
            
            if response_data.get('success'):
                # Save the result
                output_base64 = response_data['output_image_base64']
                output_image_data = base64.b64decode(output_base64)
                
                output_path = 'test_output_ben2_base.png'
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
                
                processing_time = response_data.get('processing_time_seconds', 0)
                print(f'   ✅ Success: {processing_time:.2f}s')
                print(f'   📁 Saved: {output_path}')
                print(f'   🎨 Transparency: {transparent_pixels}')
                print(f'   📏 Output size: {output_img.size}')
                print(f'   🎭 Output mode: {output_img.mode}')
                
                if transparent_pixels:
                    print('🎉 BEN2 is now working correctly!')
                    
                    # Compare with other models
                    print('\n📊 BEN2 vs Other Models:')
                    print('   BEN2 (State-of-the-art): Latest technology, best quality')
                    print('   YOLO11 (Object-specific): Fast, good for specific objects')
                    print('   InSPyReNet (Stable): Reliable, proven performance')
                else:
                    print('⚠️  BEN2 processed but no transparency detected')
            else:
                error_msg = response_data.get('message', 'Unknown error')
                print(f'   ❌ Failed: {error_msg}')
        else:
            print('   ❌ No response received')
    
    except Exception as e:
        print(f'   ❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ben2())
