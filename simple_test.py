#!/usr/bin/env python3
"""
Simple test to demonstrate the transparent background MCP server functionality.
"""

import asyncio
import base64
import json
from PIL import Image
import io

# Import our MCP server components
from transparent_background_mcp.server import handle_call_tool


async def main():
    print("🎨 Simple Transparent Background MCP Test")
    print("=" * 50)
    
    # Load the test image
    image_path = r'C:\Users\joe\OneDrive\Desktop\da9f2bb6-d701-46c6-b691-84e5f7110fb6.jpg'
    
    print(f"📸 Loading image: {image_path}")
    
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"   Original size: {img.size}")
        print(f"   Original mode: {img.mode}")
        
        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"   Base64 length: {len(image_base64)} characters")
    
    # Test background removal
    print("\n🤖 Removing background...")
    
    result = await handle_call_tool("remove_background", {
        "image_data": image_base64,
        "model_name": "inspyrenet-base",
        "output_format": "PNG"
    })
    
    if result and len(result) > 0:
        response_data = json.loads(result[0].text)
        
        if response_data.get("success"):
            print("✅ Background removal successful!")
            print(f"   Model used: {response_data.get('model_used')}")
            print(f"   Processing time: {response_data.get('processing_time_seconds'):.2f} seconds")
            print(f"   Output format: {response_data.get('output_format')}")
            
            # Save the result
            output_base64 = response_data["output_image_base64"]
            output_image_data = base64.b64decode(output_base64)
            
            output_path = "simple_test_output.png"
            with open(output_path, 'wb') as f:
                f.write(output_image_data)
            
            print(f"   Output saved to: {output_path}")
            
            # Verify the output
            with Image.open(output_path) as output_img:
                print(f"   Output size: {output_img.size}")
                print(f"   Output mode: {output_img.mode}")
                
                if output_img.mode == 'RGBA':
                    alpha_channel = output_img.split()[-1]
                    min_alpha = min(alpha_channel.getdata())
                    max_alpha = max(alpha_channel.getdata())
                    has_transparency = min_alpha < 255
                    print(f"   Has transparent pixels: {has_transparency}")
                    print(f"   Alpha range: {min_alpha} - {max_alpha}")
                    
                    if has_transparency:
                        print("🎉 SUCCESS: Background successfully removed with transparency!")
                    else:
                        print("⚠️  WARNING: No transparent pixels found")
                else:
                    print("⚠️  WARNING: Output image doesn't have alpha channel")
        else:
            print("❌ Background removal failed")
            print(f"   Error: {response_data.get('message', 'Unknown error')}")
    else:
        print("❌ No response from server")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
