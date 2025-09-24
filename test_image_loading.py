#!/usr/bin/env python3
"""
Test script to verify the improved image loading functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transparent_background_mcp.utils.image_utils import ImageProcessor

def test_image_loading():
    """Test loading image from file path."""
    image_path = r"C:\Users\joe\OneDrive\Desktop\da9f2bb6-d701-46c6-b691-84e5f7110fb6.jpg"
    
    try:
        # Test the new load_image method
        image = ImageProcessor.load_image(image_path)
        print(f"✅ Successfully loaded image from file path")
        print(f"   Size: {image.size}")
        print(f"   Mode: {image.mode}")
        
        # Test base64 encoding
        base64_data = ImageProcessor.encode_image_to_base64(image)
        print(f"✅ Successfully encoded image to base64")
        print(f"   Base64 length: {len(base64_data)} characters")
        
        # Test loading from base64
        image2 = ImageProcessor.load_image(base64_data)
        print(f"✅ Successfully loaded image from base64")
        print(f"   Size: {image2.size}")
        print(f"   Mode: {image2.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_image_loading()
    if success:
        print("\n🎉 All tests passed! The image loading functionality works correctly.")
    else:
        print("\n💥 Tests failed!")
