"""Image processing utilities for the transparent background MCP server."""

import base64
import io
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image encoding, decoding, and basic processing operations."""
    
    @staticmethod
    def decode_base64_image(base64_data: str) -> Image.Image:
        """
        Decode base64 image data to PIL Image.
        
        Args:
            base64_data: Base64 encoded image data (with or without data URL prefix)
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If image data is invalid
        """
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            # Create PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if image.mode not in ('RGB', 'RGBA'):
                if image.mode == 'P' and 'transparency' in image.info:
                    image = image.convert('RGBA')
                else:
                    image = image.convert('RGB')
                    
            logger.debug(f"Decoded image: {image.size}, mode: {image.mode}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    @staticmethod
    def encode_image_to_base64(
        image: Image.Image, 
        format: str = "PNG", 
        quality: Optional[int] = None
    ) -> str:
        """
        Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Output format (PNG, JPEG, WEBP)
            quality: JPEG quality (1-100), ignored for PNG
            
        Returns:
            Base64 encoded image string
        """
        try:
            buffer = io.BytesIO()
            
            # Handle format-specific options
            save_kwargs = {"format": format.upper()}
            
            if format.upper() == "JPEG":
                # Convert RGBA to RGB for JPEG
                if image.mode == "RGBA":
                    # Create white background
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  # Use alpha as mask
                    image = background
                
                if quality is not None:
                    save_kwargs["quality"] = quality
                    save_kwargs["optimize"] = True
            
            elif format.upper() == "PNG":
                # Ensure RGBA for PNG with transparency
                if image.mode == "RGB":
                    image = image.convert("RGBA")
                save_kwargs["optimize"] = True
            
            image.save(buffer, **save_kwargs)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.debug(f"Encoded image: {image.size}, format: {format}")
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            raise ValueError(f"Image encoding failed: {e}")
    
    @staticmethod
    def validate_image_size(image: Image.Image, max_size: Tuple[int, int] = (4096, 4096)) -> bool:
        """
        Validate image dimensions.
        
        Args:
            image: PIL Image object
            max_size: Maximum allowed (width, height)
            
        Returns:
            True if image size is valid
        """
        width, height = image.size
        max_width, max_height = max_size
        
        if width > max_width or height > max_height:
            logger.warning(f"Image size {image.size} exceeds maximum {max_size}")
            return False
            
        return True
    
    @staticmethod
    def resize_image_if_needed(
        image: Image.Image, 
        max_size: Tuple[int, int] = (2048, 2048),
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resize image if it exceeds maximum dimensions.
        
        Args:
            image: PIL Image object
            max_size: Maximum allowed (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image object
        """
        width, height = image.size
        max_width, max_height = max_size
        
        if width <= max_width and height <= max_height:
            return image
        
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale_factor = min(max_width / width, max_height / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = min(width, max_width)
            new_height = min(height, max_height)
        
        logger.info(f"Resizing image from {image.size} to ({new_width}, {new_height})")
        
        # Use high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image
    
    @staticmethod
    def create_transparency_mask(image: Image.Image, threshold: int = 10) -> Image.Image:
        """
        Create a transparency mask from an image.
        
        Args:
            image: PIL Image object
            threshold: Threshold for transparency (0-255)
            
        Returns:
            PIL Image mask
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Extract alpha channel
        alpha = image.split()[-1]
        
        # Create binary mask
        mask = alpha.point(lambda x: 255 if x > threshold else 0, mode='L')
        
        return mask
    
    @staticmethod
    def apply_transparency_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Apply transparency mask to an image.
        
        Args:
            image: PIL Image object
            mask: PIL Image mask (grayscale)
            
        Returns:
            PIL Image with applied transparency
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Ensure mask is grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Apply mask as alpha channel
        image.putalpha(mask)
        
        return image
