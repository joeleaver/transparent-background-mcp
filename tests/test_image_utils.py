"""Tests for image processing utilities."""

import base64
import io
import pytest
from PIL import Image

from transparent_background_mcp.utils.image_utils import ImageProcessor


class TestImageProcessor:
    """Test image processing functionality."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        image = Image.new('RGB', (100, 100), color='red')
        return image
    
    @pytest.fixture
    def sample_rgba_image(self):
        """Create a sample RGBA image for testing."""
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        return image
    
    @pytest.fixture
    def sample_base64_image(self, sample_image):
        """Create base64 encoded image data."""
        buffer = io.BytesIO()
        sample_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def test_decode_base64_image_valid(self, sample_base64_image):
        """Test decoding valid base64 image data."""
        image = ImageProcessor.decode_base64_image(sample_base64_image)
        
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)
        assert image.mode in ('RGB', 'RGBA')
    
    def test_decode_base64_image_with_data_url(self, sample_base64_image):
        """Test decoding base64 image with data URL prefix."""
        data_url = f"data:image/png;base64,{sample_base64_image}"
        image = ImageProcessor.decode_base64_image(data_url)
        
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)
    
    def test_decode_base64_image_invalid(self):
        """Test decoding invalid base64 data."""
        with pytest.raises(ValueError):
            ImageProcessor.decode_base64_image("invalid_base64_data")
    
    def test_encode_image_to_base64_png(self, sample_image):
        """Test encoding image to base64 PNG."""
        base64_data = ImageProcessor.encode_image_to_base64(sample_image, format="PNG")
        
        assert isinstance(base64_data, str)
        assert len(base64_data) > 0
        
        # Verify we can decode it back
        decoded_image = ImageProcessor.decode_base64_image(base64_data)
        assert decoded_image.size == sample_image.size
    
    def test_encode_image_to_base64_jpeg(self, sample_rgba_image):
        """Test encoding RGBA image to base64 JPEG (should convert to RGB)."""
        base64_data = ImageProcessor.encode_image_to_base64(sample_rgba_image, format="JPEG", quality=90)
        
        assert isinstance(base64_data, str)
        assert len(base64_data) > 0
        
        # Verify we can decode it back
        decoded_image = ImageProcessor.decode_base64_image(base64_data)
        assert decoded_image.size == sample_rgba_image.size
        assert decoded_image.mode == 'RGB'  # Should be converted from RGBA
    
    def test_validate_image_size_valid(self, sample_image):
        """Test image size validation with valid size."""
        is_valid = ImageProcessor.validate_image_size(sample_image, max_size=(200, 200))
        assert is_valid is True
    
    def test_validate_image_size_invalid(self, sample_image):
        """Test image size validation with invalid size."""
        is_valid = ImageProcessor.validate_image_size(sample_image, max_size=(50, 50))
        assert is_valid is False
    
    def test_resize_image_if_needed_no_resize(self, sample_image):
        """Test resize when image is within limits."""
        resized = ImageProcessor.resize_image_if_needed(sample_image, max_size=(200, 200))
        
        assert resized.size == sample_image.size
        assert resized is sample_image  # Should return same object
    
    def test_resize_image_if_needed_with_resize(self, sample_image):
        """Test resize when image exceeds limits."""
        resized = ImageProcessor.resize_image_if_needed(sample_image, max_size=(50, 50))
        
        assert resized.size == (50, 50)
        assert resized is not sample_image  # Should return new object
    
    def test_resize_image_maintain_aspect_ratio(self):
        """Test resize maintaining aspect ratio."""
        # Create rectangular image
        image = Image.new('RGB', (200, 100), color='blue')
        resized = ImageProcessor.resize_image_if_needed(
            image, 
            max_size=(100, 100), 
            maintain_aspect_ratio=True
        )
        
        # Should maintain 2:1 aspect ratio
        assert resized.size == (100, 50)
    
    def test_resize_image_no_aspect_ratio(self):
        """Test resize without maintaining aspect ratio."""
        # Create rectangular image
        image = Image.new('RGB', (200, 100), color='blue')
        resized = ImageProcessor.resize_image_if_needed(
            image, 
            max_size=(100, 100), 
            maintain_aspect_ratio=False
        )
        
        # Should fit exactly to max size
        assert resized.size == (100, 100)
    
    def test_create_transparency_mask(self, sample_rgba_image):
        """Test creating transparency mask from RGBA image."""
        mask = ImageProcessor.create_transparency_mask(sample_rgba_image, threshold=100)
        
        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'  # Grayscale
        assert mask.size == sample_rgba_image.size
    
    def test_create_transparency_mask_rgb_input(self, sample_image):
        """Test creating transparency mask from RGB image (should convert to RGBA)."""
        mask = ImageProcessor.create_transparency_mask(sample_image, threshold=100)
        
        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'  # Grayscale
        assert mask.size == sample_image.size
    
    def test_apply_transparency_mask(self, sample_image):
        """Test applying transparency mask to image."""
        # Create a simple mask
        mask = Image.new('L', sample_image.size, color=128)  # 50% transparency
        
        result = ImageProcessor.apply_transparency_mask(sample_image, mask)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == sample_image.size
        
        # Check that alpha channel was applied
        alpha_channel = result.split()[-1]
        assert alpha_channel.getpixel((0, 0)) == 128
    
    def test_apply_transparency_mask_rgba_input(self, sample_rgba_image):
        """Test applying transparency mask to RGBA image."""
        # Create a simple mask
        mask = Image.new('L', sample_rgba_image.size, color=255)  # Full opacity
        
        result = ImageProcessor.apply_transparency_mask(sample_rgba_image, mask)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == sample_rgba_image.size
    
    def test_apply_transparency_mask_color_mask(self, sample_image):
        """Test applying color mask (should convert to grayscale)."""
        # Create a color mask
        mask = Image.new('RGB', sample_image.size, color=(128, 128, 128))
        
        result = ImageProcessor.apply_transparency_mask(sample_image, mask)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == sample_image.size
