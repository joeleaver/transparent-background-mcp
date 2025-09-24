"""Tests for model management utilities."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from transparent_background_mcp.utils.model_manager import ModelManager


class TestModelManager:
    """Test model management functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        manager = ModelManager()
        
        assert manager.cache_dir is not None
        assert manager.cache_dir.exists()
        assert "transparent-background-mcp" in str(manager.cache_dir)
    
    def test_init_custom_cache_dir(self, temp_cache_dir):
        """Test initialization with custom cache directory."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        assert str(manager.cache_dir) == temp_cache_dir
        assert manager.cache_dir.exists()
    
    def test_get_available_models(self):
        """Test getting available models information."""
        manager = ModelManager()
        models = manager.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check for expected models
        expected_models = [
            "ben2-base", "inspyrenet-base", 
            "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"
        ]
        
        for model_name in expected_models:
            assert model_name in models
            
            model_info = models[model_name]
            assert "name" in model_info
            assert "description" in model_info
            assert "type" in model_info
            assert "size_mb" in model_info
            assert "vram_requirement_gb" in model_info
            assert "performance" in model_info
    
    def test_get_model_info_valid(self):
        """Test getting information for a valid model."""
        manager = ModelManager()
        model_info = manager.get_model_info("ben2-base")
        
        assert model_info is not None
        assert model_info["name"] == "BEN2 Base"
        assert model_info["type"] == "background_removal"
        assert isinstance(model_info["size_mb"], int)
        assert isinstance(model_info["vram_requirement_gb"], float)
    
    def test_get_model_info_invalid(self):
        """Test getting information for an invalid model."""
        manager = ModelManager()
        model_info = manager.get_model_info("nonexistent-model")
        
        assert model_info is None
    
    def test_is_model_cached_not_cached(self, temp_cache_dir):
        """Test checking if model is cached when it's not."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        is_cached = manager.is_model_cached("ben2-base")
        
        assert is_cached is False
    
    def test_is_model_cached_cached(self, temp_cache_dir):
        """Test checking if model is cached when it is."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create fake model directory and file
        model_dir = Path(temp_cache_dir) / "ben2-base"
        model_dir.mkdir()
        (model_dir / "model.pt").touch()
        
        is_cached = manager.is_model_cached("ben2-base")
        
        assert is_cached is True
    
    def test_is_model_cached_invalid_model(self, temp_cache_dir):
        """Test checking cache for invalid model."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        is_cached = manager.is_model_cached("nonexistent-model")
        
        assert is_cached is False
    
    def test_get_cached_models_empty(self, temp_cache_dir):
        """Test getting cached models when none are cached."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        cached_models = manager.get_cached_models()
        
        assert isinstance(cached_models, list)
        assert len(cached_models) == 0
    
    def test_get_cached_models_with_cached(self, temp_cache_dir):
        """Test getting cached models when some are cached."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create fake cached models
        for model_name in ["ben2-base", "inspyrenet-base"]:
            model_dir = Path(temp_cache_dir) / model_name
            model_dir.mkdir()
            (model_dir / "model.pt").touch()
        
        cached_models = manager.get_cached_models()
        
        assert isinstance(cached_models, list)
        assert len(cached_models) == 2
        assert "ben2-base" in cached_models
        assert "inspyrenet-base" in cached_models
    
    def test_get_cache_size_empty(self, temp_cache_dir):
        """Test getting cache size when empty."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        cache_info = manager.get_cache_size()
        
        assert isinstance(cache_info, dict)
        assert "total_size_mb" in cache_info
        assert "file_count" in cache_info
        assert "cache_dir" in cache_info
        
        assert cache_info["total_size_mb"] == 0.0
        assert cache_info["file_count"] == 0
        assert cache_info["cache_dir"] == temp_cache_dir
    
    def test_get_cache_size_with_files(self, temp_cache_dir):
        """Test getting cache size with files."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create fake model files
        model_dir = Path(temp_cache_dir) / "ben2-base"
        model_dir.mkdir()
        
        # Create a file with known size
        model_file = model_dir / "model.pt"
        model_file.write_bytes(b"0" * 1024)  # 1KB file
        
        cache_info = manager.get_cache_size()
        
        assert cache_info["total_size_mb"] > 0
        assert cache_info["file_count"] == 1
    
    def test_clear_cache_specific_model(self, temp_cache_dir):
        """Test clearing cache for specific model."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create fake model directory
        model_dir = Path(temp_cache_dir) / "ben2-base"
        model_dir.mkdir()
        (model_dir / "model.pt").touch()
        
        # Verify it exists
        assert model_dir.exists()
        
        # Clear specific model
        result = manager.clear_cache("ben2-base")
        
        assert result["cleared"] == "ben2-base"
        assert result["status"] == "success"
        assert not model_dir.exists()
    
    def test_clear_cache_nonexistent_model(self, temp_cache_dir):
        """Test clearing cache for nonexistent model."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        result = manager.clear_cache("nonexistent-model")
        
        assert result["cleared"] == "nonexistent-model"
        assert result["status"] == "not_found"
    
    def test_clear_cache_all(self, temp_cache_dir):
        """Test clearing all cache."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        
        # Create fake model directories
        for model_name in ["ben2-base", "inspyrenet-base"]:
            model_dir = Path(temp_cache_dir) / model_name
            model_dir.mkdir()
            (model_dir / "model.pt").touch()
        
        # Verify they exist
        assert (Path(temp_cache_dir) / "ben2-base").exists()
        assert (Path(temp_cache_dir) / "inspyrenet-base").exists()
        
        # Clear all cache
        result = manager.clear_cache()
        
        assert result["cleared"] == "all"
        assert result["status"] == "success"
        
        # Cache directory should still exist but be empty
        assert Path(temp_cache_dir).exists()
        assert len(list(Path(temp_cache_dir).iterdir())) == 0
