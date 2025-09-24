"""Model management and caching utilities."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloading, caching, and metadata."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for model cache. If None, uses default.
        """
        if cache_dir is None:
            cache_dir = os.getenv(
                "MODEL_CACHE_DIR", 
                str(Path.home() / ".cache" / "transparent-background-mcp")
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def get_available_models(self) -> Dict[str, Dict[str, any]]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary of model information
        """
        return {
            "ben2-base": {
                "name": "BEN2 Base",
                "description": "Latest state-of-the-art background removal with superior hair matting",
                "type": "background_removal",
                "size_mb": 213,
                "vram_requirement_gb": 3.5,
                "performance": "Excellent",
                "release_date": "2025-01",
                "paper": "https://arxiv.org/abs/2501.xxxxx",
                "huggingface_repo": "PeterL1n/BEN2",
                "supported_formats": ["PNG", "JPEG"],
                "batch_processing": True,
            },

            "yolo11l-seg": {
                "name": "YOLO11 Large Segmentation",
                "description": "Large YOLO11 model balancing performance and speed",
                "type": "segmentation",
                "size_mb": 87,
                "vram_requirement_gb": 2.2,
                "performance": "Very Good",
                "release_date": "2024-10",
                "classes": 80,
                "supported_formats": ["PNG", "JPEG"],
                "batch_processing": True,
            },

            "yolo11s-seg": {
                "name": "YOLO11 Small Segmentation",
                "description": "Small YOLO11 model for faster processing",
                "type": "segmentation",
                "size_mb": 22,
                "vram_requirement_gb": 1.2,
                "performance": "Good",
                "release_date": "2024-10",
                "classes": 80,
                "supported_formats": ["PNG", "JPEG"],
                "batch_processing": True,
            },

            "inspyrenet-base": {
                "name": "InSPyReNet Base",
                "description": "Stable, high-quality background removal model",
                "type": "background_removal",
                "size_mb": 65,
                "vram_requirement_gb": 2.0,
                "performance": "Very Good",
                "release_date": "2022-10",
                "paper": "https://arxiv.org/abs/2210.09292",
                "supported_formats": ["PNG", "JPEG"],
                "batch_processing": True,
            },
        }
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        models = self.get_available_models()
        return models.get(model_name)
    
    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if a model is already cached locally.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is cached
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False
        
        # Check for common model file patterns
        model_dir = self.cache_dir / model_name
        if not model_dir.exists():
            return False
        
        # Look for model files
        model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
        return len(model_files) > 0
    
    def get_cached_models(self) -> List[str]:
        """
        Get list of cached model names.
        
        Returns:
            List of cached model names
        """
        cached_models = []
        available_models = self.get_available_models()
        
        for model_name in available_models.keys():
            if self.is_model_cached(model_name):
                cached_models.append(model_name)
        
        return cached_models
    
    def get_cache_size(self) -> Dict[str, any]:
        """
        Get cache directory size information.
        
        Returns:
            Cache size information
        """
        total_size = 0
        file_count = 0
        
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "cache_dir": str(self.cache_dir),
        }
    
    def clear_cache(self, model_name: Optional[str] = None) -> Dict[str, any]:
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to clear, or None for all models
            
        Returns:
            Information about cleared cache
        """
        if model_name:
            # Clear specific model
            model_dir = self.cache_dir / model_name
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Cleared cache for model: {model_name}")
                return {"cleared": model_name, "status": "success"}
            else:
                return {"cleared": model_name, "status": "not_found"}
        else:
            # Clear all cache
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all model cache")
                return {"cleared": "all", "status": "success"}
            else:
                return {"cleared": "all", "status": "not_found"}
