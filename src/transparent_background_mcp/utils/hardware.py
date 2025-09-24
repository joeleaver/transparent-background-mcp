"""Hardware detection and system optimization utilities."""

import logging
import platform
import psutil
from typing import Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detects system hardware capabilities and provides optimization recommendations."""
    
    def __init__(self):
        self.system_info = self._detect_system()
        self.gpu_info = self._detect_gpu()
        
    def _detect_system(self) -> Dict[str, any]:
        """Detect basic system information."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version(),
        }
    
    def _detect_gpu(self) -> Dict[str, any]:
        """Detect GPU capabilities."""
        gpu_info = {
            "available": False,
            "device_count": 0,
            "devices": [],
            "total_vram_gb": 0,
        }
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - GPU detection disabled")
            return gpu_info
            
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                vram_gb = round(device_props.total_memory / (1024**3), 2)
                
                device_info = {
                    "index": i,
                    "name": device_props.name,
                    "vram_gb": vram_gb,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
                
                gpu_info["devices"].append(device_info)
                gpu_info["total_vram_gb"] += vram_gb
                
        return gpu_info
    
    def get_recommended_models(self) -> List[Dict[str, any]]:
        """Get model recommendations based on hardware capabilities."""
        recommendations = []
        
        # Memory-based recommendations
        memory_gb = self.system_info["memory_gb"]
        vram_gb = self.gpu_info["total_vram_gb"] if self.gpu_info["available"] else 0
        
        if vram_gb >= 12:
            # High-end GPU setup
            recommendations.extend([
                {
                    "model": "ben2-base",
                    "priority": 1,
                    "reason": "Latest state-of-the-art with excellent hair matting",
                    "performance": "Excellent",
                    "batch_size": 8,
                },
                {
                    "model": "yolo11x-seg",
                    "priority": 2,
                    "reason": "Best YOLO model for object-specific removal",
                    "performance": "Excellent",
                    "batch_size": 6,
                },
            ])
        elif vram_gb >= 8:
            # Mid-range GPU setup
            recommendations.extend([
                {
                    "model": "ben2-base",
                    "priority": 1,
                    "reason": "Latest model with good performance on 8GB VRAM",
                    "performance": "Very Good",
                    "batch_size": 4,
                },
                {
                    "model": "yolo11l-seg",
                    "priority": 2,
                    "reason": "Large YOLO model for object detection",
                    "performance": "Very Good",
                    "batch_size": 4,
                },
            ])
        elif vram_gb >= 4:
            # Entry-level GPU setup
            recommendations.extend([
                {
                    "model": "inspyrenet-base",
                    "priority": 1,
                    "reason": "Stable model optimized for 4GB VRAM",
                    "performance": "Good",
                    "batch_size": 2,
                },
                {
                    "model": "yolo11m-seg",
                    "priority": 2,
                    "reason": "Medium YOLO model for object detection",
                    "performance": "Good",
                    "batch_size": 3,
                },
            ])
        else:
            # CPU-only or low VRAM
            recommendations.extend([
                {
                    "model": "inspyrenet-base",
                    "priority": 1,
                    "reason": "Best CPU performance with reasonable quality",
                    "performance": "Fair (CPU)",
                    "batch_size": 1,
                },
                {
                    "model": "yolo11n-seg",
                    "priority": 2,
                    "reason": "Nano YOLO model for minimal resource usage",
                    "performance": "Fair (CPU)",
                    "batch_size": 1,
                },
            ])
            
        return recommendations
    
    def get_optimal_batch_size(self, model_name: str) -> int:
        """Get optimal batch size for a specific model."""
        vram_gb = self.gpu_info["total_vram_gb"] if self.gpu_info["available"] else 0
        
        # Model-specific VRAM requirements (approximate)
        model_vram_requirements = {
            "ben2-base": 3.5,
            "yolo11x-seg": 2.8,
            "yolo11l-seg": 2.2,
            "yolo11m-seg": 1.8,
            "yolo11s-seg": 1.2,
            "yolo11n-seg": 0.8,
            "inspyrenet-base": 2.0,
        }
        
        if not self.gpu_info["available"]:
            return 1  # CPU processing
            
        model_vram = model_vram_requirements.get(model_name, 2.0)
        
        # Calculate batch size with safety margin
        safety_margin = 0.8  # Use 80% of available VRAM
        available_vram = vram_gb * safety_margin
        
        if available_vram < model_vram:
            return 1  # Minimum batch size
            
        batch_size = int(available_vram / model_vram)
        return min(batch_size, 8)  # Cap at 8 for stability
    
    def get_system_summary(self) -> Dict[str, any]:
        """Get comprehensive system summary."""
        return {
            "system": self.system_info,
            "gpu": self.gpu_info,
            "recommendations": self.get_recommended_models(),
            "torch_available": TORCH_AVAILABLE,
        }
