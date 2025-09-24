"""Tests for hardware detection utilities."""

import pytest
from unittest.mock import Mock, patch

from transparent_background_mcp.utils.hardware import HardwareDetector


class TestHardwareDetector:
    """Test hardware detection functionality."""
    
    def test_init(self):
        """Test hardware detector initialization."""
        detector = HardwareDetector()
        
        assert detector.system_info is not None
        assert detector.gpu_info is not None
        assert "platform" in detector.system_info
        assert "memory_gb" in detector.system_info
        assert "available" in detector.gpu_info
    
    def test_system_detection(self):
        """Test system information detection."""
        detector = HardwareDetector()
        system_info = detector._detect_system()
        
        assert "platform" in system_info
        assert "architecture" in system_info
        assert "cpu_count" in system_info
        assert "memory_gb" in system_info
        assert "python_version" in system_info
        
        assert isinstance(system_info["cpu_count"], int)
        assert isinstance(system_info["memory_gb"], float)
        assert system_info["cpu_count"] > 0
        assert system_info["memory_gb"] > 0
    
    @patch('transparent_background_mcp.utils.hardware.TORCH_AVAILABLE', False)
    def test_gpu_detection_no_torch(self):
        """Test GPU detection when PyTorch is not available."""
        detector = HardwareDetector()
        gpu_info = detector._detect_gpu()
        
        assert gpu_info["available"] is False
        assert gpu_info["device_count"] == 0
        assert gpu_info["devices"] == []
        assert gpu_info["total_vram_gb"] == 0
    
    @patch('transparent_background_mcp.utils.hardware.TORCH_AVAILABLE', True)
    @patch('transparent_background_mcp.utils.hardware.torch')
    def test_gpu_detection_no_cuda(self, mock_torch):
        """Test GPU detection when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        detector = HardwareDetector()
        gpu_info = detector._detect_gpu()
        
        assert gpu_info["available"] is False
        assert gpu_info["device_count"] == 0
        assert gpu_info["devices"] == []
        assert gpu_info["total_vram_gb"] == 0
    
    @patch('transparent_background_mcp.utils.hardware.TORCH_AVAILABLE', True)
    @patch('transparent_background_mcp.utils.hardware.torch')
    def test_gpu_detection_with_cuda(self, mock_torch):
        """Test GPU detection when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        # Mock device properties
        mock_props = Mock()
        mock_props.name = "NVIDIA GeForce RTX 3080"
        mock_props.total_memory = 10 * 1024**3  # 10GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        detector = HardwareDetector()
        gpu_info = detector._detect_gpu()
        
        assert gpu_info["available"] is True
        assert gpu_info["device_count"] == 1
        assert len(gpu_info["devices"]) == 1
        assert gpu_info["total_vram_gb"] == 10.0
        
        device = gpu_info["devices"][0]
        assert device["name"] == "NVIDIA GeForce RTX 3080"
        assert device["vram_gb"] == 10.0
        assert device["compute_capability"] == "8.6"
    
    def test_model_recommendations_high_vram(self):
        """Test model recommendations for high VRAM systems."""
        detector = HardwareDetector()
        detector.gpu_info = {
            "available": True,
            "total_vram_gb": 12.0,
            "device_count": 1,
            "devices": []
        }
        
        recommendations = detector.get_recommended_models()
        
        assert len(recommendations) >= 2
        assert any(rec["model"] == "ben2-base" for rec in recommendations)
        assert any(rec["model"].startswith("yolo11") for rec in recommendations)
        
        # Check that high-end models are recommended
        ben2_rec = next(rec for rec in recommendations if rec["model"] == "ben2-base")
        assert ben2_rec["priority"] == 1
        assert ben2_rec["performance"] == "Excellent"
    
    def test_model_recommendations_low_vram(self):
        """Test model recommendations for low VRAM systems."""
        detector = HardwareDetector()
        detector.gpu_info = {
            "available": True,
            "total_vram_gb": 2.0,
            "device_count": 1,
            "devices": []
        }
        
        recommendations = detector.get_recommended_models()
        
        assert len(recommendations) >= 2
        # Should recommend smaller models for low VRAM
        assert any(rec["model"] == "inspyrenet-base" for rec in recommendations)
        assert any("yolo11" in rec["model"] for rec in recommendations)
    
    def test_model_recommendations_cpu_only(self):
        """Test model recommendations for CPU-only systems."""
        detector = HardwareDetector()
        detector.gpu_info = {
            "available": False,
            "total_vram_gb": 0,
            "device_count": 0,
            "devices": []
        }
        
        recommendations = detector.get_recommended_models()
        
        assert len(recommendations) >= 2
        # Should recommend CPU-friendly models
        assert any(rec["model"] == "inspyrenet-base" for rec in recommendations)
        assert any(rec["model"] == "yolo11n-seg" for rec in recommendations)
        
        # Check performance expectations
        for rec in recommendations:
            assert "CPU" in rec["performance"]
    
    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        detector = HardwareDetector()
        
        # Test with GPU
        detector.gpu_info = {"available": True, "total_vram_gb": 8.0}
        batch_size = detector.get_optimal_batch_size("inspyrenet-base")
        assert isinstance(batch_size, int)
        assert batch_size >= 1
        assert batch_size <= 8
        
        # Test with CPU
        detector.gpu_info = {"available": False, "total_vram_gb": 0}
        batch_size = detector.get_optimal_batch_size("inspyrenet-base")
        assert batch_size == 1
    
    def test_system_summary(self):
        """Test system summary generation."""
        detector = HardwareDetector()
        summary = detector.get_system_summary()
        
        assert "system" in summary
        assert "gpu" in summary
        assert "recommendations" in summary
        assert "torch_available" in summary
        
        assert isinstance(summary["recommendations"], list)
        assert isinstance(summary["torch_available"], bool)
