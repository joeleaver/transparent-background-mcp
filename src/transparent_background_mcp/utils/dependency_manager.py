"""
Lazy dependency management for background removal models.
Handles on-demand installation of model-specific dependencies.
"""

import importlib
import logging
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages lazy loading of model dependencies."""
    
    def __init__(self):
        self._installed_packages = set()
        self._installation_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def ensure_dependencies(self, model_name: str) -> Tuple[bool, str]:
        """
        Ensure all dependencies for a model are installed.
        
        Args:
            model_name: Name of the model requiring dependencies
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        dependencies = self._get_model_dependencies(model_name)
        
        if not dependencies:
            return True, "No additional dependencies required"
        
        # Check if already installed
        missing_deps = []
        for dep_name, import_name in dependencies:
            if not self._is_package_available(import_name):
                missing_deps.append(dep_name)
        
        if not missing_deps:
            return True, "All dependencies already available"
        
        # Install missing dependencies
        async with self._installation_lock:
            return await self._install_dependencies(missing_deps, model_name)
    
    def _get_model_dependencies(self, model_name: str) -> List[Tuple[str, str]]:
        """
        Get required dependencies for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of (package_name, import_name) tuples
        """
        dependencies_map = {
            "ben2-base": [
                ("transformers>=4.30.0", "transformers"),
                ("torch>=2.0.0", "torch"),
                ("transparent-background>=1.3.4", "transparent_background"),
            ],
            "inspyrenet_base": [
                ("rembg>=2.0.67", "rembg"),
                ("pillow>=10.0.0", "PIL"),
            ],
            "yolo11n-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
            ],
            "yolo11s-seg": [
                ("ultralytics>=8.3.0", "ultralytics"), 
                ("torch>=2.0.0", "torch"),
            ],
            "yolo11m-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
            ],
            "yolo11l-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
            ],
            "yolo11x-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
            ],
        }
        
        return dependencies_map.get(model_name, [])
    
    def _is_package_available(self, import_name: str) -> bool:
        """
        Check if a package is available for import.
        
        Args:
            import_name: Name to use for import
            
        Returns:
            True if package is available
        """
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    async def _install_dependencies(self, packages: List[str], model_name: str) -> Tuple[bool, str]:
        """
        Install missing dependencies.
        
        Args:
            packages: List of package names to install
            model_name: Name of the model requiring these dependencies
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Installing dependencies for {model_name}: {packages}")
            
            # Run pip install in a separate thread to avoid blocking
            def install_packages():
                cmd = [sys.executable, "-m", "pip", "install"] + packages
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                return result
            
            # Run installation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, install_packages)
            
            if result.returncode == 0:
                # Mark packages as installed
                for pkg in packages:
                    self._installed_packages.add(pkg)
                
                logger.info(f"Successfully installed dependencies for {model_name}")
                return True, f"Successfully installed {len(packages)} dependencies"
            else:
                error_msg = f"Failed to install dependencies: {result.stderr}"
                logger.error(error_msg)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Dependency installation timed out (5 minutes)"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during installation: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_installation_status(self) -> Dict[str, bool]:
        """
        Get installation status for all known models.
        
        Returns:
            Dictionary mapping model names to availability status
        """
        status = {}
        
        model_names = [
            "ben2-base", "inspyrenet_base", 
            "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"
        ]
        
        for model_name in model_names:
            dependencies = self._get_model_dependencies(model_name)
            if not dependencies:
                status[model_name] = True
                continue
                
            # Check if all dependencies are available
            all_available = True
            for _, import_name in dependencies:
                if not self._is_package_available(import_name):
                    all_available = False
                    break
            
            status[model_name] = all_available
        
        return status


# Global dependency manager instance
dependency_manager = DependencyManager()
