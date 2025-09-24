"""
Lazy dependency management for background removal models.
Handles on-demand installation of model-specific dependencies.
"""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
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

    def _packages_root(self) -> Path:
        """Base directory for dynamically installed packages."""
        root = os.getenv("MODEL_DEP_CACHE_DIR")
        if not root:
            root = str(Path.home() / ".cache" / "transparent-background-mcp" / "site-packages")
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _target_dir_for_model(self, model_name: str) -> Path:
        """Directory where this model's dependencies are installed via --target."""
        d = self._packages_root() / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_path_on_sys_path(self, path: Path) -> None:
        """Ensure the given path is importable in this process."""
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)

    async def ensure_dependencies(self, model_name: str) -> Tuple[bool, str]:
        """
        Ensure all dependencies for a model are installed and importable.
        Uses pip --target into a persistent cache dir to avoid writing into the
        uvx/venv environment, and extends sys.path at runtime.
        """
        dependencies = self._get_model_dependencies(model_name)

        # Make sure our target directory is on sys.path before import checks
        target_dir = self._target_dir_for_model(model_name)
        self._ensure_path_on_sys_path(target_dir)

        if not dependencies:
            return True, "No additional dependencies required"

        # Check if already installed (including our target_dir)
        missing_deps = []
        for dep_name, import_name in dependencies:
            if not self._is_package_available(import_name):
                missing_deps.append(dep_name)

        if not missing_deps:
            return True, "All dependencies already available"

        # Install missing dependencies into target_dir
        async with self._installation_lock:
            return await self._install_dependencies(missing_deps, model_name, str(target_dir))

    def _get_model_dependencies(self, model_name: str) -> List[Tuple[str, str]]:
        """
        Get required dependencies for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of (package_name, import_name) tuples
        """
        dependencies_map = {
            # BEN2
            "ben2-base": [
                ("transformers>=4.30.0", "transformers"),
                ("torch>=2.0.0", "torch"),
                ("huggingface-hub>=0.20.0", "huggingface_hub"),
                ("transparent-background>=1.3.4", "transparent_background"),
            ],
            # InSPyReNet (fix name: use hyphen, keep an underscore alias for robustness)
            "inspyrenet-base": [
                ("rembg>=2.0.67", "rembg"),
                ("pillow>=10.0.0", "PIL"),
            ],
            "inspyrenet_base": [  # legacy alias
                ("rembg>=2.0.67", "rembg"),
                ("pillow>=10.0.0", "PIL"),
            ],
            # YOLO variants
            "yolo11n-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
                ("opencv-python>=4.8.0", "cv2"),
            ],
            "yolo11s-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
                ("opencv-python>=4.8.0", "cv2"),
            ],
            "yolo11m-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
                ("opencv-python>=4.8.0", "cv2"),
            ],
            "yolo11l-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
                ("opencv-python>=4.8.0", "cv2"),
            ],
            "yolo11x-seg": [
                ("ultralytics>=8.3.0", "ultralytics"),
                ("torch>=2.0.0", "torch"),
                ("opencv-python>=4.8.0", "cv2"),
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

    def _extra_for_model(self, model_name: str) -> Optional[str]:
        """Return the optional-dependency extra that matches a model name."""
        if model_name.startswith("ben2"):
            return "ben2"
        if model_name.startswith("inspyrenet"):
            return "inspyrenet"
        if model_name.startswith("yolo"):
            return "yolo"
        return None

    def _uvx_hint(self, model_name: str) -> str:
        """Build a helpful hint for uvx users to pre-install extras."""
        extra = self._extra_for_model(model_name)
        if not extra:
            return ""
        return (
            " Tip: if you are using uvx, pre-install optional deps with: "
            f"uvx --from git+https://github.com/joeleaver/transparent-background-mcp.git#egg=transparent-background-mcp[{extra}] "
            "transparent-background-mcp-server"
        )

    async def _install_dependencies(self, packages: List[str], model_name: str, target_dir: str) -> Tuple[bool, str]:
        """
        Install missing dependencies into a persistent --target directory.

        Args:
            packages: List of package names to install
            model_name: Name of the model requiring these dependencies
            target_dir: Directory to install packages into (added to sys.path)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Installing dependencies for {model_name} into {target_dir}: {packages}")

            # Run pip install in a separate thread to avoid blocking
            def install_packages():
                cmd = [
                    sys.executable, "-m", "pip", "install",
                    "--no-input", "--disable-pip-version-check",
                    "-t", target_dir,
                ] + packages
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for first-time large deps
                )
                return result

            # Run installation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, install_packages)

            if result.returncode == 0:
                # Ensure path is active in current process
                self._ensure_path_on_sys_path(Path(target_dir))
                # Mark packages as installed
                for pkg in packages:
                    self._installed_packages.add(pkg)

                logger.info(f"Successfully installed dependencies for {model_name}")
                return True, f"Successfully installed {len(packages)} dependencies"
            else:
                error_msg = f"Failed to install dependencies: {result.stderr}" + self._uvx_hint(model_name)
                logger.error(error_msg)
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "Dependency installation timed out (10 minutes)" + self._uvx_hint(model_name)
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during installation: {str(e)}" + self._uvx_hint(model_name)
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
            "ben2-base", "inspyrenet-base",
            "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"
        ]

        for model_name in model_names:
            # Make sure this model's target dir is importable when checking
            target = self._target_dir_for_model(model_name)
            self._ensure_path_on_sys_path(target)

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
