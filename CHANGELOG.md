# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Transparent Background MCP Server
- Support for multiple AI models:
  - BEN2 (Background Erase Network 2) for state-of-the-art background removal
  - YOLO11 segmentation models (nano, small, medium, large, extra-large)
  - InSPyReNet for stable background removal
- 100% local processing - no data leaves your machine
- Automatic GPU detection with CPU fallback
- Automatic model downloading and caching
- Hardware-based model recommendations
- Batch processing support
- Object-specific background removal with YOLO11
- MCP tools:
  - `remove_background` - Single image processing
  - `batch_remove_background` - Multiple image processing
  - `yolo_segment_objects` - Object-specific segmentation
  - `get_available_models` - Model information and recommendations
  - `get_system_info` - Hardware capabilities
- Support for multiple IDE integrations:
  - Claude Desktop
  - VSCode with GitHub Copilot
  - Cursor
  - Any MCP-compatible client
- Direct GitHub installation support with `uvx`
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Detailed documentation and examples

### Technical Features
- Automatic hardware detection and optimization
- Memory-efficient model loading and unloading
- GPU memory management with automatic cache clearing
- Support for PNG and JPEG output formats
- Base64 image encoding/decoding
- Image resizing and validation
- Alpha channel and transparency handling
- Configurable confidence thresholds
- Environment variable configuration
- Comprehensive error handling and logging

### Documentation
- Comprehensive README with installation and usage instructions
- Hardware requirements and performance guidelines
- IDE integration guides for popular editors
- Troubleshooting section
- Development setup instructions
- Example usage scripts
- API documentation for all tools

## [0.1.0] - 2025-01-XX

### Added
- Initial project structure
- Core MCP server implementation
- Model abstraction layer
- Hardware detection utilities
- Image processing utilities
- Model management and caching
- Basic test coverage
- GitHub Actions CI/CD
- MIT License
- Project documentation
