# 🎨 Transparent Background MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Processing: 100% Local](https://img.shields.io/badge/processing-100%25%20local-green.svg)](https://github.com/joeleaver/transparent-background-mcp)

> **🔒 Privacy-First:** All AI processing happens locally on your machine. Your images never leave your device.

A powerful Model Context Protocol (MCP) server that removes backgrounds from images using state-of-the-art AI models. Perfect for creating transparent PNGs from any image, with support for multiple cutting-edge models including BEN2, YOLO11, and InSPyReNet.

## ✨ Features

- **🤖 Curated AI Models**: BEN2 (state-of-the-art), InSPyReNet (portraits), YOLO11-S/L (speed)
- **🔒 100% Local Processing**: Your images never leave your machine
- **📁 Smart Input Handling**: Accepts both file paths and base64 data automatically
- **⚡ GPU Acceleration**: Automatic GPU detection with CPU fallback
- **📦 Automatic Model Downloads**: No manual setup required
- **🎯 Object-Specific Removal**: Target specific objects with YOLO11
- **⚡ Batch Processing**: Process multiple images efficiently
- **🔧 Hardware Optimization**: Automatic model recommendations based on your system

## 🖼️ Visual Results Comparison

See the difference between our curated models with a real test image:

<div align="center">

### Original Image
<img src="model_test_results/original_comparison.jpg" alt="Original test image" width="200"/>

### Model Results

| **BEN2 (Default)** | **InSPyReNet (Portraits)** |
|:---:|:---:|
| <img src="model_test_results/ben2-base_result.png" alt="BEN2 result" width="250"/> | <img src="model_test_results/inspyrenet-base_result.png" alt="InSPyReNet result" width="250"/> |
| **State-of-the-art quality** | **Excellent for portraits** |
| Load: 11.5s, Process: 10.7s | Load: 1.1s, Process: 1.3s |

| **YOLO11-L (Balanced)** | **YOLO11-S (Speed)** |
|:---:|:---:|
| <img src="model_test_results/yolo11l-seg_result.png" alt="YOLO11-L result" width="250"/> | <img src="model_test_results/yolo11s-seg_result.png" alt="YOLO11-S result" width="250"/> |
| **Fast & balanced** | **Fastest processing** |
| Load: 0.1s, Process: 0.6s | Load: 0.2s, Process: 0.3s |

</div>

> 💡 **Tip**: All models support both file paths and base64 input. BEN2 is used by default for the highest quality results.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Optional: NVIDIA GPU with 4GB+ VRAM for faster processing

### Installation (one line)
```bash
pip install transparent-background-mcp
```

- This installs the MCP server and all required dependencies (PyTorch, Ultralytics, rembg, etc.).
- Model weights are downloaded automatically on first use by the underlying libraries.

### Run the server
```bash
transparent-background-mcp-server
```

If you want to warm up models (download weights ahead of time), just run one quick request per model (e.g., process a tiny image once). Subsequent runs will be fast.

## 🔧 IDE Integration

### Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "transparent-background-mcp-server",
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```


### VSCode with GitHub Copilot

Add to your workspace `.vscode/mcp.json`:

```json
{
  "servers": {
    "transparent-background-mcp": {
      "command": "transparent-background-mcp-server",
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp"
      }
    }
  }
}
```

### Cursor

In Cursor Settings → MCP Servers:

```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "transparent-background-mcp-server",
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp"
      }
    }
  }
}
```

### Other MCP Clients

For any MCP-compatible client, use this standard configuration:

```json
{
  "servers": {
    "transparent-background-mcp": {
      "command": "transparent-background-mcp-server",
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp"
      }
    }
  }
}
```

**Note:** Configuration file locations vary by client. Check your client's documentation for the specific config file path.

## 🛠️ Available Tools

### `remove_background`
Remove background from a single image using AI models.

**Parameters:**
- `image_data` (required): Image file path or base64 encoded image data
- `model_name` (optional): Model to use (`ben2-base`, `inspyrenet-base`, `yolo11s-seg`, `yolo11l-seg`)
- `output_format` (optional): Output format (`PNG`, `JPEG`)
- `confidence_threshold` (optional): Detection confidence threshold (0.0-1.0)

Note: When `image_data` is a file path, the output image is also saved next to the input file by default, using the same filename with a suffix and appropriate extension.
- Default suffix: `-no-bg` (can be changed via `OUTPUT_SUFFIX` env var)
- Example: `photo.jpg` -> `photo-no-bg.png`
- The response still includes base64 data for clients that prefer inline content.

### `batch_remove_background`
Process multiple images efficiently.

**Parameters:**
- `images_data` (required): Array of image file paths or base64 image data
- `model_name` (optional): Model to use
- `output_format` (optional): Output format
- `confidence_threshold` (optional): Detection confidence threshold

Note: For any entries that are file paths, outputs are saved alongside each input file using the same default suffix behavior as above (e.g., `image.jpg` -> `image-no-bg.png`). The response also includes base64-encoded outputs.

### `yolo_segment_objects`
Segment specific objects using YOLO11 models.

**Parameters:**
- `image_data` (required): Base64 encoded image data
- `model_name` (optional): YOLO model variant (`yolo11s-seg` or `yolo11l-seg`)
- `target_classes` (optional): Specific object classes to segment (e.g., `["person", "car"]`)
- `confidence_threshold` (optional): Detection confidence threshold
- `combine_masks` (optional): Whether to combine multiple object masks

### `get_available_models`
Get information about available models and system recommendations.

### `get_system_info`
Get system hardware information and capabilities.

## 🤖 Available Models

| Model | Type | Size | VRAM | Performance | Specialty |
|-------|------|------|------|-------------|-----------|
| **BEN2** | Background Removal | 213MB | 3.5GB | Excellent | Hair matting, fine details (default) |
| **InSPyReNet** | Background Removal | 65MB | 2.0GB | Very Good | Portraits, stable results |
| **YOLO11l-seg** | Segmentation | 87MB | 2.2GB | Very Good | Balanced performance |
| **YOLO11s-seg** | Segmentation | 22MB | 1.2GB | Good | Fast processing |

## 📋 Hardware Requirements

### **Minimum Requirements**
- **CPU**: Any modern processor
- **RAM**: 4GB system memory
- **GPU**: None (CPU processing supported)
- **Storage**: 1GB for models

### **Recommended Setup**
- **CPU**: Multi-core processor
- **RAM**: 8GB+ system memory
- **GPU**: 4GB+ VRAM (NVIDIA recommended)
- **Storage**: 2GB for multiple models

### **Optimal Performance**
- **CPU**: High-performance multi-core
- **RAM**: 16GB+ system memory
- **GPU**: 8GB+ VRAM (RTX 3070+, RTX 4060 Ti+)
- **Storage**: SSD with 5GB+ free space

## 🎯 Model Performance Comparison

All models have been tested with a 1024x1024 test image on a CPU-only system. Here are the performance results:

### ⚡ Performance Metrics

| Model | Load Time | Process Time | Quality | Best For |
|-------|-----------|--------------|---------|----------|
| **yolo11s-seg** | 0.23s | 0.34s | ⭐⭐⭐⭐ | Speed, fast processing |
| **yolo11l-seg** | 0.14s | 0.62s | ⭐⭐⭐⭐⭐ | Balanced quality/speed |
| **inspyrenet-base** | 1.07s | 1.33s | ⭐⭐⭐⭐⭐ | Excellent for portraits |
| **ben2-base** | 11.50s | 10.70s | ⭐⭐⭐⭐⭐ | State-of-the-art quality (default) |

### 🏆 Model Recommendations

- **🥇 Default & Best Quality**: `ben2-base` - State-of-the-art results (default model)
- **🎨 For Portraits**: `inspyrenet-base` - Excellent for people and detailed subjects
- **🚀 For Speed**: `yolo11s-seg` - Sub-second processing
- **⚖️ For Balance**: `yolo11l-seg` - Good quality in under 1 second

### 📊 Key Insights

- **BEN2** is the default model providing the highest quality results
- **InSPyReNet** excels at portrait and people photography
- **YOLO11 models** offer the best speed when you need fast processing
- All models support both **file paths** and **base64 image data** as input
- GPU acceleration significantly improves processing times for all models

## 💡 Usage Examples

### Basic Background Removal

**Using File Paths (Recommended):**
```python
# Simply provide the file path - BEN2 is used by default for best quality
"Remove the background from C:/Users/me/photos/portrait.jpg"
```

**Using Base64 (Also Supported):**
```python
# The MCP server also accepts base64 encoded images
"Remove the background from this base64 image data"
# BEN2 model used by default for highest quality
```

### Speed vs Quality Examples

**For Speed (< 1 second):**
```python
"Remove background from /path/to/image.jpg using yolo11s-seg for fastest processing"
```

**For Quality (default - best results):**
```python
"Remove background from /path/to/portrait.jpg"
# Uses BEN2 by default for highest quality
```

### Object-Specific Removal
```python
# Target specific objects with YOLO models
"Use yolo11l-seg to remove only the person from /path/to/photo.jpg, keeping the background"
# The system will automatically segment and remove only people
```

### Batch Processing
```python
# Process multiple images efficiently
"Remove backgrounds from these product photos using yolo11s-seg"
# Provide multiple file paths: ["/path/to/photo1.jpg", "/path/to/photo2.jpg"]
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Model cache directory
MODEL_CACHE_DIR=~/.cache/transparent-background-mcp

# Force CPU usage (disable GPU)
FORCE_CPU=false

# Default model
DEFAULT_MODEL=ben2-base

# Logging level
LOG_LEVEL=INFO

# Maximum batch size
MAX_BATCH_SIZE=4

# Clear GPU cache after operations
CLEAR_GPU_CACHE=true
```

### Advanced Configuration

For advanced users, you can customize model behavior:

```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "transparent-background-mcp-server",
      "env": {
        "MODEL_CACHE_DIR": "/custom/cache/path",
        "DEFAULT_MODEL": "ben2-base",
        "MAX_BATCH_SIZE": "8",
        "LOG_LEVEL": "DEBUG",
        "FORCE_CPU": "false",
        "CLEAR_GPU_CACHE": "true"
      }
    }
  }
}

## 🐛 Troubleshooting

### Common Issues

#### "Model not found" or download errors
- **Solution**: Check internet connection and ensure sufficient disk space
- **Alternative**: Try a different model or clear cache with `get_system_info` tool

#### GPU out of memory errors
- **Solution**: Use a smaller model (e.g., `yolo11s-seg` instead of `yolo11l-seg`)
- **Alternative**: Set `FORCE_CPU=true` to use CPU processing

#### Slow processing on CPU
- **Expected**: CPU processing is slower than GPU
- **Solution**: Consider upgrading to a GPU-enabled system or use smaller models


### Performance Tips

1. **Use GPU**: Ensure CUDA-compatible GPU with sufficient VRAM
2. **Batch Processing**: Process multiple images together for efficiency
3. **Model Selection**: Choose appropriate model size for your hardware
4. **Image Size**: Resize large images before processing to save memory
5. **Cache Management**: Models are cached after first download

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/joeleaver/transparent-background-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/joeleaver/transparent-background-mcp/discussions)

## 🔧 Development

### Local Development

```bash
# Clone the repository
git clone https://github.com/joeleaver/transparent-background-mcp.git
cd transparent-background-mcp

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Testing with MCP Inspector
```bash
# Test the server locally
transparent-background-mcp-server

# Test with MCP Inspector
npx @modelcontextprotocol/inspector transparent-background-mcp-server
```

### Testing GitHub Installation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BEN2**: Background Erase Network 2 for state-of-the-art background removal
- **Ultralytics**: YOLO11 models for object segmentation
- **InSPyReNet**: Stable background removal implementation
- **REMBG**: Background removal library ecosystem
- **Model Context Protocol**: Framework for AI tool integration

## 🔗 Related Projects

- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [REMBG](https://github.com/danielgatis/rembg)
- [Transparent Background](https://github.com/plemeri/transparent-background)

---

**Made with ❤️ for the AI community. 100% local, 100% private, 100% powerful.**
```
