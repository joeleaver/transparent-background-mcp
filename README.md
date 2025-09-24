# 🎨 Transparent Background MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Processing: 100% Local](https://img.shields.io/badge/processing-100%25%20local-green.svg)](https://github.com/joeleaver/transparent-background-mcp)

> **🔒 Privacy-First:** All AI processing happens locally on your machine. Your images never leave your device.

A powerful Model Context Protocol (MCP) server that removes backgrounds from images using state-of-the-art AI models. Perfect for creating transparent PNGs from any image, with support for multiple cutting-edge models including BEN2, YOLO11, and InSPyReNet.

## ✨ Features

- **🤖 Multiple AI Models**: BEN2 (latest), YOLO11 segmentation, InSPyReNet
- **🔒 100% Local Processing**: Your images never leave your machine
- **⚡ GPU Acceleration**: Automatic GPU detection with CPU fallback
- **📦 Automatic Model Downloads**: No manual setup required
- **🎯 Object-Specific Removal**: Target specific objects with YOLO11
- **⚡ Batch Processing**: Process multiple images efficiently
- **🔧 Hardware Optimization**: Automatic model recommendations based on your system

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- 4GB+ RAM (8GB+ recommended)
- Optional: NVIDIA GPU with 4GB+ VRAM for faster processing

### Installation Options

#### Option 1: Direct from GitHub (Recommended)
No installation required! Use directly in your MCP client configuration:

```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ],
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp"
      }
    }
  }
}
```

#### Option 2: Local Installation
```bash
# Install from GitHub
pip install git+https://github.com/joeleaver/transparent-background-mcp.git

# Or clone and install locally
git clone https://github.com/joeleaver/transparent-background-mcp.git
cd transparent-background-mcp
pip install -e .
```

#### Option 3: PyPI (when available)
```bash
pip install transparent-background-mcp
```

### Testing the Installation
```bash
# Test the server directly
uvx --from git+https://github.com/joeleaver/transparent-background-mcp.git transparent-background-mcp-server

# Or if installed locally
transparent-background-mcp-server
```

## 🔧 IDE Integration

### Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ],
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Alternative using uv run:**
```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--from",
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ],
      "env": {
        "MODEL_CACHE_DIR": "~/.cache/transparent-background-mcp"
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
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ],
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
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ],
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
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/joeleaver/transparent-background-mcp.git",
        "transparent-background-mcp-server"
      ],
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
- `image_data` (required): Base64 encoded image data
- `model_name` (optional): Model to use (`ben2-base`, `inspyrenet-base`, `yolo11m-seg`, etc.)
- `output_format` (optional): Output format (`PNG`, `JPEG`)
- `confidence_threshold` (optional): Detection confidence threshold (0.0-1.0)

### `batch_remove_background`
Process multiple images efficiently.

**Parameters:**
- `images_data` (required): Array of base64 encoded image data
- `model_name` (optional): Model to use
- `output_format` (optional): Output format
- `confidence_threshold` (optional): Detection confidence threshold

### `yolo_segment_objects`
Segment specific objects using YOLO11 models.

**Parameters:**
- `image_data` (required): Base64 encoded image data
- `model_name` (optional): YOLO model variant (`yolo11n-seg` to `yolo11x-seg`)
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
| **BEN2** | Background Removal | 213MB | 3.5GB | Excellent | Hair matting, fine details |
| **YOLO11x-seg** | Segmentation | 136MB | 2.8GB | Excellent | Object-specific removal |
| **YOLO11l-seg** | Segmentation | 87MB | 2.2GB | Very Good | Balanced performance |
| **YOLO11m-seg** | Segmentation | 50MB | 1.8GB | Good | General purpose |
| **YOLO11s-seg** | Segmentation | 22MB | 1.2GB | Good | Fast processing |
| **YOLO11n-seg** | Segmentation | 6MB | 0.8GB | Fair | Minimal resources |
| **InSPyReNet** | Background Removal | 65MB | 2.0GB | Very Good | Stable, reliable |

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

## 💡 Usage Examples

### Basic Background Removal
```python
# Using the MCP server through your IDE
"Remove the background from this image using the best available model"
# Attach your image file
```

### Object-Specific Removal
```python
# Target specific objects
"Use YOLO to remove only the person from this image, keeping everything else"
# Specify target_classes: ["person"]
```

### Batch Processing
```python
# Process multiple images
"Remove backgrounds from all these product photos using InSPyReNet"
# Attach multiple image files
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
DEFAULT_MODEL=inspyrenet-base

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
      "command": "uvx",
      "args": ["--from", "git+https://github.com/joeleaver/transparent-background-mcp.git", "transparent-background-mcp-server"],
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
- **Solution**: Use a smaller model (e.g., `yolo11n-seg` instead of `yolo11x-seg`)
- **Alternative**: Set `FORCE_CPU=true` to use CPU processing

#### Slow processing on CPU
- **Expected**: CPU processing is slower than GPU
- **Solution**: Consider upgrading to a GPU-enabled system or use smaller models

#### "uvx command not found"
- **Solution**: Install uv package manager: `pip install uv`
- **Alternative**: Use `pip install git+https://github.com/joeleaver/transparent-background-mcp.git`

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
uv run src/transparent_background_mcp/server.py

# Test with MCP Inspector
npx @modelcontextprotocol/inspector uv run src/transparent_background_mcp/server.py
```

### Testing GitHub Installation
```bash
# Test direct GitHub installation
uvx --from git+https://github.com/joeleaver/transparent-background-mcp.git transparent-background-mcp-server

# Test with MCP Inspector from GitHub
npx @modelcontextprotocol/inspector uvx --from git+https://github.com/joeleaver/transparent-background-mcp.git transparent-background-mcp-server
```

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
