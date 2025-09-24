# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. MCP Connection Closed Error

**Error:**
```
MCP error -32000: Connection closed
Command: uvx --from /Users/joe/transparent-background-mcp
Args: 
```

**Problem:** Missing the script name in the uvx command.

**Solution:** Make sure your MCP configuration includes the script name:

**❌ Incorrect:**
```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": ["--from", "/path/to/transparent-background-mcp"]
    }
  }
}
```

**✅ Correct:**
```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/joeleaver/transparent-background-mcp.git", 
        "transparent-background-mcp-server"
      ]
    }
  }
}
```

**Key Point:** The last argument `"transparent-background-mcp-server"` is the script name that uvx should execute.

### 2. Module Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'transparent_background_mcp.models'
```

**Problem:** Heavy dependencies being imported at startup.

**Solution:** This has been fixed with lazy loading. Make sure you're using the latest version.

### 3. MCP Timeout During Startup

**Error:**
```
MCP error -32001: Request timed out
```

**Problem:** Heavy dependencies downloading during server startup.

**Solution:** This has been fixed with lazy dependency loading. Dependencies are now installed on-demand when you first use each model.

### 4. First Model Use Takes Long Time

**Expected Behavior:** The first time you use a model, it may take 30-60 seconds to install dependencies.

**What's Happening:**
- BEN2: Downloads ~500MB (PyTorch + Transformers)
- YOLO11: Downloads ~300MB (PyTorch + Ultralytics)  
- InSPyReNet: Downloads ~200MB (OpenCV + rembg)

**Subsequent Uses:** Fast (~3 seconds) as dependencies are cached.

### 5. GPU Not Detected

**Warning:**
```
PyTorch not available - GPU detection disabled
```

**Explanation:** This is normal during initial startup. GPU detection happens when model dependencies are installed.

### 6. Path Format Issues (Windows)

**Error:**
```
× Failed to resolve `--with` requirement
╰─▶ Distribution not found at:
    file:///C:/Users/joe/transparent-background-mcp
```

**Problem:** Incorrect path format for local development on Windows.

**Solution:** Use the correct Windows path format:

**❌ Incorrect:**
```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": ["--from", "/Users/joe/transparent-background-mcp", "transparent-background-mcp-server"]
    }
  }
}
```

**✅ Correct (Windows):**
```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": ["--from", "file:///C:/Users/joe/Dropbox/Dev/transparent-background-mcp", "transparent-background-mcp-server"]
    }
  }
}
```

**Or use the GitHub URL (recommended):**
```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/joeleaver/transparent-background-mcp.git", "transparent-background-mcp-server"]
    }
  }
}
```

### 7. Dependency Conflicts

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solution:** These warnings are usually harmless. The MCP server will still work correctly.

## 🚀 Quick Test

To verify your installation is working:

1. **Check Server Startup:**
   ```bash
   # From GitHub (recommended)
   uvx --from git+https://github.com/joeleaver/transparent-background-mcp.git transparent-background-mcp-server --help

   # From local path (Windows)
   uvx --from "file:///C:/full/path/to/transparent-background-mcp" transparent-background-mcp-server --help
   ```

2. **Expected Output:**
   ```
   Starting Transparent Background MCP Server
   System info: {'platform': '...', 'memory_gb': ...}
   GPU available: True/False
   ```

3. **Server Should Start in ~3 seconds** (not 60+ seconds like before)

## 📞 Getting Help

If you're still having issues:

1. **Check the logs** in your MCP client for detailed error messages
2. **Verify your configuration** matches the examples exactly
3. **Test the uvx command directly** in terminal first
4. **Make sure you have the latest version** of the package

## 🎯 Working Configuration Template

Copy this exact configuration:

```json
{
  "mcpServers": {
    "transparent-background-mcp": {
      "command": "uvx",
      "args": [
        "--from", 
        "git+https://github.com/joeleaver/transparent-background-mcp.git", 
        "transparent-background-mcp-server"
      ]
    }
  }
}
```

**Important:** Don't forget the `"transparent-background-mcp-server"` at the end!
