#!/usr/bin/env python3
"""
Script to rebuild the package and update MCP configuration for development.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Main function to rebuild and update configuration."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🔨 Building package...")
    try:
        result = subprocess.run([sys.executable, "-m", "build"], 
                              capture_output=True, text=True, check=True)
        print("✅ Package built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 1
    
    # Find the latest wheel file
    dist_dir = project_root / "dist"
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel files found in dist/")
        return 1
    
    # Get the latest wheel file
    latest_wheel = max(wheel_files, key=os.path.getctime)
    wheel_path = str(latest_wheel.absolute()).replace("\\", "\\\\")
    
    print(f"📦 Latest wheel: {latest_wheel.name}")
    
    # Update development configuration
    config_path = project_root / "mcp_config_dev.json"
    config = {
        "mcpServers": {
            "transparent-background-mcp": {
                "command": "uvx",
                "args": [
                    "--from",
                    wheel_path,
                    "transparent-background-mcp-server"
                ]
            }
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated configuration: {config_path}")
    print(f"🔧 Wheel path: {wheel_path}")
    print("\n📋 Next steps:")
    print("1. Copy the contents of mcp_config_dev.json to your MCP client configuration")
    print("2. Restart your MCP client")
    print("3. Test the background removal functionality")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
