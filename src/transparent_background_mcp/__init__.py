"""
Transparent Background MCP Server

A local AI-powered background removal MCP server using state-of-the-art models.
"""

__version__ = "0.1.0"
__author__ = "Joe Leaver"
__email__ = "joe@example.com"

# Lazy imports to avoid dependency issues at package import time
def main():
    """Main entry point."""
    from .server import main as _main
    return _main()

def cli_main():
    """CLI entry point."""
    from .server import cli_main as _cli_main
    return _cli_main()

__all__ = ["main", "cli_main"]
