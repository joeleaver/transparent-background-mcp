"""
Transparent Background MCP Server

A local AI-powered background removal MCP server using state-of-the-art models.
"""

__version__ = "0.1.0"
__author__ = "Joe Leaver"
__email__ = "joe@example.com"

from .server import main, cli_main

__all__ = ["main", "cli_main"]
