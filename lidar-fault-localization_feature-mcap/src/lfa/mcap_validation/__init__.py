"""
MCAP-based LiDAR fog validation pipeline.

Compares synthetic fog (generated from clean scans) against real fog data
using distributional statistics and optional geometric metrics.
"""

from .cli import main

__all__ = ['main']
