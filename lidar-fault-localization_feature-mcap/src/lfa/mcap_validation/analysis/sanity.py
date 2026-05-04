"""
Sanity checks and overlay visualizations.
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def generate_overlay_visualization(
    clean_xyz: np.ndarray,
    comparison_xyz: np.ndarray,
    output_path: Path,
    title: str,
    voxel_size: float
):
    """
    Generate 2D bird's-eye view overlay (XY projection).
    
    Args:
        clean_xyz: (N, 3) clean points
        comparison_xyz: (M, 3) comparison points (synthetic or real fog)
        output_path: PNG output path
        title: Plot title
        voxel_size: Voxelization used for metrics
    """
    plt.figure(figsize=(12, 10))
    
    # Plot clean in blue
    plt.scatter(clean_xyz[:, 0], clean_xyz[:, 1], c='blue', s=1, alpha=0.3, label='Clean')
    
    # Plot comparison in red
    plt.scatter(comparison_xyz[:, 0], comparison_xyz[:, 1], c='red', s=1, alpha=0.5, label=title.split(':')[0])
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'{title}\nVoxel size: {voxel_size}m')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved overlay: {output_path.name}")
