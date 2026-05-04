"""
Deterministic voxel grid downsampling.
"""

import numpy as np


def voxelize_points(points: np.ndarray, voxel_size: float, method: str = 'centroid') -> np.ndarray:
    """
    Downsample point cloud to voxel grid.
    
    Args:
        points: (N, 3) array [x, y, z]
        voxel_size: Voxel edge length in meters
        method: 'centroid' (default) or 'first'
    
    Returns:
        (M, 3) voxelized points where M <= N
    """
    if len(points) == 0:
        return points
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    if method == 'centroid':
        # Compute centroid of points in each voxel
        voxelized = np.zeros((len(unique_voxels), 3), dtype=points.dtype)
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            voxelized[i] = points[mask].mean(axis=0)
        return voxelized
    
    elif method == 'first':
        # Take first point in each voxel
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_idx]
    
    else:
        raise ValueError(f"Unknown voxelization method: {method}")
