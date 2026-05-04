"""
Geometric and intensity metrics computation.
"""

from typing import Dict, List
import numpy as np
from scipy.spatial import cKDTree
from .voxelize import voxelize_points


def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray, voxel_size: float) -> Dict:
    """
    Compute symmetric Chamfer distance between two point clouds.
    
    Args:
        pc1, pc2: (N, 3) arrays
        voxel_size: Voxelization resolution in meters
    
    Returns:
        dict with chamfer_mean, chamfer_p50, chamfer_p95
    """
    # Voxelize both clouds
    pc1_vox = voxelize_points(pc1, voxel_size)
    pc2_vox = voxelize_points(pc2, voxel_size)
    
    if len(pc1_vox) == 0 or len(pc2_vox) == 0:
        return {'chamfer_mean': np.nan, 'chamfer_p50': np.nan, 'chamfer_p95': np.nan}
    
    # Build KDTree for fast nearest-neighbor
    tree1 = cKDTree(pc1_vox)
    tree2 = cKDTree(pc2_vox)
    
    # pc1 -> pc2 distances
    dist_1to2, _ = tree2.query(pc1_vox, k=1)
    
    # pc2 -> pc1 distances
    dist_2to1, _ = tree1.query(pc2_vox, k=1)
    
    # Symmetric Chamfer: mean of both directions
    all_dists = np.concatenate([dist_1to2, dist_2to1])
    
    return {
        'chamfer_mean': float(np.mean(all_dists)),
        'chamfer_p50': float(np.percentile(all_dists, 50)),
        'chamfer_p95': float(np.percentile(all_dists, 95))
    }


def compute_hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray, voxel_size: float) -> Dict:
    """
    Compute Hausdorff distance (max and 95th percentile).
    
    Args:
        pc1, pc2: (N, 3) arrays
        voxel_size: Voxelization resolution in meters
    
    Returns:
        dict with hausdorff_max, hausdorff_p95
    """
    # Voxelize
    pc1_vox = voxelize_points(pc1, voxel_size)
    pc2_vox = voxelize_points(pc2, voxel_size)
    
    if len(pc1_vox) == 0 or len(pc2_vox) == 0:
        return {'hausdorff_max': np.nan, 'hausdorff_p95': np.nan}
    
    tree1 = cKDTree(pc1_vox)
    tree2 = cKDTree(pc2_vox)
    
    dist_1to2, _ = tree2.query(pc1_vox, k=1)
    dist_2to1, _ = tree1.query(pc2_vox, k=1)
    
    # Hausdorff = max of max distances in both directions
    max_1to2 = np.max(dist_1to2)
    max_2to1 = np.max(dist_2to1)
    hausdorff_max = max(max_1to2, max_2to1)
    
    # Also compute 95th percentile (more robust)
    all_dists = np.concatenate([dist_1to2, dist_2to1])
    hausdorff_p95 = np.percentile(all_dists, 95)
    
    return {
        'hausdorff_max': float(hausdorff_max),
        'hausdorff_p95': float(hausdorff_p95)
    }


def compute_intensity_metrics(intensity1: np.ndarray, intensity2: np.ndarray) -> Dict:
    """
    Compare intensity distributions.
    
    Intensities should be pre-normalized per scan.
    
    Args:
        intensity1, intensity2: (N,) arrays (normalized)
    
    Returns:
        dict with intensity_rmse, intensity_mae
    """
    if len(intensity1) != len(intensity2):
        # Different point counts - resample to compare distributions
        # Use histogram comparison instead
        bins = np.linspace(0, 1, 50)  # Assume normalized to [0,1]
        hist1, _ = np.histogram(intensity1, bins=bins, density=True)
        hist2, _ = np.histogram(intensity2, bins=bins, density=True)
        
        rmse = np.sqrt(np.mean((hist1 - hist2) ** 2))
        mae = np.mean(np.abs(hist1 - hist2))
    else:
        rmse = np.sqrt(np.mean((intensity1 - intensity2) ** 2))
        mae = np.mean(np.abs(intensity1 - intensity2))
    
    return {
        'intensity_rmse': float(rmse),
        'intensity_mae': float(mae)
    }


def normalize_intensity_median(intensity: np.ndarray) -> np.ndarray:
    """
    Normalize intensity using median and MAD.
    
    Args:
        intensity: (N,) array
    
    Returns:
        (N,) normalized array
    """
    median = np.median(intensity)
    mad = np.median(np.abs(intensity - median))
    
    if mad == 0:
        return np.zeros_like(intensity)
    
    return (intensity - median) / mad


def aggregate_sequence_metrics(frame_results: List[Dict]) -> Dict:
    """
    Aggregate per-frame metrics across a sequence.
    
    Computes statistics (mean, std, median, p25, p75, min, max) for:
    - Chamfer distances (mean, p50, p95)
    - Hausdorff distances (max, p95)
    - Intensity metrics (rmse, mae)
    - Point counts
    
    Args:
        frame_results: List of per-frame metric dicts with keys:
            'chamfer_mean', 'chamfer_p50', 'chamfer_p95',
            'hausdorff_max', 'hausdorff_p95',
            'intensity_rmse', 'intensity_mae',
            'synth_n_points', 'real_n_points'
    
    Returns:
        Dict with aggregated statistics for each metric
    """
    if not frame_results:
        return {}
    
    # Define metric keys to aggregate
    metric_keys = [
        'chamfer_mean', 'chamfer_p50', 'chamfer_p95',
        'hausdorff_max', 'hausdorff_p95',
        'intensity_rmse', 'intensity_mae',
        'synth_n_points', 'real_n_points'
    ]
    
    aggregated = {'n_frames': len(frame_results)}
    
    for key in metric_keys:
        # Extract values (skip NaN)
        values = [r[key] for r in frame_results if not np.isnan(r.get(key, np.nan))]
        
        if not values:
            # All NaN - set stats to NaN
            aggregated[f'{key}_mean'] = np.nan
            aggregated[f'{key}_std'] = np.nan
            aggregated[f'{key}_median'] = np.nan
            aggregated[f'{key}_p25'] = np.nan
            aggregated[f'{key}_p75'] = np.nan
            aggregated[f'{key}_min'] = np.nan
            aggregated[f'{key}_max'] = np.nan
        else:
            values = np.array(values)
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_median'] = float(np.median(values))
            aggregated[f'{key}_p25'] = float(np.percentile(values, 25))
            aggregated[f'{key}_p75'] = float(np.percentile(values, 75))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
    
    return aggregated
