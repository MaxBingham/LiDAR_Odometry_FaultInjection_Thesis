"""
Fog simulation wrapper using existing FogSimulator.
"""

from typing import List, Dict
import numpy as np
from lfi.fault_models.fog_injector import FogSimulator


def apply_fog_to_frames(clean_frames: List[Dict], visibility: float, fog_metric: str = "distance") -> List[Dict]:
    """
    Apply fog simulation to clean frames.
    
    Args:
        clean_frames: List of clean frame dicts with xyz, intensity
        visibility: Visibility in meters
        fog_metric: 'distance' or 'chamfer'
    
    Returns:
        List of synthetic fog frames with same structure
    """
    print(f"\n{'='*60}")
    print(f"  APPLYING FOG SIMULATION (V={visibility}m, metric={fog_metric})")
    print(f"{'='*60}")
    
    fog_sim = FogSimulator(V=visibility, metric=fog_metric, seed=42)
    synthetic_frames = []
    
    total_points = 0
    total_deleted = 0
    total_backscattered = 0
    
    for i, clean_frame in enumerate(clean_frames):
        # Combine xyz + intensity into (N,4)
        points_4d = np.hstack([clean_frame['xyz'], clean_frame['intensity'][:, None]])
        
        # Apply fog
        foggy_points = fog_sim.apply_noise_fog(points_4d)
        
        # Check for zero points (should never happen with probability-based model)
        if len(foggy_points) == 0:
            raise RuntimeError(
                f"Frame {i}: Fog simulation deleted ALL points!\n"
                f"Original: {len(points_4d)} points, Visibility: {visibility}m\n"
                f"This should not happen with probability-based fog model."
            )
        
        synthetic_frames.append({
            'xyz': foggy_points[:, :3],
            'intensity': foggy_points[:, 3],
            'timestamp': clean_frame['timestamp']
        })
        
        # Accumulate stats
        total_points += fog_sim.stats['total']
        total_deleted += fog_sim.stats['deleted']
        total_backscattered += fog_sim.stats['backscattered']
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(clean_frames)} frames...")
    
    # Print summary
    print(f"\n✓ Fog simulation complete:")
    print(f"  Total points: {total_points:,}")
    print(f"  Deleted: {total_deleted:,} ({100*total_deleted/total_points:.1f}%)")
    print(f"  Backscattered: {total_backscattered:,} ({100*total_backscattered/total_points:.1f}%)")
    print(f"  Remaining: {total_points - total_deleted:,} ({100*(total_points-total_deleted)/total_points:.1f}%)")
    
    return synthetic_frames
