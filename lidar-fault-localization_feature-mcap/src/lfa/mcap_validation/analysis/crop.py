"""
Spatial cropping of point cloud frames to a 3D bounding box (region of interest).
"""

from typing import Dict, Optional
import numpy as np


def crop_frame(frame: Dict, roi: Dict) -> Dict:
    """
    Crop a frame's point cloud to an axis-aligned 3D bounding box.

    Args:
        frame: dict with 'xyz' (N,3), 'intensity' (N,), 'timestamp'
        roi: dict with keys x_min, x_max, y_min, y_max, z_min, z_max.
             Any missing axis pair defaults to -inf/+inf (no filtering on that axis).

    Returns:
        New frame dict containing only points inside the ROI.
    """
    xyz = frame['xyz']

    mask = np.ones(len(xyz), dtype=bool)

    if 'x_min' in roi or 'x_max' in roi:
        mask &= xyz[:, 0] >= roi.get('x_min', -np.inf)
        mask &= xyz[:, 0] <= roi.get('x_max', np.inf)

    if 'y_min' in roi or 'y_max' in roi:
        mask &= xyz[:, 1] >= roi.get('y_min', -np.inf)
        mask &= xyz[:, 1] <= roi.get('y_max', np.inf)

    if 'z_min' in roi or 'z_max' in roi:
        mask &= xyz[:, 2] >= roi.get('z_min', -np.inf)
        mask &= xyz[:, 2] <= roi.get('z_max', np.inf)

    return {
        'xyz': xyz[mask],
        'intensity': frame['intensity'][mask],
        'timestamp': frame['timestamp'],
    }


def validate_roi(roi: Dict, label: str) -> Dict:
    """
    Validate and normalise an ROI dict from YAML.

    Ensures min <= max for every axis that is specified.
    Returns the validated roi dict (unchanged if already valid).
    """
    for axis in ('x', 'y', 'z'):
        lo_key = f'{axis}_min'
        hi_key = f'{axis}_max'
        if lo_key in roi and hi_key in roi:
            if roi[lo_key] > roi[hi_key]:
                raise ValueError(
                    f"ROI for '{label}': {lo_key} ({roi[lo_key]}) > {hi_key} ({roi[hi_key]})"
                )
    return roi
