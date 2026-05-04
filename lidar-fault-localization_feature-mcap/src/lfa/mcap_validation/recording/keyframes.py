"""
Keyframe parsing and validation from YAML.
Supports both timestamp-based and index-based selection.
"""

from pathlib import Path
from typing import List, Dict, Tuple, NamedTuple, Optional
import yaml
import numpy as np

from lfa.mcap_validation.analysis.crop import validate_roi


class KeyframeSequence(NamedTuple):
    """Container for keyframe sequence information."""
    frame_pairs: List[Tuple[int, int, str]]  # List of (clean_idx, fog_idx, label)
    label: str
    is_sequence: bool  # True if this was a range, False if single frame
    roi: Optional[Dict] = None  # 3D bounding box {x_min, x_max, y_min, ...}


def find_nearest_frame_by_timestamp(frames: List[Dict], target_time: float, tolerance: float = 0.1) -> int:
    """
    Find frame index with timestamp closest to target time.
    
    Args:
        frames: List of frame dicts with 'timestamp' key
        target_time: Target timestamp in seconds
        tolerance: Maximum allowed time difference in seconds
    
    Returns:
        Frame index
    
    Raises:
        ValueError: If no frame within tolerance
    """
    timestamps = np.array([f['timestamp'] for f in frames])
    diffs = np.abs(timestamps - target_time)
    min_idx = np.argmin(diffs)
    min_diff = diffs[min_idx]
    
    if min_diff > tolerance:
        raise ValueError(
            f"No frame found within {tolerance}s of target time {target_time:.3f}s. "
            f"Nearest frame at {timestamps[min_idx]:.3f}s ({min_diff:.3f}s away)"
        )
    
    return int(min_idx)


def extract_time_window(frames: List[Dict], center_time: float, duration: float, tolerance: float = 0.1) -> Tuple[int, int]:
    """
    Extract frame range for a time window centered on reference time.
    
    Args:
        frames: List of frame dicts with 'timestamp' key
        center_time: Center timestamp in seconds
        duration: Total window duration in seconds
        tolerance: Timestamp matching tolerance in seconds
    
    Returns:
        (start_idx, end_idx) inclusive
    
    Raises:
        ValueError: If window extends beyond available data
    """
    timestamps = np.array([f['timestamp'] for f in frames])
    
    # Define window bounds
    half_duration = duration / 2.0
    window_start = center_time - half_duration
    window_end = center_time + half_duration
    
    # Find frames within window
    in_window = (timestamps >= window_start) & (timestamps <= window_end)
    indices = np.where(in_window)[0]
    
    if len(indices) == 0:
        raise ValueError(
            f"No frames found in time window [{window_start:.3f}s, {window_end:.3f}s] "
            f"centered at {center_time:.3f}s. Available range: [{timestamps[0]:.3f}s, {timestamps[-1]:.3f}s]"
        )
    
    start_idx = int(indices[0])
    end_idx = int(indices[-1])
    
    print(f"  Time window [{window_start:.3f}s, {window_end:.3f}s] → frames [{start_idx}, {end_idx}] ({end_idx - start_idx + 1} frames)")
    
    return start_idx, end_idx


def _parse_index_value(value, param_name: str, max_val: int, keyframe_num: int, label: str):
    """
    Parse an index value that can be either int or [start, end].
    
    Args:
        value: Either int or list [start, end]
        param_name: 'clean_idx' or 'fog_idx' for error messages
        max_val: Maximum valid index
        keyframe_num: Keyframe number for error messages
        label: Label for error messages
    
    Returns:
        tuple: (start, end) where end is inclusive
    
    Raises:
        ValueError: If validation fails
    """
    if isinstance(value, int):
        # Single frame
        if value < 0 or value >= max_val:
            raise ValueError(
                f"Keyframe {keyframe_num} ({label}): {param_name}={value} out of bounds [0, {max_val-1}]"
            )
        return (value, value)
    
    elif isinstance(value, list):
        # Range [start, end]
        if len(value) != 2:
            raise ValueError(
                f"Keyframe {keyframe_num} ({label}): {param_name} range must have exactly 2 elements [start, end], got {len(value)}"
            )
        
        start, end = value
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"Keyframe {keyframe_num} ({label}): {param_name} range values must be integers, got {type(start).__name__}, {type(end).__name__}"
            )
        
        if start > end:
            raise ValueError(
                f"Keyframe {keyframe_num} ({label}): {param_name} range start ({start}) > end ({end})"
            )
        
        if start < 0 or end >= max_val:
            raise ValueError(
                f"Keyframe {keyframe_num} ({label}): {param_name} range [{start}, {end}] out of bounds [0, {max_val-1}]"
            )
        
        return (start, end)
    
    else:
        raise ValueError(
            f"Keyframe {keyframe_num} ({label}): {param_name} must be int or [start, end], got {type(value).__name__}"
        )


def load_keyframes(keyframe_path: Path, clean_frames: List[Dict], fog_frames: List[Dict], 
                   timestamp_tolerance: float = 0.1, validate_offset: bool = True) -> Tuple[List[KeyframeSequence], Optional[Dict]]:
    """
    Load and validate keyframes from YAML file.
    
    Supports TWO modes:
    
    1. TIMESTAMP MODE (recommended):
       - clean_time: 8.129, fog_time: 28.376  (single landmark)
       - clean_time: [8.129, 10.0], fog_time: [28.376, 10.0]  (time window)
       Auto-computes time offset and finds matching frames
    
    2. INDEX MODE (legacy):
       - clean_idx: 50, fog_idx: 42  (single frame)
       - clean_idx: [5, 100], fog_idx: [10, 105]  (frame range)
       Direct frame access by index
    
    Args:
        keyframe_path: Path to keyframes.yaml
        clean_frames: List of clean frame dicts with 'timestamp' key
        fog_frames: List of fog frame dicts with 'timestamp' key
        timestamp_tolerance: Max time difference for frame matching (seconds)
        validate_offset: If True, checks time offset consistency across landmarks
    
    Returns:
        Tuple of:
        - List of KeyframeSequence objects
        - Dict with alignment info (offset, validation) or None if index mode
    
    Raises:
        ValueError: If timestamps out of range or offset inconsistent
    """
    n_clean = len(clean_frames)
    n_fog = len(fog_frames)
    
    if not keyframe_path.exists():
        raise FileNotFoundError(f"Keyframes file not found: {keyframe_path}")
    
    with open(keyframe_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'keyframes' not in data:
        raise ValueError(f"keyframes.yaml must contain 'keyframes' list")
    
    sequences = []
    total_frame_pairs = 0
    
    # Detect mode: timestamp vs index
    first_kf = data['keyframes'][0] if data['keyframes'] else {}
    use_timestamps = 'clean_time' in first_kf or 'fog_time' in first_kf
    use_indices = 'clean_idx' in first_kf or 'fog_idx' in first_kf
    
    if use_timestamps and use_indices:
        raise ValueError("Keyframes cannot mix timestamp mode (clean_time/fog_time) and index mode (clean_idx/fog_idx)")
    
    if not use_timestamps and not use_indices:
        raise ValueError("Keyframes must specify either timestamps (clean_time/fog_time) or indices (clean_idx/fog_idx)")
    
    alignment_info = None
    
    # ========== TIMESTAMP MODE ==========
    if use_timestamps:
        print("\n📍 Timestamp-based keyframe selection mode")
        offsets = []  # Track offsets for validation
        
        for i, kf in enumerate(data['keyframes']):
            clean_time = kf.get('clean_time')
            fog_time = kf.get('fog_time')
            label = kf.get('label', f'keyframe_{i:03d}')
            
            # Parse optional ROI
            raw_roi = kf.get('roi', None)
            roi = validate_roi(raw_roi, label) if raw_roi else None
            
            if clean_time is None or fog_time is None:
                raise ValueError(f"Keyframe {i}: missing clean_time or fog_time in timestamp mode")
            
            # Handle single timestamp or [reference_time, window_duration]
            if isinstance(clean_time, (int, float)) and isinstance(fog_time, (int, float)):
                # Single landmark timestamp
                clean_ref = float(clean_time)
                fog_ref = float(fog_time)
                
                # Find nearest frames
                clean_idx = find_nearest_frame_by_timestamp(clean_frames, clean_ref, timestamp_tolerance)
                fog_idx = find_nearest_frame_by_timestamp(fog_frames, fog_ref, timestamp_tolerance)
                
                # Record offset
                offset = fog_ref - clean_ref
                offsets.append((label, offset, clean_ref, fog_ref))
                
                # Create single frame pair
                frame_pairs = [(clean_idx, fog_idx, label)]
                is_sequence = False
                
                print(f"  ✓ {label}: clean {clean_ref:.3f}s → fog {fog_ref:.3f}s (offset: {offset:.3f}s, frames: {clean_idx}/{fog_idx})")
                
            elif isinstance(clean_time, list) and isinstance(fog_time, list):
                # Time window: [reference_time, duration]
                if len(clean_time) != 2 or len(fog_time) != 2:
                    raise ValueError(f"Keyframe {i} ({label}): time window must be [reference_time, duration]")
                
                clean_ref, clean_duration = float(clean_time[0]), float(clean_time[1])
                fog_ref, fog_duration = float(fog_time[0]), float(fog_time[1])
                
                if clean_duration != fog_duration:
                    raise ValueError(f"Keyframe {i} ({label}): window durations must match ({clean_duration} vs {fog_duration})")
                
                # Record offset
                offset = fog_ref - clean_ref
                offsets.append((label, offset, clean_ref, fog_ref))
                
                # Extract time windows
                print(f"  Extracting {clean_duration}s window for {label}:")
                print(f"    Clean reference: {clean_ref:.3f}s")
                clean_start, clean_end = extract_time_window(clean_frames, clean_ref, clean_duration, timestamp_tolerance)
                
                print(f"    Fog reference: {fog_ref:.3f}s")
                fog_start, fog_end = extract_time_window(fog_frames, fog_ref, fog_duration, timestamp_tolerance)
                
                # Generate frame pairs (must have equal lengths)
                clean_count = clean_end - clean_start + 1
                fog_count = fog_end - fog_start + 1
                
                if clean_count != fog_count:
                    raise ValueError(
                        f"Keyframe {i} ({label}): extracted frame counts don't match "
                        f"(clean: {clean_count}, fog: {fog_count}). Adjust window duration or tolerance."
                    )
                
                frame_pairs = []
                for j in range(clean_count):
                    c_idx = clean_start + j
                    f_idx = fog_start + j
                    pair_label = f"{label}_frame_{j:03d}"
                    frame_pairs.append((c_idx, f_idx, pair_label))
                
                is_sequence = True
                print(f"  ✓ {label}: {clean_count} frame pairs (offset: {offset:.3f}s)")
            else:
                raise ValueError(f"Keyframe {i} ({label}): clean_time and fog_time must both be scalars or both be [ref, duration] lists")
            
            sequences.append(KeyframeSequence(frame_pairs, label, is_sequence, roi))
            if roi:
                print(f"    ROI: x[{roi.get('x_min','')},{roi.get('x_max','')}] "
                      f"y[{roi.get('y_min','')},{roi.get('y_max','')}] "
                      f"z[{roi.get('z_min','')},{roi.get('z_max','')}]")
            total_frame_pairs += len(frame_pairs)
        
        # Validate offset consistency
        if validate_offset and len(offsets) > 1:
            offset_values = [o[1] for o in offsets]
            mean_offset = np.mean(offset_values)
            std_offset = np.std(offset_values)
            max_deviation = np.max(np.abs(offset_values - mean_offset))
            
            print(f"\n  Offset validation:")
            print(f"    Mean offset: {mean_offset:.3f}s")
            print(f"    Std deviation: {std_offset:.3f}s")
            print(f"    Max deviation: {max_deviation:.3f}s")
            
            if max_deviation > 0.1:
                print(f"    ⚠ WARNING: Large offset variation detected! Check if clocks were stable.")
                for label, offset, ct, ft in offsets:
                    dev = offset - mean_offset
                    print(f"      {label}: {offset:.3f}s (deviation: {dev:+.3f}s)")
            else:
                print(f"    ✓ Offset consistent across landmarks")
            
            alignment_info = {
                'mean_offset': mean_offset,
                'std_offset': std_offset,
                'max_deviation': max_deviation,
                'offsets': offsets
            }
        elif len(offsets) == 1:
            alignment_info = {
                'mean_offset': offsets[0][1],
                'std_offset': 0.0,
                'max_deviation': 0.0,
                'offsets': offsets
            }
    
    # ========== INDEX MODE (LEGACY) ==========
    else:
        print("\n🔢 Index-based keyframe selection mode (legacy)")
        
        for i, kf in enumerate(data['keyframes']):
            clean_idx = kf.get('clean_idx')
            fog_idx = kf.get('fog_idx')
            label = kf.get('label', f'keyframe_{i:03d}')
            
            # Parse optional ROI
            raw_roi = kf.get('roi', None)
            roi = validate_roi(raw_roi, label) if raw_roi else None
            
            if clean_idx is None or fog_idx is None:
                raise ValueError(f"Keyframe {i}: missing clean_idx or fog_idx")
            
            # Parse indices (handles both int and [start, end])
            clean_start, clean_end = _parse_index_value(clean_idx, 'clean_idx', n_clean, i, label)
            fog_start, fog_end = _parse_index_value(fog_idx, 'fog_idx', n_fog, i, label)
            
            # Calculate range lengths
            clean_len = clean_end - clean_start + 1
            fog_len = fog_end - fog_start + 1
            
            # Validate range lengths match
            if clean_len != fog_len:
                raise ValueError(
                    f"Keyframe {i} ({label}): clean range length ({clean_len}) != fog range length ({fog_len}). "
                    f"Ranges must have same number of frames."
                )
            
            # Generate frame pairs
            frame_pairs = []
            for offset in range(clean_len):
                c_idx = clean_start + offset
                f_idx = fog_start + offset
                pair_label = f"{label}_frame_{offset:03d}" if clean_len > 1 else label
                frame_pairs.append((c_idx, f_idx, pair_label))
            
            is_sequence = (clean_len > 1)
            sequences.append(KeyframeSequence(frame_pairs, label, is_sequence, roi))
            if roi:
                print(f"    ROI: x[{roi.get('x_min','')},{roi.get('x_max','')}] "
                      f"y[{roi.get('y_min','')},{roi.get('y_max','')}] "
                      f"z[{roi.get('z_min','')},{roi.get('z_max','')}]")
            total_frame_pairs += len(frame_pairs)
    
    # Print summary
    print(f"\n✓ Loaded {len(sequences)} keyframe sequence(s) from {keyframe_path.name}")
    print(f"  Total frame pairs: {total_frame_pairs}")
    
    return sequences, alignment_info
