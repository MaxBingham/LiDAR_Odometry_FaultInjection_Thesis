"""
MCAP loading with strict schema validation.
"""

import struct
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    raise ImportError("Install mcap support: pip install mcap mcap-ros2-support")


class McapLoader:
    """Load and validate MCAP LiDAR data."""
    
    def __init__(self, mcap_path: Path, topic_name: str, intensity_field_priority: List[str] = None):
        """
        Initialize MCAP loader with strict validation.
        
        Args:
            mcap_path: Path to MCAP file
            topic_name: Exact topic name
            intensity_field_priority: Field names to try for intensity ['intensity', 'i', 'reflectivity']
        """
        self.mcap_path = Path(mcap_path)
        self.topic_name = topic_name
        self.intensity_field_priority = intensity_field_priority or ['intensity', 'i', 'reflectivity']
        
        if not self.mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {mcap_path}")
        
        # Validate and inspect
        self._validate_mcap()
        
    def _validate_mcap(self):
        """Validate MCAP contains required topic and intensity field."""
        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            
            # Check topic exists
            topic_found = False
            for channel in summary.channels.values():
                if channel.topic == self.topic_name:
                    topic_found = True
                    break
            
            if not topic_found:
                available = [ch.topic for ch in summary.channels.values()]
                raise ValueError(
                    f"Topic '{self.topic_name}' not found in {self.mcap_path.name}.\n"
                    f"Available topics: {available}"
                )
            
            # Check first message for intensity field
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if channel.topic != self.topic_name:
                    continue
                
                if not hasattr(ros_msg, 'fields'):
                    raise ValueError(f"Message on {self.topic_name} has no 'fields' attribute")
                
                field_names = [f.name for f in ros_msg.fields]
                
                # Find intensity field
                self.intensity_field = None
                for candidate in self.intensity_field_priority:
                    if candidate in field_names:
                        self.intensity_field = candidate
                        break
                
                if self.intensity_field is None:
                    raise ValueError(
                        f"No intensity field found in {self.mcap_path.name}.\n"
                        f"Available fields: {field_names}\n"
                        f"Tried: {self.intensity_field_priority}"
                    )
                
                print(f"✓ MCAP validated: {self.mcap_path.name}")
                print(f"  Topic: {self.topic_name}")
                print(f"  Intensity field: {self.intensity_field}")
                print(f"  Fields: {field_names}")
                break
    
    def load_all_frames(self, max_frames: int = None, skip_seconds: float = 0.0) -> List[Dict]:
        """
        Load frames from MCAP, optionally skipping an initial time window and capping count.

        Args:
            max_frames: Maximum number of frames to return (None = all)
            skip_seconds: Skip frames whose timestamp is within this many seconds
                          of the first message timestamp (0 = no skip)

        Returns:
            List of dicts with keys: xyz (N,3), intensity (N,), timestamp (float)
        """
        frames = []
        invalid_intensity_count = 0
        _first_ts = None  # will be set from the first message seen on the topic
        
        # PointField datatype to (struct format, byte size, numpy dtype)
        DTYPE_MAP = {
            1: ('b', 1, 'i1'),   # INT8
            2: ('B', 1, 'u1'),   # UINT8
            3: ('h', 2, 'i2'),   # INT16
            4: ('H', 2, 'u2'),   # UINT16
            5: ('i', 4, 'i4'),   # INT32
            6: ('I', 4, 'u4'),   # UINT32
            7: ('f', 4, 'f4'),   # FLOAT32
            8: ('d', 8, 'f8'),   # FLOAT64
        }
        
        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if channel.topic != self.topic_name:
                    continue

                # --- skip_seconds / max_frames filtering ---
                msg_ts = message.log_time / 1e9
                if _first_ts is None:
                    _first_ts = msg_ts
                if skip_seconds > 0.0 and (msg_ts - _first_ts) < skip_seconds:
                    continue
                if max_frames is not None and len(frames) >= max_frames:
                    break

                # Parse PointCloud2
                field_map = {field.name: (field.offset, field.datatype) for field in ros_msg.fields}
                point_step = ros_msg.point_step
                n_points = len(ros_msg.data) // point_step
                is_bigendian = ros_msg.is_bigendian
                endian = '>' if is_bigendian else '<'
                
                # Extract XYZ (always FLOAT32 in practice)
                xyz = np.zeros((n_points, 3), dtype=np.float32)
                for i, axis in enumerate(['x', 'y', 'z']):
                    offset, dtype = field_map[axis]
                    fmt, size, _ = DTYPE_MAP[dtype]
                    for j in range(n_points):
                        start = j * point_step + offset
                        xyz[j, i] = struct.unpack(f'{endian}{fmt}', ros_msg.data[start:start+size])[0]
                
                # Extract intensity (CRITICAL FIX: use correct datatype)
                offset, dtype = field_map[self.intensity_field]
                fmt, size, np_dtype = DTYPE_MAP[dtype]
                
                # Use struct format for single-byte types, endian prefix for multi-byte
                if size == 1:
                    struct_fmt = fmt  # No endian prefix for single-byte types
                else:
                    struct_fmt = f'{endian}{fmt}'
                
                intensity = np.zeros(n_points, dtype=np.float32)
                for j in range(n_points):
                    start = j * point_step + offset
                    raw_value = struct.unpack(struct_fmt, ros_msg.data[start:start+size])[0]
                    intensity[j] = float(raw_value)
                
                # Normalize integer intensity to [0, 1]
                if dtype in [1, 2, 3, 4, 5, 6]:  # Integer types
                    if dtype in [1, 3, 5]:  # Signed
                        max_val = 2 ** (size * 8 - 1) - 1
                    else:  # Unsigned
                        max_val = 2 ** (size * 8) - 1
                    intensity = intensity / max_val
                
                # Filter NaN/invalid points (but don't filter intensity==0, that's valid after normalization)
                valid_mask = ~(np.isnan(xyz).any(axis=1) | np.isnan(intensity))
                invalid_count = n_points - valid_mask.sum()
                invalid_intensity_count += invalid_count
                
                if invalid_count > 0 and invalid_count / n_points > 0.05:
                    print(f"⚠ Frame {len(frames)}: {invalid_count}/{n_points} ({100*invalid_count/n_points:.1f}%) invalid points")
                
                xyz = xyz[valid_mask]
                intensity = intensity[valid_mask]
                
                timestamp = msg_ts

                frames.append({
                    'xyz': xyz,
                    'intensity': intensity,
                    'timestamp': timestamp,
                    'n_invalid': invalid_count
                })
        
        if invalid_intensity_count > 0:
            total_points = sum(len(f['xyz']) + f['n_invalid'] for f in frames)
            pct = 100 * invalid_intensity_count / total_points
            print(f"⚠ Total invalid points filtered: {invalid_intensity_count}/{total_points} ({pct:.2f}%)")
        
        print(f"✓ Loaded {len(frames)} frames from {self.mcap_path.name}")
        return frames
