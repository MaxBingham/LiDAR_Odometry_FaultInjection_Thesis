from pathlib import Path
import os
import numpy as np
import json
from lfi.fault_models.FOG_Injector import FogSimulator
from lfi.fault_models.RAIN_Injection import RainSimulator

# Simulator cache
_SIM_CACHE = {
    "fog": {},
    "rain": {},
}

# Global stats (single source of truth)
_FAULT_STATS = {
    "total_frames": 0,
    "total_points": 0,
    "total_deleted": 0,
    "total_backscattered": 0,
    "total_modified": 0,
}

_STATS_FILE = Path(os.environ.get("LFI_STATS_FILE", "/tmp/lfi_stats.json"))
_DISTANCE_STATS_FILE = Path(os.environ.get("LFI_DISTANCE_STATS_FILE", "/tmp/lfi_distance_stats.json"))


def apply_fault_model(
    points: np.ndarray,
    faultmodel: str | None,
    visibility: float = 0.1,
    rain_rate: float = 10.0,
    fog_metric: str = "distance",
) -> np.ndarray:
    """
    Apply a fault model to a LiDAR point cloud and update global statistics.
    
    Args:
        points: (N, 3) array of xyz coordinates
        faultmodel: Type of fault model ("fog", "rain", "none")
        visibility: Visibility in meters (for fog)
        rain_rate: Rain rate in mm/h (for rain)
        fog_metric: Fog parameterization to use ("distance" or "chamfer")
    """

    if faultmodel is None or str(faultmodel).lower() == "none":
        return points

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n_in = points.shape[0]

    # --- Fog ----------------------------------------------------------------
    if faultmodel == "fog":
        cache_key = (visibility, fog_metric)
        sim = _SIM_CACHE["fog"].get(cache_key)
        if sim is None:
            sim = FogSimulator(visibility, metric=fog_metric)
            _SIM_CACHE["fog"][cache_key] = sim

        result = sim.apply_noise_fog(points)

        _FAULT_STATS["total_frames"] += 1
        _FAULT_STATS["total_points"] += n_in
        _FAULT_STATS["total_deleted"] += n_in - result.shape[0]
        _FAULT_STATS["total_backscattered"] += sim.stats.get("backscattered", 0)
        _FAULT_STATS["total_modified"] += sim.stats.get("modified", 0)
        save_fault_stats()
        save_distance_stats(visibility, fog_metric)

        return result

    # --- Rain ---------------------------------------------------------------
    if faultmodel == "rain":
        sim = _SIM_CACHE["rain"].get(rain_rate)
        if sim is None:
            sim = RainSimulator(rain_rate)
            _SIM_CACHE["rain"][rain_rate] = sim

        result = sim.apply_noise_rain(points)

        _FAULT_STATS["total_frames"] += 1
        _FAULT_STATS["total_points"] += n_in
        _FAULT_STATS["total_deleted"] += n_in - result.shape[0]
        _FAULT_STATS["total_backscattered"] += sim.stats.get("backscattered", 0)
        _FAULT_STATS["total_modified"] += sim.stats.get("modified", 0)
        save_fault_stats()

        return result

    raise ValueError(f"Unknown fault model: {faultmodel}")


# ---------------------------------------------------------------------------

def reset_fault_stats() -> None:
    _FAULT_STATS.clear()
    _FAULT_STATS.update({
        "total_frames": 0,
        "total_points": 0,
        "total_deleted": 0,
        "total_backscattered": 0,
        "total_modified": 0,
    })
    if _STATS_FILE.exists():
        _STATS_FILE.unlink()
    if _DISTANCE_STATS_FILE.exists():
        _DISTANCE_STATS_FILE.unlink()


def save_fault_stats() -> None:
    _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATS_FILE, "w") as f:
        json.dump(_FAULT_STATS, f, indent=2)


def load_fault_stats() -> dict:
    if _STATS_FILE.exists():
        with open(_STATS_FILE, "r") as f:
            return json.load(f)
    return {}


def get_fog_simulator(visibility: float, fog_metric: str = "distance"):
    """Get the cached FogSimulator instance for accessing distance statistics.
    
    Returns:
        FogSimulator instance or None if not found in cache
    """
    cache_key = (float(visibility), str(fog_metric))  # Ensure consistent types
    return _SIM_CACHE["fog"].get(cache_key)


def save_distance_stats(visibility: float, fog_metric: str = "distance") -> None:
    """Save distance statistics from fog simulator to file."""
    # Direct cache lookup with consistent types
    cache_key = (float(visibility), str(fog_metric))
    sim = _SIM_CACHE["fog"].get(cache_key)
    
    if sim is not None:
        distance_stats = sim.get_distance_statistics()
        if distance_stats:
            _DISTANCE_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_DISTANCE_STATS_FILE, "w") as f:
                json.dump(distance_stats, f, indent=2)


def load_distance_stats() -> list:
    """Load distance statistics from file."""
    if _DISTANCE_STATS_FILE.exists():
        with open(_DISTANCE_STATS_FILE, "r") as f:
            return json.load(f)
    return []
