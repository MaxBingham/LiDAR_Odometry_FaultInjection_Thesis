"""
Lidar Fault Injection (lfi) package.

Provides fault injection models for simulating sensor degradation:
- Fog/visibility degradation (FOG_Injector)
- Rain effects (RAIN_Injection)

Main entry point: apply_fault_model()
"""

from .apply_fault_model import apply_fault_model, save_fault_stats, load_fault_stats, reset_fault_stats

__all__ = [
    'apply_fault_model',
    'save_fault_stats',
    'load_fault_stats',
    'reset_fault_stats',
]
