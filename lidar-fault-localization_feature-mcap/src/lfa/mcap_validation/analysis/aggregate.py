"""
Distributional statistics over full sequences.
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy import stats as scipy_stats


def _process_frames(frames, range_bins):
    """Process a list of frames into per-bin counts and intensity aggregates."""
    n_bins = len(range_bins) - 1
    n_frames = len(frames)
    
    counts_per_frame = np.zeros((n_frames, n_bins))
    total_I_per_frame = np.zeros((n_frames, n_bins))
    all_intensities = [[] for _ in range(n_bins)]
    
    for f_idx, frame in enumerate(frames):
        ranges = np.linalg.norm(frame['xyz'], axis=1)
        intensity = frame['intensity']
        
        bin_idx = np.digitize(ranges, range_bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid]
        intensity = intensity[valid]
        
        for i in range(n_bins):
            mask = bin_idx == i
            counts_per_frame[f_idx, i] = mask.sum()
            if mask.sum() > 0:
                total_I_per_frame[f_idx, i] = intensity[mask].sum()
                all_intensities[i].append(intensity[mask])
    
    # Aggregate
    counts_mean = counts_per_frame.mean(axis=0)
    total_I_mean = total_I_per_frame.mean(axis=0)
    
    mean_I = np.full(n_bins, np.nan)
    median_I = np.full(n_bins, np.nan)
    for i in range(n_bins):
        if all_intensities[i]:
            combined = np.concatenate(all_intensities[i])
            mean_I[i] = combined.mean()
            median_I[i] = np.median(combined)
    
    return counts_mean, total_I_mean, mean_I, median_I


def _safe_divide(a, b, fill=np.nan):
    """Divide a/b, returning fill where b <= 0."""
    out = np.full_like(a, fill, dtype=np.float64)
    valid = (b > 0) & np.isfinite(b)
    out[valid] = a[valid] / b[valid]
    return out


def compute_range_binned_statistics(
    clean_frames: List[Dict],
    synth_frames: List[Dict],
    real_frames: List[Dict],
    range_bins: np.ndarray = None
) -> Dict:
    """
    Compute deletion and intensity statistics binned by range.
    
    Returns dict with per-bin: counts, survival ratios,
    mean/median/total intensity, and intensity ratios.
    """
    if range_bins is None:
        range_bins = np.arange(0, 101, 10)
    
    centers = (range_bins[:-1] + range_bins[1:]) / 2
    
    clean_counts, clean_total_I, clean_mean_I, clean_median_I = \
        _process_frames(clean_frames, range_bins)
    synth_counts, synth_total_I, synth_mean_I, synth_median_I = \
        _process_frames(synth_frames, range_bins)
    real_counts, real_total_I, real_mean_I, real_median_I = \
        _process_frames(real_frames, range_bins)
    
    # Survival (raw — not clipped, values > 1 indicate backscatter)
    synth_survival = _safe_divide(synth_counts, clean_counts, fill=1.0)
    real_survival = _safe_divide(real_counts, clean_counts, fill=1.0)
    
    return {
        'range_bins': range_bins,
        'range_centers': centers,
        
        # Point counts (mean per frame)
        'clean_counts': clean_counts,
        'synth_counts': synth_counts,
        'real_counts': real_counts,
        
        # Survival / deletion (raw, unclipped)
        'synth_survival': synth_survival,
        'real_survival': real_survival,
        'synth_deletion': 1 - np.clip(synth_survival, 0, 1),
        'real_deletion': 1 - np.clip(real_survival, 0, 1),
        
        # Mean intensity (subject to survivorship bias at long range)
        'clean_intensity_mean': clean_mean_I,
        'synth_intensity_mean': synth_mean_I,
        'real_intensity_mean': real_mean_I,
        'synth_mean_I_ratio': _safe_divide(synth_mean_I, clean_mean_I),
        'real_mean_I_ratio': _safe_divide(real_mean_I, clean_mean_I),
        
        # Median intensity (robust to retroreflector outliers)
        'clean_intensity_median': clean_median_I,
        'synth_intensity_median': synth_median_I,
        'real_intensity_median': real_median_I,
        'synth_median_I_ratio': _safe_divide(synth_median_I, clean_median_I),
        'real_median_I_ratio': _safe_divide(real_median_I, clean_median_I),
        
        # Total integrated intensity (survivorship-bias-free)
        'clean_total_intensity': clean_total_I,
        'synth_total_intensity': synth_total_I,
        'real_total_intensity': real_total_I,
        'synth_total_I_ratio': _safe_divide(synth_total_I, clean_total_I),
        'real_total_I_ratio': _safe_divide(real_total_I, clean_total_I),
        
        # Backscatter flag
        'real_backscatter_flag': real_survival > 1.05,
    }


def fit_extinction_coefficient(survival_prob: np.ndarray, range_centers: np.ndarray,
                               valid_range: Tuple[float, float] = (10.0, 80.0)) -> Dict:
    """
    Fit extinction coefficient α: S(r) = exp(-2αr)
    
    ln(S) = -2α·r  →  slope = -2α  →  α = -slope/2
    Visibility: V = 3.0/α (Koschmieder)
    """
    mask = ((range_centers >= valid_range[0]) &
            (range_centers <= valid_range[1]) &
            (survival_prob > 0) & np.isfinite(survival_prob))
    
    if np.sum(mask) < 3:
        return {'alpha': np.nan, 'visibility_est': np.nan, 'r_squared': np.nan}
    
    r_valid = range_centers[mask]
    log_s = np.log(survival_prob[mask])
    
    slope, intercept, r_value, _, std_err = scipy_stats.linregress(r_valid, log_s)
    
    alpha = -slope / 2
    
    return {
        'alpha': float(alpha),
        'visibility_est': float(3.0 / alpha) if alpha > 0 else float('inf'),
        'r_squared': float(r_value ** 2),
        'slope': float(slope),
        'intercept': float(intercept),
    }


def compute_curve_similarity(real_survival: np.ndarray, synth_survival: np.ndarray) -> Dict:
    """Compare survival curves: RMSE, MAE, correlation."""
    valid = (np.isfinite(real_survival) & np.isfinite(synth_survival) &
             (real_survival >= 0) & (synth_survival >= 0))
    
    if np.sum(valid) < 2:
        return {'survival_rmse': np.nan, 'survival_mae': np.nan,
                'survival_correlation': np.nan}
    
    r, s = real_survival[valid], synth_survival[valid]
    
    return {
        'survival_rmse': float(np.sqrt(np.mean((r - s) ** 2))),
        'survival_mae': float(np.mean(np.abs(r - s))),
        'survival_correlation': float(np.corrcoef(r, s)[0, 1])
            if np.std(r) > 0 and np.std(s) > 0 else np.nan,
    }
