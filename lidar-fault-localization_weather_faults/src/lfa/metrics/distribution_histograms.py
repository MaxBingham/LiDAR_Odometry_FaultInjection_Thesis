import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def compute_distribution_histograms(data_list: dict, bin_width: float = 0.5, max_distance: float = 50.0) -> Tuple[np.ndarray, list, list]:
    """
    Computes radial distance histograms for point clouds.

    Parameters
    ----------
    data_list : dict
        Dictionary with labels as keys and point clouds (np.ndarray of shape (N, 4)) as values
    bin_width : float
        Width of distance bins in meters
    max_distance : float
        Maximum distance for binning in meters (default: 50.0)
    
    Returns
    -------
    tuple
        (bins, labels, normalized_histograms)
    """

    labels = list(data_list.keys())
    point_clouds = [data_list[label] for label in labels]

    # Radiale Abstände berechnen
    distances = []
    for pc in point_clouds:
        r = np.linalg.norm(pc[:, :3], axis=1)  # Nur die ersten 3 Spalten (x, y, z)
        distances.append(r)

    # Bins definieren mit fester maximaler Entfernung
    bins = np.arange(0.0, max_distance + bin_width, bin_width)

    # Histogramme berechnen
    histograms = [np.histogram(d, bins=bins)[0] for d in distances]

    # Normierung auf das erste Histogramm (Referenz)
    first_hist = histograms[0]

    # Normalisierung auf Referenz (NaN, wenn ein Bin 0 in Referenz)
    normalized_histograms = []
    for hist in histograms:
        nh = np.empty_like(hist, dtype=float)
        nh[:] = np.nan  # Standard NaN
        mask = first_hist != 0
        nh[mask] = hist[mask] / first_hist[mask] * 100  # nur dort, wo Referenz != 0
        normalized_histograms.append(nh)

    return bins, labels, normalized_histograms
