import numpy as np
from typing import Tuple, Dict
from scipy.spatial import cKDTree


def modified_hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Computes the Modified Hausdorff Distance between two point clouds.
    
    The Modified Hausdorff Distance is defined as the average of the 
    minimum distances from each point in pc1 to pc2 and vice versa.
    
    Parameters
    ----------
    pc1 : np.ndarray
        First point cloud (shape: (N, >=3))
    pc2 : np.ndarray
        Second point cloud (shape: (M, >=3))
    
    Returns
    -------
    float
        Modified Hausdorff Distance in meters
    """
    
    if len(pc1) == 0 or len(pc2) == 0:
        return np.inf
    
    # Use only xyz coordinates
    pc1_xyz = pc1[:, :3]
    pc2_xyz = pc2[:, :3]
    

    # Build KD-Trees
    tree_2 = cKDTree(pc2_xyz)
    tree_1 = cKDTree(pc1_xyz)

    # Nearest-neighbor distances
    distances_1_to_2, _ = tree_2.query(pc1_xyz)
    distances_2_to_1, _ = tree_1.query(pc2_xyz)

    mean_1_to_2 = np.mean(distances_1_to_2)
    mean_2_to_1 = np.mean(distances_2_to_1)

    return max(mean_1_to_2, mean_2_to_1)


def compare_point_clouds_hausdorff(data_list: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Computes Modified Hausdorff Distance between reference and other point clouds.
    
    Parameters
    ----------
    data_list : Dict[str, np.ndarray]
        Dictionary with labels as keys and point clouds (np.ndarray of shape (N, >=3)) as values.
        The first entry is assumed to be the reference point cloud.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with comparison labels as keys and MHD values as values.
        Keys follow the format: "Reference vs <label>"
    """
    
    labels = list(data_list.keys())
    point_clouds = [data_list[label] for label in labels]
    
    # First point cloud is the reference
    ref_pc = point_clouds[0]
    ref_label = labels[0]
    
    results = {}
    
    # Compare reference with all other point clouds
    for i in range(1, len(point_clouds)):
        comparison_label = f"{ref_label} vs {labels[i]}"
        mhd = modified_hausdorff_distance(ref_pc, point_clouds[i])
        results[comparison_label] = mhd
    
    return results
