import numpy as np
from typing import Dict
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


def compute_surface_normals(pc: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Computes surface normals for a point cloud using local PCA.
    
    Parameters
    ----------
    pc : np.ndarray
        Point cloud (shape: (N, >=3))
    k : int
        Number of neighbors to use for normal estimation
    
    Returns
    -------
    np.ndarray
        Surface normals (shape: (N, 3))
    """
    
    if len(pc) < k:
        # If point cloud is too small, use simple z-direction normals
        normals = np.zeros((len(pc), 3))
        normals[:, 2] = 1.0
        return normals
    
    pc_xyz = pc[:, :3]
    tree = cKDTree(pc_xyz)
    normals = np.zeros((len(pc_xyz), 3))
    
    for i, point in enumerate(pc_xyz):
        # Find k nearest neighbors
        _, indices = tree.query(point, k=k)
        neighbors = pc_xyz[indices]
        
        # Center the neighbors
        centered = neighbors - neighbors.mean(axis=0)
        
        # Compute PCA to get the surface normal (smallest eigenvector)
        try:
            pca = PCA(n_components=3)
            pca.fit(centered)
            # The normal is the eigenvector with smallest eigenvalue
            normal = pca.components_[-1]
            normals[i] = normal
        except:
            # Fallback to z-direction
            normals[i] = np.array([0, 0, 1])
    
    return normals


def standard_deviation_along_surface_normal(pc: np.ndarray, k: int = 10) -> float:
    """
    Computes SDASN for a point cloud according to the standard definition:
    SDASN = sqrt(smallest eigenvalue of local PCA), aggregated over all points.
    """

    if len(pc) < k:
        return np.nan

    pc_xyz = pc[:, :3]
    tree = cKDTree(pc_xyz)

    sdasn_per_point = []

    for point in pc_xyz:
        _, indices = tree.query(point, k=k)
        neighbors = pc_xyz[indices]

        centered = neighbors - neighbors.mean(axis=0)

        try:
            pca = PCA(n_components=3)
            pca.fit(centered)

            lambda_min = pca.explained_variance_[-1]
            sdasn_per_point.append(np.sqrt(lambda_min))

        except:
            continue

    # Robust aggregation (empfohlen)
    return np.median(sdasn_per_point)



def compute_point_clouds_sdasn(data_list: Dict[str, np.ndarray], k: int = 10) -> Dict[str, float]:
    """
    Computes SDASN for each point cloud independently.
    
    Parameters
    ----------
    data_list : Dict[str, np.ndarray]
        Dictionary with labels as keys and point clouds (np.ndarray of shape (N, >=3)) as values.
    k : int
        Number of neighbors for normal estimation
    
    Returns
    -------
    Dict[str, float]
        Dictionary with labels as keys and SDASN values as values.
    """
    
    results = {}
    
    for label, pc in data_list.items():
        sdasn = standard_deviation_along_surface_normal(pc, k=k)
        results[label] = sdasn
    
    return results
