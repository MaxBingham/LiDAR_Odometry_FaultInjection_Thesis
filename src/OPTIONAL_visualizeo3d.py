import numpy as np
import open3d as o3d

# Paths to be changed manually
POINTCLOUD_PATH = "data/07/velodyne/000100.bin"
TRAJECTORY_PATH = "results/est_poses_sigma_0.01_seed_0.txt"


def load_kitti_bin(path):
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]


def load_kitti_poses(pose_file):
    """Load KITTI poses and return XYZ positions"""
    poses = np.loadtxt(pose_file)
    # Extract translation: columns 3, 7, 11
    positions = poses[:, [3, 7, 11]]
    return positions


def visualize_pointcloud():
    points = load_kitti_bin(POINTCLOUD_PATH)
    print(f"Loaded {points.shape[0]} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=1200, height=800)


def visualize_trajectory():
    positions = load_kitti_poses(TRAJECTORY_PATH)
    print(f"Loaded {len(positions)} poses")

    # Create line connecting all poses
    lines = [[i, i+1] for i in range(len(positions)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

    o3d.visualization.draw_geometries([line_set], window_name="Trajectory", width=1200, height=800)


if __name__ == "__main__":
    # Change this to switch mode
    MODE = "trajectory"  # "pointcloud" or "trajectory"
    
    if MODE == "pointcloud":
        visualize_pointcloud()
    elif MODE == "trajectory":
        visualize_trajectory()
