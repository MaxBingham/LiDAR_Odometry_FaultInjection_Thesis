import numpy as np
from pathlib import Path
import open3d as o3d

"Load original scans" 
def load_scan(filepath) -> np.ndarray:
    scan = np.fromfile(filepath, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan[:, :3]  # Return only xyz


#!!!! possibly error due to missmatch in arrays 
"create foggy dataset wrapper"
class FogDataset:

    def __init__(self, data_dir, modifier=None, voxel_size=0.00001):   #adjust downsample size here

        self.data_dir = Path(data_dir)

        #sorted list of bin files
        self.files = sorted(self.data_dir.glob("*.bin"))

        if not self.files:
            raise ValueError(f"No .bin files found in {data_dir}")

        self.modifier = modifier
        self.voxel_size = voxel_size # stored vsize 

        print(f"Foggy data loaded:{len(self.files)} scans")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        #1 load raw scan 
        points = load_scan(self.files[idx])

        #2 apply fog modifier if any
        if self.modifier is not None:
            points = self.modifier.apply_noise(points)


        #downsample via voxel grid - disable with by commenting out 
        if self.voxel_size is not None and self.voxel_size > 0 and len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            points = np.asarray(pcd.points)



        # Kiss-ICP expects (points, timestamps) tuple
        timestamps = np.zeros(len(points))
        return points, timestamps
        



