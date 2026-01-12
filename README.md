Structure: 

fault_injector_minimal/
├── configs/ # Fog configuration files
│ ├── fog_200m.yaml # Very clear (200m visibility)
│ ├── fog_very_light.yaml # Minimal fog (150m visibility)
│ ├── fog_light.yaml # Light fog (100m visibility)
│ ├── fog_medium.yaml # Medium fog (50m visibility)
│ ├── fog_heavy.yaml # Heavy fog (20m visibility)
│ └── fog_very_heavy.yaml # Severe fog (10m visibility)
│
├── src/ # Source code
│ ├── evaluate.py # Main evaluation script
│ ├── batch_evaluate.py # Batch processing for multiple configs
│ ├── FOG_Injector.py # Physics-based fog simulation
│ ├── fog_dataset.py # Dataset wrapper with fault injection
│ └── metrics.py # Trajectory error metrics (APE/RPE)
│
├── testing/ # Testing utilities
│ ├── TEST_Noise.py # Gaussian noise simulator
│ └── OPTIONAL_visualizeo3d.py # Point cloud/trajectory visualization
│
├── data/ # KITTI dataset (not tracked)
│ ├── velodyne/ # .bin point cloud files
│ └── poses/ # Ground truth poses (.txt)
│
└── results/ # Output directory (auto-created)
├── *.txt # Estimated poses (KITTI format)
├── *.csv # Aggregated metrics
└── YYYY-MM-DD_HH-MM-SS/ # Timestamped run folders


Dependencies: 

Copy: 

```bash
pip install kiss-icp evo numpy open3d pyyaml

How to run: 

Copy: BATCH EVAL WILL FOLLOW

python src/evaluate.py \
  --data data/velodyne \
  --gt data/poses/07.txt \
  --sigma 0.01 \
  --fault-type fog \
  --visibility 50.0 \
  --distance 10.0 \
  --output results/fog_metrics.csv


--> --data: path to velodyne folder; --gt: Path to Ground Truth; 
    --sigma: std for gaussian noise -> When not using gaussian noise set to zero! 

Output: 

Estimated poses: results/est_poses_sigma_{value}_seed_{n}.txt (KITTI format)
Metrics CSV: Contains APE RMSE, RPE RMSE, and fog statistics per run
Timestamped folders: Detailed pose outputs in TUM/KITTI/NumPy formats