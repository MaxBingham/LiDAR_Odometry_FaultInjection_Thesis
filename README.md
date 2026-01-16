# LiDAR Odometry Fault Injection - KITTI Dataset

Physics-based fog simulation for LiDAR odometry evaluation using Kiss-ICP on KITTI sequences.

## Project Structure

```
fault_injector_minimal/
├── configs/                    # Fog configuration files
│   ├── fog_200m.yaml          # Very clear (200m visibility)
│   ├── fog_very_light.yaml    # Minimal fog (150m visibility)
│   ├── fog_light.yaml         # Light fog (100m visibility)
│   ├── fog_medium.yaml        # Medium fog (50m visibility)
│   ├── fog_heavy.yaml         # Heavy fog (20m visibility)
│   └── fog_very_heavy.yaml    # Severe fog (10m visibility)
│
├── src/                        # Source code
│   ├── evaluate.py            # Main evaluation script
│   ├── batch_evaluate.py      # Batch processing for multiple configs
│   ├── FOG_Injector.py        # Physics-based fog simulation
│   ├── fog_dataset.py         # Dataset wrapper with fault injection
│   └── metrics.py             # Trajectory error metrics (APE/RPE)
│
├── testing/                    # Testing utilities
│   ├── TEST_Noise.py          # Gaussian noise simulator
│   └── OPTIONAL_visualizeo3d.py # Point cloud/trajectory visualization
│
├── data/                       # KITTI dataset (not tracked)
│   ├── 07/velodyne/           # .bin point cloud files
│   └── 07.txt                 # Ground truth poses
│
└── results/                    # Output directory (auto-created)
    ├── *.txt                  # Estimated poses (KITTI format)
    ├── *.csv                  # Aggregated metrics
    └── *.zip                  # EVO result archives
```

## Dependencies

```bash
pip install kiss-icp evo numpy open3d pyyaml
```

## Usage

### Single Evaluation Run

```bash
python src/evaluate.py \
  --data data/07/velodyne \
  --gt data/07.txt \
  --sigma 0.0 \
  --fault-type fog \
  --visibility 50.0 \
  --distance 100.0 \
  --output results/fog_metrics.csv
```

### Arguments

- `--data`: Path to KITTI velodyne folder
- `--gt`: Path to ground truth poses (.txt)
- `--sigma`: Standard deviation for Gaussian noise (set to 0.0 when using fog)
- `--fault-type`: Choose `fog` or `gaussian`
- `--visibility`: Fog visibility distance in meters (V)
- `--distance`: Distance parameter for fog simulation
- `--output`: CSV output path
- `--skip-metrics`: Skip evo metrics calculation (faster, poses only)

### Batch Evaluation (Coming Soon)

```bash
python src/batch_evaluate.py
```

## Output

- **Estimated poses**: `results/est_poses_sigma_{value}.txt` (KITTI format)
- **Metrics CSV**: Contains APE RMSE, RPE RMSE, and fog statistics per run
- **Statistics**: Total points, deleted points, backscattered points, p_delete, lambda

## Fog Model

Implements Beer-Lambert law for atmospheric attenuation:
- **Point deletion**: Extinction based on visibility
- **Backscattering**: Exponential range distribution
- **Atmospheric turbulence**: Distance-dependent geometric distortion

Parameters automatically computed from visibility (V) using empirical models.