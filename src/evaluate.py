#!/usr/bin/env python3

import csv
import logging
import numpy as np
from pathlib import Path
import argparse
import sys, os

# Ensure local src folder is on sys.path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kiss_icp.pipeline import OdometryPipeline
from fog_dataset import FogDataset
from FOG_Injector import FogSimulator
from RAIN_Injection import RainSimulator

from metrics import evo_ape_kitti, evo_rpe_kitti
from fog_dataset import load_scan

#possible need to save kiss icp results in kitti format for evo 
def save_poses_kitti(path: Path, poses: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    poses_3x4 = poses[:, :3, :4]
    np.savetxt(path, poses_3x4.reshape(len(poses), 12))

#Run ICP Function  

def run_odometry(dataset, visualize=True):
    pipeline = OdometryPipeline(dataset=dataset, visualize=visualize)
    pipeline.run()
    return pipeline.poses


def main (): 
    ap = argparse.ArgumentParser(
        description="Fog robustness evaluation using KISS-ICP + evo"
    )

    ap.add_argument("--data", required=True,
                    help="Path to KITTI velodyne folder")
    ap.add_argument("--gt", required=True,
                    help="Path to KITTI ground-truth poses (.txt)")
    ap.add_argument("--sigma", type=float, nargs="+", required=True,
                    help="Gaussian Noise std in meters")
    ap.add_argument("--output", default="results/fog_metrics.csv")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="Skip evo metrics and CSV saving (only run Kiss-ICP)")
    ap.add_argument('--fault-type', type=str, choices=['fog','rain'], default='fog',
                    help="Type of fault to inject: 'fog' or 'rain'")
    ap.add_argument('--visibility', type=float, default=50.0,
                    help='Visibility distance for fog simulation (meters)')
    ap.add_argument('--distance', type=float, default=10.0,
                    help='Distance parameter for fog simulation')
    ap.add_argument('--visualize', action='store_true',
                    help='Enable Kiss-ICP visualizer (verbose)')
    ap.add_argument('--rain-rate', type=float, default=10.0,
                    help='Rain rate in mm/h (used when --fault-type rain)')
    ap.add_argument('--voxel-size', type=float, default=0.05,
                    help='Voxel downsample size in meters (passed to dataset)')

    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    #Control the path works

    


    csv_path = Path(args.output)

#csv writer 
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id",
                "fault_type",
                "sigma",
                "ape_rmse",
                "rpe_rmse",
                "total_points",
                "deleted_points",
                "backscattered_points",
                "p_delete",
                "lambda_m"
            ])

#loop
    run_id = 0
    for sigma in args.sigma:
        run_id += 1

        #1 Create simulator based on fault type
        if args.fault_type == 'fog':
            simulator = FogSimulator(V=args.visibility)
        elif args.fault_type == 'rain':
            simulator = RainSimulator(args.rain_rate, d_min=args.distance)
        else:
            raise ValueError(f"Unknown fault type: {args.fault_type}")

        #2 dataset
        dataset = FogDataset(data_dir=args.data, modifier=simulator, voxel_size=args.voxel_size)

        #3 run odometry
        poses = run_odometry(dataset, visualize=args.visualize)
        est_path = out_dir / f"est_poses_sigma_{sigma:.2f}.txt"
        save_poses_kitti(est_path, poses)
        
        # Collect fault statistics
        if args.fault_type == 'fog':
            total_pts = simulator.stats['total']
            deleted_pts = simulator.stats['deleted']
            backscattered_pts = simulator.stats['backscattered']
            # Calculate p_delete using the formula (with clipping)
            p_delete_val = 1 + simulator.a * np.exp(simulator.b * simulator.V)
            p_delete_val = np.clip(p_delete_val, 0, 1)
            lambda_val = simulator.lambda_
        elif args.fault_type == 'rain':
            stats = simulator.get_statistics()
            total_pts = stats.get('total', 0)
            deleted_pts = stats.get('deleted', 0)
            backscattered_pts = stats.get('backscattered', 0)
            p_delete_val = stats.get('delete_rate', 0)
            lambda_val = 0
        else:
            total_pts = 0
            deleted_pts = 0
            backscattered_pts = 0
            p_delete_val = 0
            lambda_val = 0
        
        logging.info(f"[Run {run_id}] fault_type={args.fault_type}, sigma={sigma:.3f}")
        logging.info(f"  Saved: {est_path}")
        if args.fault_type == 'fog':
            logging.info(f"  Fog Stats: Total={total_pts}, Deleted={deleted_pts}, Backscattered={backscattered_pts}")
            logging.info(f"  p_delete={p_delete_val:.3f}, lambda={lambda_val:.3f}m")
        elif args.fault_type == 'rain':
            logging.info(f"  Rain Stats: Total={total_pts}, Deleted={deleted_pts}, Backscattered={backscattered_pts}")
            logging.info(f"  delete_rate={p_delete_val:.3f}, rain_rate={args.rain_rate:.1f} mm/h")

        if not args.skip_metrics: #Can skip metrics (currently not working) by using argument: --skip-metrics
            # 4) Run evo metrics (using Python API directly - much faster!)
            ape_stats = evo_ape_kitti(gt=args.gt, est=est_path)
            rpe_stats = evo_rpe_kitti(gt=args.gt, est=est_path, delta_m=1.0)

            ape_rmse = ape_stats["rmse"]
            rpe_rmse = rpe_stats["rmse"]

            logging.info(f"  APE RMSE: {ape_rmse:.3f} m")
            logging.info(f"  RPE RMSE: {rpe_rmse:.3f} m/m")

            #csv
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_id,
                    args.fault_type,
                    sigma,
                    ape_rmse,
                    rpe_rmse,
                    total_pts,
                    deleted_pts,
                    backscattered_pts,
                    p_delete_val,
                    lambda_val
                ])
    if not args.skip_metrics:
        logging.info(f"\n✓ Done. Results saved to {csv_path}")
    else:
        logging.info(f"\n✓ Done. Poses saved to results/")
if __name__ == "__main__":
    main()