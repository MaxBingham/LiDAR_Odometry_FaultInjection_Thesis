#!/usr/bin/env python3

import csv
from html import parser
import numpy as np
from pathlib import Path
import argparse

from kiss_icp.pipeline import OdometryPipeline
from fog_dataset import FogDataset
from TEST_Noise import FogSimulator as GaussianNoiseSimulator
from FOG_Injector import FogSimulator

from metrics import evo_ape_kitti, evo_rpe_kitti
from fog_dataset import load_scan

#possible need to save kiss icp results in kitti format for evo 
def save_poses_kitti(path: Path, poses: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    poses_3x4 = poses[:, :3, :4]
    np.savetxt(path, poses_3x4.reshape(len(poses), 12))

#Run ICP Function  

def run_odometry(dataset):
    pipeline = OdometryPipeline(dataset=dataset, visualize=True)
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Random seeds for repeated runs")
    ap.add_argument("--output", default="results/fog_metrics.csv")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="Skip evo metrics and CSV saving (only run Kiss-ICP)")
    ap.add_argument('--fault-type', type=str, choices=['gaussian','fog'], default='gaussian',
                    help="Type of fault to inject: 'gaussian' or 'fog'")
    ap.add_argument('--visibility', type=float, default=50.0,
                    help='Visibility distance for fog simulation (meters)')
    ap.add_argument('--distance', type=float, default=10.0,
                    help='Distance parameter for fog simulation')
    

    args = ap.parse_args()

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
                "sigma",
                "seed",
                "ape_rmse",
                "rpe_rmse"
            ])

#loop
    run_id = 0
    for sigma in args.sigma:
        for seed in args.seeds:
            run_id += 1
            np.random.seed(seed)

            #1 Create simulator based on fault type
            if args.fault_type == 'gaussian':
                simulator = GaussianNoiseSimulator(sigma=sigma)
            elif args.fault_type == 'fog':
                simulator = FogSimulator(distance=args.distance, V=args.visibility)
            else:
                raise ValueError(f"Unknown fault type: {args.fault_type}")

            #2 dataset
            dataset = FogDataset(data_dir=args.data, modifier=simulator)

            #3 run odometry
            poses = run_odometry(dataset)
            est_path = out_dir / f"est_poses_sigma_{sigma:.2f}_seed_{seed}.txt"
            save_poses_kitti(est_path, poses)
            
            # Collect fog statistics
            if args.fault_type == 'fog':
                total_pts = simulator.stats['total']
                deleted_pts = simulator.stats['deleted']
                backscattered_pts = simulator.stats['backscattered']
                p_delete_val = 1 + simulator.a * np.exp(simulator.b * simulator.V)
                lambda_val = simulator.lambda_
            else:
                total_pts = 0
                deleted_pts = 0
                backscattered_pts = 0
                p_delete_val = 0
                lambda_val = 0
            
            print(f"[Run {run_id}] fault_type={args.fault_type}, sigma={sigma:.3f}, seed={seed}")
            print(f"  Saved: {est_path}")
            if args.fault_type == 'fog':
                print(f"  Fog Stats: Total={total_pts}, Deleted={deleted_pts}, Backscattered={backscattered_pts}")
                print(f"  p_delete={p_delete_val:.3f}, lambda={lambda_val:.3f}m")

            if not args.skip_metrics: #Can skip metrics (currently not working) by using argument: --skip-metrics
                # 4) Run evo metrics
                ape_stats = evo_ape_kitti(
                    gt=args.gt,
                    est=est_path,
                    out_zip=out_dir / f"ape_sigma_{sigma:.2f}_seed_{seed}.zip"
                )

                rpe_stats = evo_rpe_kitti(
                    gt=args.gt,
                    est=est_path,
                    out_zip=out_dir / f"rpe_sigma_{sigma:.2f}_seed_{seed}.zip"
                )

                ape_rmse = ape_stats["rmse"]
                rpe_rmse = rpe_stats["rmse"]

                print(f"  APE RMSE: {ape_rmse:.3f} m")
                print(f"  RPE RMSE: {rpe_rmse:.3f} m")

                #csv
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        run_id,
                        args.fault_type,
                        sigma,
                        seed,
                        ape_rmse,
                        rpe_rmse,
                        total_pts,
                        deleted_pts,
                        backscattered_pts,
                        p_delete_val,
                        lambda_val
                    ])
    if not args.skip_metrics:
        print(f"\n✓ Done. Results saved to {csv_path}")
    else:
        print(f"\n✓ Done. Poses saved to results/")
if __name__ == "__main__":
    main()