#!/usr/bin/env python3

import csv
import numpy as np
from pathlib import Path
import argparse

from kiss_icp.pipeline import OdometryPipeline
from fog_dataset import FogDataset
from fog_simulator import FogSimulator

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

    args = ap.parse_args()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

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

            #1 fog model 
            fog = FogSimulator(sigma=sigma)

            #2 dataset
            dataset = FogDataset(data_dir=args.data, modifier=fog)

            #3 run odometry
            poses = run_odometry(dataset)
            est_path = out_dir / f"est_poses_sigma_{sigma:.2f}_seed_{seed}.txt"
            save_poses_kitti(est_path, poses)
            
            print(f"[Run {run_id}] sigma={sigma:.3f}, seed={seed}")
            print(f"  Saved: {est_path}")

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
                        sigma,
                        seed,
                        ape_rmse,
                        rpe_rmse
                    ])
    if not args.skip_metrics:
        print(f"\n✓ Done. Results saved to {csv_path}")
    else:
        print(f"\n✓ Done. Poses saved to results/")
if __name__ == "__main__":
    main()