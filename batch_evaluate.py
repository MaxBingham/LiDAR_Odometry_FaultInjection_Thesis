#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

DATA = "data/velodyne"
GT = "data/poses/07.txt"
OUT = "results/all_fog_experiments.csv"
CACHE = "results/cache/original_poses.npy"

configs = sorted(Path('configs').glob('fog_*.yaml'))
if not configs:
    print("No fog_*.yaml files found")
    sys.exit(1)

print(f"\n{'='*60}\nBATCH EVAL - {len(configs)} configs\n{'='*60}\n")

for i, cfg in enumerate(configs, 1):
    print(f"[{i}/{len(configs)}] {cfg.name}")
    try:
        subprocess.run([
            'python3', 'evaluate.py', DATA,
            '--config', str(cfg), '--gt', GT,
            '--output', OUT, '--cache', CACHE
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"✗ Failed")
        sys.exit(1)

print(f"\n{'='*60}\n✓ ALL DONE → {OUT}\n{'='*60}\n")
