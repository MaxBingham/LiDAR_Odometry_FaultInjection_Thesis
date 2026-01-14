#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import os

DATA = "data/07/velodyne"
GT = "data/07.txt"
OUT = "results/all_fog_experiments.csv"

configs = sorted(Path('configs').glob('fog_*.yaml'))
if not configs:
    print("No fog_*.yaml files found")
    sys.exit(1)

print(f"\n{'='*60}\nBATCH EVAL - {len(configs)} configs\n{'='*60}\n")

for i, cfg in enumerate(configs, 1):
    print(f"[{i}/{len(configs)}] {cfg.name}")
    
    # Clear previous run results to ensure clean evaluation
    result_dir = Path("results")
    for file in result_dir.glob("est_poses_*.txt"):
        file.unlink()
    for file in result_dir.glob("*.zip"):
        file.unlink()
    
    try:
        subprocess.run([
            'python3', 'src/evaluate.py',
            '--data', DATA,
            '--gt', GT,
            '--sigma', '0.0',
            '--fault-type', 'fog',
            '--visibility', str(cfg),  # Parse from config file
            '--output', OUT
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"✗ Failed")
        sys.exit(1)

print(f"\n{'='*60}\n✓ ALL DONE → {OUT}\n{'='*60}\n")
