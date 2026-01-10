import numpy as np
import subprocess 
import evo
import zipfile
import json
from subprocess import run


def evo_ape_kitti(gt, est, out_zip):
    r = run(["evo_ape", "kitti", str(gt), str(est), "-a", "--save_results", str(out_zip), "--no_warnings"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError("evo_ape failed")
    # Read stats.json from the results zip
    with zipfile.ZipFile(out_zip, 'r') as zf:
        with zf.open('stats.json') as f:
            data = json.load(f)
    stats = data.get("results", {}).get("stats", data)
    return stats

def evo_rpe_kitti(gt, est, out_zip, delta_m=1):
    r = run([
        "evo_rpe", "kitti", str(gt), str(est),
        "-a", "-r", "trans_part",
        "--delta", str(delta_m), "--delta_unit", "f",
        "--save_results", str(out_zip), "--no_warnings"
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError("evo_rpe failed")
    # Read stats.json from the results zip
    with zipfile.ZipFile(out_zip, 'r') as zf:
        with zf.open('stats.json') as f:
            data = json.load(f)
    stats = data.get("results", {}).get("stats", data)
    return stats
