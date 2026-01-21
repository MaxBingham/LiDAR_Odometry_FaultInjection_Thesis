import numpy as np
import zipfile
import json
from subprocess import run
from pathlib import Path


def _read_stats_from_zip(out_zip: Path):
    if not out_zip.exists():
        raise RuntimeError(f"evo did not produce output zip: {out_zip}")
    with zipfile.ZipFile(out_zip, 'r') as zf:
        if 'stats.json' not in zf.namelist():
            raise RuntimeError('stats.json missing from evo output')
        with zf.open('stats.json') as f:
            data = json.load(f)
    stats = data.get('results', {}).get('stats', data)
    return stats


def evo_ape_kitti(gt, est, out_zip=None):
    """Run evo_ape (KITTI) and return stats dict. If out_zip is None a temporary
    file next to `est` will be used.
    """
    est = Path(est)
    if out_zip is None:
        out_zip = est.parent / (est.stem + '_ape.zip')
    r = run(["evo_ape", "kitti", str(gt), str(est), "-a", "--save_results", str(out_zip), "--no_warnings"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"evo_ape failed: {r.stdout}\n{r.stderr}")
    return _read_stats_from_zip(Path(out_zip))


def evo_rpe_kitti(gt, est, out_zip=None, delta_m=1.0):
    """Run evo_rpe (KITTI). Default delta is 1.0 meter (delta_unit 'm').
    If out_zip is None a temporary file next to `est` will be used.
    """
    est = Path(est)
    if out_zip is None:
        out_zip = est.parent / (est.stem + '_rpe.zip')
    r = run([
        "evo_rpe", "kitti", str(gt), str(est),
        "-a", "-r", "trans_part",
        "--delta", str(delta_m), "--delta_unit", "m",
        "--save_results", str(out_zip), "--no_warnings"
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"evo_rpe failed: {r.stdout}\n{r.stderr}")
    return _read_stats_from_zip(Path(out_zip))
