
import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def _should_force_agg() -> bool:
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    has_qt = any(
        find_spec(pkg) is not None
        for pkg in ("PyQt5", "PyQt6", "PySide2", "PySide6")
    )
    return not has_display or not has_qt


def _configure_evo_backend(env: dict, out_dir: Path) -> dict:
    if not _should_force_agg():
        return env
    env["MPLBACKEND"] = "Agg"
    evo_home = out_dir / ".evo_home"
    settings_path = evo_home / ".evo" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if not settings_path.exists():
        from evo.tools.settings_template import DEFAULT_SETTINGS_DICT

        settings = dict(DEFAULT_SETTINGS_DICT)
        settings["plot_backend"] = "Agg"
        settings_path.write_text(json.dumps(settings, indent=4, sort_keys=True))
    env["HOME"] = str(evo_home)
    return env


def _parse_evo_output(output: str) -> dict:
    """Parse EVO stdout for metrics."""
    metrics = {}
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # EVO outputs lines like: "      rmse      0.824036"
        parts = line.split()
        if len(parts) >= 2:
            metric_name = parts[0].lower()
            try:
                value = float(parts[-1])
                metrics[metric_name] = value
            except ValueError:
                continue
    return metrics

def run_kiss_icp(sequence, data_root, visualize, fault_model, visibility, rain_rate, fog_metric="distance"):
    import sys
    
    # Use the venv Python executable to ensure we use the correct environment
    venv_kiss_icp = Path(sys.executable).parent / "kiss_icp_pipeline"
    
    cmd = [
        str(venv_kiss_icp),
        "--dataloader", "kitti",
        "--sequence", sequence,
    ]

    if visualize:
        cmd.append("--visualize")

    fault_model = fault_model.lower()
    if fault_model != "none":
        cmd += ["--fault_model", fault_model]
        if fault_model == "fog":
            cmd += ["--visibility", str(visibility)]
            cmd += ["--fog_metric", fog_metric]
        elif fault_model == "rain":
            cmd += ["--rain_rate", str(rain_rate)]

    cmd.append(f"{Path(__file__).parent.parent.parent / data_root}")

    env = os.environ.copy()
    env['kiss_icp_out_dir'] = str(Path(__file__).parent.parent.parent / "results")

    # Silence verbose pipeline output to avoid filling parent buffers; keep stderr for debugging.
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent.parent / "external" / "kiss-icp" / "config",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"\n{'='*80}")
        print(f"ERROR: kiss_icp_pipeline failed with exit code {result.returncode}")
        print(f"Command: {' '.join(cmd)}")
        if result.stdout:
            print("\nSTDOUT output:")
            print(result.stdout)
        if result.stderr:
            print("\nSTDERR output:")
            print(result.stderr)
        print(f"{'='*80}\n")
        raise subprocess.CalledProcessError(result.returncode, cmd)

    # latest → actual folder produced by kiss-icp
    results_root = Path(__file__).parent.parent.parent / "results"
    latest = results_root / "latest"
    return latest.resolve()



def run_evo(sequence, result_dir, visualize):
    gt = f"{sequence}_gt_kitti.txt"
    est = f"{sequence}_poses_kitti.txt"

    out_dir = result_dir / "evo"
    out_dir.mkdir(exist_ok=True)

    ape_cmd = [
        "evo_ape", "kitti",
        gt, est,
        "--save_results", str(out_dir / "ape.zip"),
    ]

    rpe_cmd = [
        "evo_rpe", "kitti",
        gt, est,
        "--save_results", str(out_dir / "rpe.zip"),
    ]

    if visualize:
        ape_cmd += [
            "--plot",
            "--plot_mode", "xz",
            "--save_plot", str(out_dir / "ape_xz.png"),
        ]
        rpe_cmd += [
            "--plot",
            "--plot_mode", "xz",
            "--save_plot", str(out_dir / "rpe_xz.png"),
        ]

    env = _configure_evo_backend(os.environ.copy(), out_dir)

    # Run APE
    ape_result = subprocess.run(ape_cmd, cwd=result_dir, check=True, env=env, capture_output=True, text=True)
    rpe_result = subprocess.run(rpe_cmd, cwd=result_dir, check=True, env=env, capture_output=True, text=True)

    # Parse metrics from stdout
    ape_metrics = _parse_evo_output(ape_result.stdout)
    rpe_metrics = _parse_evo_output(rpe_result.stdout)

    return {"ape": ape_metrics, "rpe": rpe_metrics}




def run_lfl(args):
    """Run one experiment and return stats dict."""

    from lfi.apply_fault_model import reset_fault_stats
    reset_fault_stats()

    result_dir = run_kiss_icp(
        sequence=args.sequence,
        data_root=args.data_root,
        visualize=args.visualize,
        fault_model=args.fault_model,
        visibility=args.visibility,
        rain_rate=args.rain_rate,
        fog_metric=getattr(args, 'fog_metric', 'distance'),
    )

    evo_metrics = run_evo(
        sequence=args.sequence,
        result_dir=result_dir,
        visualize=args.visualize,
    )

    # Collect stats produced by fault model (written to temp JSON)
    from lfi.apply_fault_model import load_fault_stats, load_distance_stats

    stats = load_fault_stats()
    distance_stats = load_distance_stats() if args.fault_model == "fog" else []
    
    return {
        "sequence": args.sequence,
        "fault_model": args.fault_model,
        "visibility": args.visibility,
        "rain_rate": args.rain_rate,
        "distance_stats": distance_stats,
        **stats,
        **evo_metrics,
    }, result_dir
