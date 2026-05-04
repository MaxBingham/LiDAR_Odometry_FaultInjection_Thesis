###Automated Runner for LFL Models###

import os
import subprocess
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional, Tuple
from lfl.fault_stats_logger import FaultStatsLogger
from lfi.apply_fault_model import save_fault_stats, reset_fault_stats, load_fault_stats


def generate_plots_after_run(
    pc_csv,
    distance_csv,
    output_dir,
    mode='comprehensive',
    save_format='png',
    dpi=300,
    pbar=None,
):
    """
    Generate plots after runner completes.
    
    Args:
        pc_csv: Path to point_cloud_metrics CSV
        distance_csv: Path to distance_metrics CSV
        output_dir: Output directory for plots
        mode: 'comprehensive', 'minimal', or 'none' (skip)
        save_format: 'png', 'pdf', or 'svg'
        dpi: Resolution
        pbar: Optional progress bar to update
    
    Returns:
        (success: bool, message: str)
    """
    if mode == 'none':
        return True, "Plot generation skipped"
    
    if not pc_csv.exists():
        return False, f"Point cloud metrics CSV not found: {pc_csv}"
    
    if not distance_csv.exists():
        return False, f"Distance metrics CSV not found: {distance_csv}"
    
    try:
        # Import here to avoid circular imports
        from lfl.plot_metrics_comprehensive import (
            generate_comprehensive_plots,
            generate_minimal_report_plots
        )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if mode == 'comprehensive':
            generate_comprehensive_plots(
                pc_csv=pc_csv,
                distance_csv=distance_csv,
                output_dir=output_dir,
                visibility_filter=None,
                sequence_filter=None,
                save_format=save_format,
                dpi=dpi
            )
            return True, f"Comprehensive plots saved to {output_dir}"
        
        elif mode == 'minimal':
            generate_minimal_report_plots(
                pc_csv=pc_csv,
                distance_csv=distance_csv,
                output_dir=output_dir,
                save_format=save_format,
                dpi=dpi
            )
            return True, f"Minimal report plots saved to {output_dir}"
        
        else:
            return False, f"Unknown plot mode: {mode}"
    
    except Exception as e:
        return False, f"Plot generation failed: {str(e)}"

def run_lfl_model(
    sequence: str,
    visualize: bool,
    fault_model: str,
    visibility: float | None,
    rain_rate: float | None,
    data_root: str,
    fog_metric: str = "distance",
    pbar=None,
):
    """Run one LFL experiment and return (ok, message, stats_dict)."""

    cmd = [
        sys.executable,
        "-m",
        "lfl.cli",
        "--sequence",
        sequence,
        "--fault_model",
        fault_model,
        "--data_root",
        data_root,
    ]

    if visualize:
        cmd.append("--visualize")

    if fault_model == "fog" and visibility is not None:
        cmd += ["--visibility", str(visibility)]
        cmd += ["--fog_metric", fog_metric]
    if fault_model == "rain" and rain_rate is not None:
        cmd += ["--rain_rate", str(rain_rate)]

    try:
        reset_fault_stats()

        if pbar:
            pbar.set_postfix_str(f"Running vis={visibility} rain={rain_rate}")

        # Inherit current environment to ensure venv and PYTHONPATH are preserved
        env = os.environ.copy()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

        if result.returncode != 0:
            return False, result.stderr.strip() or "FAILED", None, None

        # Extract JSON from stderr (last line should be JSON)
        stderr_lines = result.stderr.strip().split('\n')
        json_line = stderr_lines[-1] if stderr_lines else ""
        
        try:
            stats = json.loads(json_line)
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {str(e)}", None, None
        
        # Print the captured stdout to show progress and metrics
        print(result.stdout)
        
        # Handle fog-only mode (result_dir is None)
        result_dir = stats.get("result_dir")
        if result_dir == "null" or result_dir is None:
            result_dir = None
        
        return True, "ok", stats["stats"], result_dir

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (600s)", None, None
    except Exception as e:
        return False, f"ERROR: {str(e)[:120]}", None, None
    
    

def main():
    parser = argparse.ArgumentParser(description="Automated Runner for LFL Models")
    parser.add_argument("--fault_model", required=True, choices=["fog", "rain"], help="Fault model to run")
    parser.add_argument("--sequence", default="07", help="Single KITTI sequence (default: 07)")
    parser.add_argument("--sequences", nargs="+", default=None, help="List of sequences (overrides --sequence)")
    parser.add_argument("--data_root", default="data/kitti", help="Root directory of KITTI data")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--fog_range", nargs=3, type=int, default=[30, 251, 20], metavar=('START', 'END', 'STEP'), help="Fog visibility range: start end step (default: 40 251 200)")
    parser.add_argument("--rain_range", nargs=3, type=int, default=[10, 101, 10], metavar=('START', 'END', 'STEP'), help="Rain rate range: start end step (default: 10 101 10)")
    parser.add_argument("--fog_metric", type=str, choices=["distance", "chamfer"], default="distance", help="Fog parameterization metric (default: distance)")
    parser.add_argument("--generate_plots", choices=["comprehensive", "minimal", "none"], default="comprehensive", help="Generate plots after run completes (default: comprehensive)")
    parser.add_argument("--plot_format", choices=["png", "pdf", "svg"], default="png", help="Plot output format (default: png)")
    parser.add_argument("--plot_dpi", type=int, default=300, help="Plot resolution in DPI (default: 300)")

    args = parser.parse_args()

    fault_model = args.fault_model
    sequences = args.sequences if args.sequences else [args.sequence]
    visualize = args.visualize
    data_root = args.data_root
    fog_metric = args.fog_metric

    # Generate unique run name for this runner execution
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq_str = "_".join(sequences)
    if fault_model == "fog":
        run_name = f"{fault_model}_{fog_metric}_seq{seq_str}_{timestamp}"
    else:
        run_name = f"{fault_model}_seq{seq_str}_{timestamp}"

    # For fog-only mode, use intensity folder
    if fault_model == "fog":
        output_dir = Path("results/intensity")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = FaultStatsLogger(str(output_dir), run_name=run_name)
    else:
        logger = FaultStatsLogger("results", run_name=run_name)

    print(f"\n{'='*120}")
    print("  LIDAR FAULT LOCALIZATION RUNNER")
    print(f"  Fault Model: {fault_model.upper()}")
    print(f"  Sequences: {', '.join(sequences)}")
    if fault_model == "fog":
        print(f"  Fog Metric: {fog_metric}")
    print(f"  Run Name: {run_name}")
    print(f"  Point Cloud Metrics: {logger.get_point_cloud_file()}")
    print(f"  Localization Metrics: {logger.get_localization_file()}")
    print(f"{'='*120}\n")

    if fault_model == "fog":
        configs = list(range(args.fog_range[0], args.fog_range[1], args.fog_range[2]))
    else:
        configs = list(range(args.rain_range[0], args.rain_range[1], args.rain_range[2]))

    total_configs = len(sequences) * len(configs)

    overall_pbar = tqdm(total=total_configs, desc="Overall Progress", position=0, leave=True, ncols=120, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    stats = {"success": 0, "failed": 0, "errors": []}

    for seq_idx, sequence in enumerate(sequences, 1):
        print(f"\n{'='*120}")
        print(f"  SEQUENCE {sequence}  ({seq_idx}/{len(sequences)})")
        print(f"{'='*120}\n")

        seq_pbar = tqdm(configs, desc=f"Seq {sequence} {fault_model.upper()} Configs", position=1, leave=False, ncols=120, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]')

        for idx, level in enumerate(seq_pbar, 1):
            if fault_model == "fog":
                visibility, rain_rate = level, None
                print(f"  → Running FOG config {idx}/{len(configs)}: visibility={visibility}")
            else:
                visibility, rain_rate = None, level
                print(f"  → Running RAIN config {idx}/{len(configs)}: rain_rate={rain_rate}")

            success, msg, run_stats, result_dir = run_lfl_model(
                sequence=sequence,
                visualize=visualize,
                fault_model=fault_model,
                visibility=visibility,
                rain_rate=rain_rate,
                data_root=data_root,
                fog_metric=fog_metric,
                pbar=seq_pbar,
            )

            if success:
                print(f"    ✓ COMPLETED")
                if run_stats:
                    # For fog-only mode, result_dir is None (no localization metrics)
                    if result_dir is None:
                        # Log only point cloud stats (intensity metrics)
                        logger.log_point_cloud_stats(
                            sequence=sequence,
                            fault_model=fault_model,
                            visibility=visibility or 0,
                            rain_rate=rain_rate or 0,
                            stats=run_stats,
                        )
                    else:
                        # Normal mode: separate point cloud stats from localization metrics
                        pc_stats = {k: v for k, v in run_stats.items() if k not in ["ape", "rpe", "distance_stats"]}
                        evo_metrics = {k: v for k, v in run_stats.items() if k in ["ape", "rpe"]}
                        
                        logger.log_point_cloud_stats(
                            sequence=sequence,
                            fault_model=fault_model,
                            visibility=visibility or 0,
                            rain_rate=rain_rate or 0,
                            stats=pc_stats,
                        )
                        logger.log_localization_stats(
                            sequence=sequence,
                            fault_model=fault_model,
                            visibility=visibility or 0,
                            rain_rate=rain_rate or 0,
                            evo_metrics=evo_metrics,
                        )
                    
                    # Log distance-binned statistics for fog
                    if fault_model == "fog" and visibility is not None:
                        distance_stats = run_stats.get("distance_stats", [])
                        if distance_stats:
                            print(f"    → Distance bins logged: {len(distance_stats)} bins")
                            logger.log_distance_stats(
                                sequence=sequence,
                                fault_model=fault_model,
                                visibility=visibility,
                                rain_rate=0,
                                distance_stats=distance_stats,
                            )
                        else:
                            print(f"    ⚠ Warning: No distance stats available")
                    
                stats["success"] += 1
            else:
                print(f"    ✗ FAILED: {msg}")
                stats["failed"] += 1
                label = f"vis={visibility}" if visibility is not None else f"rain={rain_rate}"
                stats["errors"].append(f"Seq {sequence} {label}: {msg}")

            seq_pbar.update(1)
            overall_pbar.update(1)

        seq_pbar.close()

    overall_pbar.close()

    print(f"\n{'='*120}")
    print("  SUMMARY")
    print(f"  Total Configs: {total_configs}")
    print(f"  ✓ Successful: {stats['success']}")
    print(f"  ✗ Failed: {stats['failed']}")

    if stats["errors"]:
        print(f"\n  Failed Configs:")
        for error in stats["errors"]:
            print(f"    • {error}")

    print(f"{'='*120}\n")
    
    # Generate plots if requested
    if args.generate_plots != 'none':
        print(f"\n{'='*120}")
        print("  GENERATING PLOTS")
        print(f"  Mode: {args.generate_plots.upper()}")
        print(f"  Format: {args.plot_format.upper()} @ {args.plot_dpi} DPI")
        print(f"{'='*120}\n")
        
        if fault_model == "fog":
            pc_csv = Path("results/intensity") / f"point_cloud_metrics_{run_name}.csv"
            distance_csv = Path("results/intensity") / f"distance_metrics_{run_name}.csv"
            plot_output_dir = Path("results/intensity/plots") / run_name
        else:
            pc_csv = Path("results") / f"point_cloud_metrics_{run_name}.csv"
            distance_csv = Path("results") / f"distance_metrics_{run_name}.csv"
            plot_output_dir = Path("results/plots") / run_name
        
        success, msg = generate_plots_after_run(
            pc_csv=pc_csv,
            distance_csv=distance_csv,
            output_dir=plot_output_dir,
            mode=args.generate_plots,
            save_format=args.plot_format,
            dpi=args.plot_dpi
        )
        
        if success:
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
        
        print(f"{'='*120}\n")

if __name__ == "__main__":
    main()

