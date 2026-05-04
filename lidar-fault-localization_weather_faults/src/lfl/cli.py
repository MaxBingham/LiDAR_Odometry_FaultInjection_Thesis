import argparse
import json
import sys
from lfl.lfl_pipeline import run_lfl

class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, width=100, max_help_position=60)

def main():
    parser = argparse.ArgumentParser(
        prog="lfl_pipeline",
        description="Pipeline for Lidar Fault Localization",
        formatter_class=CustomHelpFormatter
    )

    parser.add_argument(
        "--sequence",
        required=True,
        help="KITTI sequence to process (e.g., 00, 01, ... 21)"
    )
    parser.add_argument(
        "--data_root",
        default="data/kitti",
        help="Root directory of KITTI data (default: data/kitti)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="[Optional] Whether to visualize the results"
    )
    parser.add_argument(
        "--fault_model",
        choices=["fog", "rain", "none"],
        default="none",
        help="Fault model to apply"
    )
    parser.add_argument(
        "--visibility",
        type=float,
        default=None,
        help="Visibility in meters (required for fog)"
    )
    parser.add_argument(
        "--rain_rate",
        type=float,
        default=None,
        help="Rain rate in mm/h (required for rain)"
    )
    parser.add_argument(
        "--fog_metric",
        type=str,
        choices=["distance", "chamfer"],
        default="distance",
        help="Fog parameterization metric (default: distance)"
    )

    args = parser.parse_args()

    # Validate required parameters
    if args.fault_model == "fog" and args.visibility is None:
        parser.error("--visibility must be set when fault_model=fog")
    if args.fault_model == "rain" and args.rain_rate is None:
        parser.error("--rain_rate must be set when fault_model=rain")

    stats, result_dir = run_lfl(args)
    print(json.dumps({"stats": stats, "result_dir": str(result_dir)}), file=sys.stderr)

if __name__ == "__main__":
    main()
