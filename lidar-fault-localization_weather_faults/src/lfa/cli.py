import argparse
from lfa.lfa_pipeline import run_lfa

class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, width=100, max_help_position=60)

def main():
    parser = argparse.ArgumentParser(
        prog="lfa",
        description="Tool for Analyzing Lidar Faults on Point Cloud Data",
        formatter_class=CustomHelpFormatter
    )

    parser.add_argument(
        "--sequence",
        required=True,
        help="LIDAROC sequence to process (e.g. 20m, 10m, 5m)"
    )
    parser.add_argument(
        "--oc_type",
        required=True,
        help="Type of ground truth occlusion (e.g. mudDrop, mudUniform, fog)"
    )
    parser.add_argument(
        "--oc_level",
        required=True,
        help="Level of occlusion (e.g. low, mid, high)"
    )
    parser.add_argument(
        "--fault_model",
        required=True,
        help="Fault model to apply (e.g., fog, rain)"
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=["distribution", "hausdorff", "chamfer", "sdasn"],
        metavar="METRIC",
        default=[],
        help="[Optional] Metrics to compute (e.g., distribution, chamfer). Default: None"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="[Optional] Whether to visualize the results"
    )

    args = parser.parse_args()
    run_lfa(args)

if __name__ == "__main__":
    main()