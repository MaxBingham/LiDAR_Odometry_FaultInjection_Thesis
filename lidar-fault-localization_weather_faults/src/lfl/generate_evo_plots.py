"""Generate EVO trajectory plots for selected visibility values by re-running pipeline."""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def main():
    # Run for visibilities 50, 150, 250 across all sequences
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    visibilities = [50, 150, 250]
    
    print(f"\n{'='*80}")
    print("  EVO TRAJECTORY PLOT GENERATOR")
    print(f"  Generating plots for visibility: {', '.join(map(str, visibilities))}m")
    print(f"  Sequences: {', '.join(sequences)}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "-m", "lfl.runner",
        "--fault_model", "fog",
        "--sequences"] + sequences + [
        "--fog_range", "50", "251", "100",  # 50, 150, 250
        "--visualize"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    subprocess.run(cmd)
    
    print(f"\n{'='*80}")
    print("  EVO plots have been generated and saved in:")
    print("  results/<timestamp>/evo/ape_xz.png")
    print("  results/<timestamp>/evo/rpe_xz.png")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
