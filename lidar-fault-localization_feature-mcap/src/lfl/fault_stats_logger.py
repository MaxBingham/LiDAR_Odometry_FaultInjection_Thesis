# fault_stats_logger.py
import csv
from pathlib import Path
from datetime import datetime
import json
import subprocess
import numpy as np

class FaultStatsLogger:
    """Logger for fault injection stats with separate CSVs for point cloud and localization metrics."""

    def __init__(self, results_dir: str = "results", run_name: str = None):
        """
        Initialize logger with optional run-specific naming.
        
        Args:
            results_dir: Base directory for results
            run_name: Optional unique identifier for this run (e.g., "fog_chamfer_seq03_20260131_143022")
                     If None, appends to default CSV files
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.run_name = run_name

        # Generate filenames based on run_name
        if run_name:
            self.pc_file = self.results_dir / f"point_cloud_metrics_{run_name}.csv"
            self.loc_file = self.results_dir / f"localization_metrics_{run_name}.csv"
            self.dist_file = self.results_dir / f"distance_metrics_{run_name}.csv"
        else:
            self.pc_file = self.results_dir / "point_cloud_metrics.csv"
            self.loc_file = self.results_dir / "localization_metrics.csv"
            self.dist_file = self.results_dir / "distance_metrics.csv"

        self.pc_fields = [
            "timestamp",
            "sequence",
            "fault_model",
            "visibility",
            "rain_rate",
            "total_frames",
            "total_points",
            "deleted_points",
            "deleted_percent",
            "backscattered_points",
            "backscattered_percent",
            "modified_points",
            "modified_percent",
            "intensity_atten_mean",
            "intensity_atten_median",
            "intensity_atten_std",
        ]

        # Localization metrics CSV
        self.loc_fields = [
            "timestamp",
            "sequence",
            "fault_model",
            "visibility",
            "rain_rate",
            "ape_rmse",
            "ape_mean",
            "ape_median",
            "ape_std",
            "rpe_rmse",
            "rpe_mean",
            "rpe_median",
            "rpe_std",
        ]
        
        # Distance-binned metrics CSV
        self.dist_fields = [
            "timestamp",
            "sequence",
            "fault_model",
            "visibility",
            "rain_rate",
            "distance_min",
            "distance_max",
            "distance_center",
            "total_points",
            "modified_points",
            "modified_percent",
            "deleted_points",
            "deleted_percent",
            "backscattered_points",
            "backscattered_percent",
        ]

        # Initialize CSVs if they don't exist
        if not self.pc_file.exists():
            with open(self.pc_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.pc_fields).writeheader()

        if not self.loc_file.exists():
            with open(self.loc_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.loc_fields).writeheader()
        
        if not self.dist_file.exists():
            with open(self.dist_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.dist_fields).writeheader()

    def _get_base_row(self, sequence: str, fault_model: str, visibility: float, rain_rate: float) -> dict:
        """Get common fields for both CSV types."""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sequence": sequence,
            "fault_model": fault_model,
            "visibility": visibility or 0,
            "rain_rate": rain_rate or 0,
        }

    def log_point_cloud_stats(
        self,
        sequence: str,
        fault_model: str,
        visibility: float,
        rain_rate: float,
        stats: dict,
    ):
        """Write point cloud stats row."""
        total_points = max(stats.get("total_points", 0), 1)  # avoid div/0

        row = self._get_base_row(sequence, fault_model, visibility, rain_rate)
        row.update({
            "total_frames": stats.get("total_frames", 0),
            "total_points": stats.get("total_points", 0),
            "deleted_points": stats.get("total_deleted", 0),
            "deleted_percent": f"{100 * stats.get('total_deleted', 0) / total_points:.2f}",
            "backscattered_points": stats.get("total_backscattered", 0),
            "backscattered_percent": f"{100 * stats.get('total_backscattered', 0) / total_points:.2f}",
            "modified_points": stats.get("total_modified", 0),
            "modified_percent": f"{100 * stats.get('total_modified', 0) / total_points:.2f}",
            "intensity_atten_mean": f"{stats.get('intensity_atten_mean', 0.0):.4f}",
            "intensity_atten_median": f"{stats.get('intensity_atten_median', 0.0):.4f}",
            "intensity_atten_std": f"{stats.get('intensity_atten_std', 0.0):.4f}",
        })

        with open(self.pc_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.pc_fields).writerow(row)

    def log_localization_stats(
        self,
        sequence: str,
        fault_model: str,
        visibility: float,
        rain_rate: float,
        evo_metrics: dict,
    ):
        """Log localization metrics from EVO results."""
        try:
            ape_stats = evo_metrics.get("ape", {})
            rpe_stats = evo_metrics.get("rpe", {})

            row = self._get_base_row(sequence, fault_model, visibility, rain_rate)
            row.update({
                "ape_rmse": ape_stats.get("rmse", 0),
                "ape_mean": ape_stats.get("mean", 0),
                "ape_median": ape_stats.get("median", 0),
                "ape_std": ape_stats.get("std", 0),
                "rpe_rmse": rpe_stats.get("rmse", 0),
                "rpe_mean": rpe_stats.get("mean", 0),
                "rpe_median": rpe_stats.get("median", 0),
                "rpe_std": rpe_stats.get("std", 0),
            })

            with open(self.loc_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.loc_fields).writerow(row)

        except Exception as e:
            print(f"Warning: Could not log localization metrics: {e}")
    
    def log_distance_stats(
        self,
        sequence: str,
        fault_model: str,
        visibility: float,
        rain_rate: float,
        distance_stats: list,
    ):
        """Log distance-binned statistics.
        
        Args:
            distance_stats: List of dicts with keys: distance_min, distance_max, distance_center,
                           total, modified, deleted, backscattered, *_percent
        """
        if not distance_stats:
            return
        
        base_row = self._get_base_row(sequence, fault_model, visibility, rain_rate)
        
        for stat in distance_stats:
            row = base_row.copy()
            row.update({
                "distance_min": stat["distance_min"],
                "distance_max": stat["distance_max"],
                "distance_center": stat["distance_center"],
                "total_points": stat["total"],
                "modified_points": stat["modified"],
                "modified_percent": f"{stat['modified_percent']:.2f}",
                "deleted_points": stat["deleted"],
                "deleted_percent": f"{stat['deleted_percent']:.2f}",
                "backscattered_points": stat["backscattered"],
                "backscattered_percent": f"{stat['backscattered_percent']:.2f}",
            })
            
            with open(self.dist_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.dist_fields).writerow(row)

    def get_point_cloud_file(self) -> Path:
        return self.pc_file

    def get_localization_file(self) -> Path:
        return self.loc_file
    
    def get_distance_file(self) -> Path:
        return self.dist_file
