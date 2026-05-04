import numpy as np


class FogSimulator:
    """
    Physically correct fog model based on Teufel et al.:
      1) Per-point modification probability: p_modify(d) = 1 - exp(-d * epsilon)
      2) If modified: either delete with p_delete(V) or backscatter with Exp(lambda)
      3) Intensity attenuation via Beer-Lambert law for unmodified points
      4) Backscattered points receive weak intensity (0-32% normal distribution)
    No geometry modification except for backscattered points (false returns).
    """

    def __init__(self, V, metric="distance", epsilon=None, lambda_=None, a=None, b=None, d_min=0.63, seed=None):
        self.V = float(V)  # visibility [m]
        self.gamma = -np.log(0.05) / self.V  # extinction coefficient proxy (if needed later)
        self.metric = metric.lower()

        # Teufel et al. parameterizations from paper Section VI-A3
        if self.metric == "distance":
            # Distance-metric optimization (better fit to real fog data per paper)
            # Paper: ε = 0.32 · e^(-0.0220·V), λ = -0.00846·V + 2.29, (a,b) = (-0.63, -0.020)
            self.epsilon = 0.32 * np.exp(-0.0220 * self.V) if epsilon is None else float(epsilon)
            self.lambda_ = (-0.00846 * self.V + 2.29) if lambda_ is None else float(lambda_)
            self.a = -0.63 if a is None else float(a)
            self.b = -0.020 if b is None else float(b)
        elif self.metric == "chamfer":
            # Chamfer-metric optimization
            # Paper: ε = 0.23 · e^(-0.0082·V), λ = -0.00600·V + 2.31, (a,b) = (-0.70, -0.024)
            self.epsilon = 0.23 * np.exp(-0.0082 * self.V) if epsilon is None else float(epsilon)
            self.lambda_ = (-0.00600 * self.V + 2.31) if lambda_ is None else float(lambda_)
            self.a = -0.70 if a is None else float(a)
            self.b = -0.024 if b is None else float(b)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'distance' or 'chamfer'.")

        # Small range-noise proxy (radial-only); keep conservative
        #####HOW DID WE CHOOSE THIS######
        self.a_turb = 1.5e-2
        self.b_turb = 0.0015
        self.c_turb = 0.6e-2

        # Minimum measurable distance (sensor blind zone)
        self.d_min = float(d_min)
        self.d_max = 0.02  # max range for bias calculation
        # RNG (optional for reproducibility)
        self.rng = np.random.default_rng(seed)

        # Distance bins for detailed analysis (0-10m, 10-20m, ..., 90-100m)
        self.distance_bins = np.arange(0, 101, 10)  # [0, 10, 20, ..., 100]

        self.stats = {
            "total": 0,
            "modified": 0,
            "deleted": 0,
            "backscattered": 0,
            "intensity_atten_mean": 0.0,
            "intensity_atten_median": 0.0,
            "intensity_atten_std": 0.0,
        }

        # Distance-binned statistics: dict[bin_idx] -> {total, modified, deleted, backscattered}
        self.distance_stats = {}

    def apply_noise_fog(self, points: np.ndarray) -> np.ndarray:
        """
        Apply physically correct fog model with intensity attenuation.

        Args:
            points: (N, 4) array [x, y, z, intensity]

        Returns:
            (M, 4) array [x, y, z, intensity_attenuated] where M <= N
        """
        # Reset stats for this frame
        self.stats = {
            "total": 0,
            "modified": 0,
            "deleted": 0,
            "backscattered": 0,
            "intensity_atten_mean": 0.0,
            "intensity_atten_median": 0.0,
            "intensity_atten_std": 0.0,
        }
        # Don't reset distance_stats - accumulate across all frames

        # Validate input
        if points is None:
            return points

        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points.reshape((0, 4)) if points.ndim == 1 else points.reshape((0, 4))

        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("points must be an (N, 4) array [x, y, z, intensity]")

        N = points.shape[0]
        self.stats["total"] = int(N)

        # Work on a copy
        result = points.copy()

        # Extract geometry and intensity
        xyz = result[:, :3]
        intensity_orig = result[:, 3].copy()

        # --- Geometry
        dist = np.linalg.norm(xyz, axis=1)
        dist_safe = np.maximum(dist, 1e-6)
        dirs = xyz / dist_safe[:, None]  # unit beam directions

        # --- 1) Choose points to modify (distance-dependent)
        # Paper: p_modify(d) = 1 - exp(-d·ε)
        p_modify = 1.0 - np.exp(-dist * self.epsilon)
        p_modify = np.clip(p_modify, 0.0, 1.0)

        modify_mask = self.rng.random(N) < p_modify
        self.stats["modified"] = int(np.sum(modify_mask))

        # --- 2) If modified: delete with p_delete(V), else backscatter (false return)
        # Paper: p_delete = a · e^(b·V) + 1
        p_delete = self.a * np.exp(self.b * self.V) + 1.0
        p_delete = float(np.clip(p_delete, 0.0, 1.0))

        delete_mask = modify_mask & (self.rng.random(N) < p_delete)
        self.stats["deleted"] = int(np.sum(delete_mask))

        backscatter_mask = modify_mask & ~delete_mask
        n_back = int(np.sum(backscatter_mask))
        self.stats["backscattered"] = n_back

        # --- Track distance-binned statistics ---
        bin_indices = np.digitize(dist, self.distance_bins) - 1

        for bin_idx in range(len(self.distance_bins) - 1):
            in_bin = (bin_indices == bin_idx)
            n_in_bin = int(np.sum(in_bin))

            if n_in_bin > 0:
                if bin_idx not in self.distance_stats:
                    self.distance_stats[bin_idx] = {
                        "total": 0,
                        "modified": 0,
                        "deleted": 0,
                        "backscattered": 0,
                    }

                self.distance_stats[bin_idx]["total"] += n_in_bin
                self.distance_stats[bin_idx]["modified"] += int(np.sum(modify_mask & in_bin))
                self.distance_stats[bin_idx]["deleted"] += int(np.sum(delete_mask & in_bin))
                self.distance_stats[bin_idx]["backscattered"] += int(np.sum(backscatter_mask & in_bin))

        # --- 3) Backscattering: geometry + weak intensity
        if n_back > 0:
            # Geometry: exponential sampling for new distance
            x = self.rng.exponential(scale=self.lambda_, size=n_back)
            dist_back = dist[backscatter_mask]
            d_new = np.minimum(self.d_min + x, dist_back)

            # Update xyz for backscattered points
            result[backscatter_mask, :3] = dirs[backscatter_mask] * d_new[:, None]

            # Intensity: sample from N(16%, std=5.33%) clipped to [0%, 32%]
            back_intensity_ratio = self.rng.normal(0.16, 0.16/3.0, size=n_back)
            back_intensity_ratio = np.clip(back_intensity_ratio, 0.0, 0.32)
            result[backscatter_mask, 3] = intensity_orig[backscatter_mask] * back_intensity_ratio

        # --- 4) Unmodified points: Beer-Lambert intensity attenuation only
        # These are true surface returns, attenuated by fog along two-way path
        surface_mask = ~modify_mask
        if np.any(surface_mask):
            gamma = -np.log(0.05) / self.V
            attenuation = np.exp(-gamma * 2.0 * dist[surface_mask])
            result[surface_mask, 3] = intensity_orig[surface_mask] * attenuation

        # --- 5) Compute intensity statistics for non-deleted points
        non_deleted_mask = ~delete_mask
        if np.sum(non_deleted_mask) > 0:
            intensity_ratio = result[non_deleted_mask, 3] / np.maximum(intensity_orig[non_deleted_mask], 1e-9)
            self.stats["intensity_atten_mean"] = float(np.mean(intensity_ratio))
            self.stats["intensity_atten_median"] = float(np.median(intensity_ratio))
            self.stats["intensity_atten_std"] = float(np.std(intensity_ratio))

        # --- 6) Remove deleted points
        out = result[~delete_mask].copy()
        return out

    def get_statistics(self):
        total = int(self.stats.get("total", 0))
        if total == 0:
            return {}
        return {
            "total": total,
            "modified": int(self.stats.get("modified", 0)),
            "deleted": int(self.stats.get("deleted", 0)),
            "backscattered": int(self.stats.get("backscattered", 0)),
            "delete_rate": float(self.stats.get("deleted", 0)) / total,
            "backscatter_rate": float(self.stats.get("backscattered", 0)) / total,
            "intensity_atten_mean": float(self.stats.get("intensity_atten_mean", 0.0)),
            "intensity_atten_median": float(self.stats.get("intensity_atten_median", 0.0)),
            "intensity_atten_std": float(self.stats.get("intensity_atten_std", 0.0)),
        }

    def get_distance_statistics(self):
        """Return distance-binned statistics as list of dicts."""
        results = []
        for bin_idx in sorted(self.distance_stats.keys()):
            stats = self.distance_stats[bin_idx]
            total = stats["total"]
            if total > 0:
                results.append({
                    "bin_idx": int(bin_idx),
                    "distance_min": float(self.distance_bins[bin_idx]),
                    "distance_max": float(self.distance_bins[bin_idx + 1]),
                    "distance_center": float((self.distance_bins[bin_idx] + self.distance_bins[bin_idx + 1]) / 2),
                    "total": int(total),
                    "modified": int(stats["modified"]),
                    "deleted": int(stats["deleted"]),
                    "backscattered": int(stats["backscattered"]),
                    "modified_percent": float(100.0 * stats["modified"] / total),
                    "deleted_percent": float(100.0 * stats["deleted"] / total),
                    "backscattered_percent": float(100.0 * stats["backscattered"] / total),
                })
        return results
