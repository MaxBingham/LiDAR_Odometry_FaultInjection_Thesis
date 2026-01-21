import numpy as np


class FogSimulator:
    """
    Probabilistic fog model inspired by Teufel et al. (Exercise 1):
      1) Per-point modification probability: p_modify(d) = 1 - exp(-d * epsilon)
      2) If modified: either delete with p_delete(V) or move (backscatter) with Exp(lambda)
      3) For remaining surface returns: add radial ranging error (distance-domain noise)
    Intensity is treated as latent and absorbed into deletion/backscatter/noise mechanisms.
    """

    def __init__(self, V, a=-0.7, b=-0.024, epsilon=None, lambda_=None, d_min=0.63, seed=None):
        self.V = float(V)  # visibility [m]
        self.gamma = -np.log(0.05) / self.V  # extinction coefficient proxy (if needed later)

        # Teufel et al. (chamfer-fit) parameterizations (overrideable)
        self.epsilon = 0.23 * np.exp(-0.0082 * self.V) if epsilon is None else float(epsilon)
        self.lambda_ = (-0.00600 * self.V + 2.31) if lambda_ is None else float(lambda_)

        self.a = float(a)
        self.b = float(b)

        # Small range-noise proxy (radial-only); keep conservative
        self.a_turb = 1.5e-2
        self.b_turb = 0.0015
        self.c_turb = 0.6e-2

        # Minimum measurable distance (sensor blind zone)
        self.d_min = float(d_min)

        # RNG (optional for reproducibility)
        self.rng = np.random.default_rng(seed)

        self.stats = {
            "total": 0,
            "modified": 0,
            "deleted": 0,
            "backscattered": 0,
            "error_added": 0,
        }

    def apply_noise(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return points

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an (N,3) array of xyz coordinates")

        N = points.shape[0]
        self.stats["total"] += int(N)

        # --- Geometry
        dist = np.linalg.norm(points, axis=1)
        dist_safe = np.maximum(dist, 1e-6)
        dirs = points / dist_safe[:, None]  # unit beam directions

        # --- 1) Choose points to modify (distance-dependent)
        p_modify = 1.0 - np.exp(-dist * self.epsilon)
        # numeric safety
        p_modify = np.clip(p_modify, 0.0, 1.0)

        modify_mask = self.rng.random(N) < p_modify
        self.stats["modified"] += int(np.sum(modify_mask))

        # --- 2) If modified: delete with p_delete(V), else backscatter (false return)
        p_delete = 1.0 + self.a * np.exp(self.b * self.V)
        p_delete = float(np.clip(p_delete, 0.0, 1.0))

        delete_mask = modify_mask & (self.rng.random(N) < p_delete)
        self.stats["deleted"] += int(np.sum(delete_mask))

        backscatter_mask = modify_mask & ~delete_mask
        n_back = int(np.sum(backscatter_mask))

        if n_back > 0:
            # Teufel: new distance drawn from exponential pdf, then shifted by d_min
            x = self.rng.exponential(scale=self.lambda_, size=n_back)
            d_new = self.d_min + x

            # Place false returns along same beam direction
            points[backscatter_mask] = dirs[backscatter_mask] * d_new[:, None]
            self.stats["backscattered"] += n_back

        # --- 3) Add ranging error (radial-only) to remaining SURFACE returns
        # Surface returns are those neither deleted nor replaced by backscatter
        surface_mask = ~delete_mask & ~backscatter_mask
        n_surface = int(np.sum(surface_mask))

        if n_surface > 0:
            # Visibility-dependent base error (proxy fit)
            p_error = self.a_turb * np.exp(-self.b_turb * self.V) + self.c_turb

            # Distance scaling: saturating exponential (more plausible than linear)
            dist_surface = dist[surface_mask]
            scale = 1.0 - np.exp(-dist_surface / self.V)

            d_error = p_error * scale

            # Optional: cap (keep if you want physical bounds; enable for safety)
            d_error = np.clip(d_error, 0.0, 0.02)

            # Sample scalar range error and apply along beam direction
            dr = self.rng.normal(loc=0.0, scale=d_error, size=n_surface)
            points[surface_mask] = points[surface_mask] + dirs[surface_mask] * dr[:, None]

            self.stats["error_added"] += n_surface

        # --- 4) Apply deletion once at the end
        out = points[~delete_mask].copy()
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
            "error_added": int(self.stats.get("error_added", 0)),
            "delete_rate": float(self.stats.get("deleted", 0)) / total,
            "backscatter_rate": float(self.stats.get("backscattered", 0)) / total,
        }