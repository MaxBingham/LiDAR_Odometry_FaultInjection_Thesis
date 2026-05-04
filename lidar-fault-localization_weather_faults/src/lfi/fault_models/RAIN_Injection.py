import numpy as np


class RainSimulator:
    """
    Rain model based on fog injector (Teufel et al.) but scaled for rain physics:
    - Lower extinction coefficient (less deletion) due to sparser droplets.
    - Higher backscattering due to Mie scattering from larger droplets.
    - Increased geometric distortion (ranging noise up to ~5 cm) to highlight importance.
    - Parameters tuned based on rain rate R [mm/h], following empirical comparisons.

    Key differences from fog:
    - Fog: High deletion (~25-30%), low ranging error (<2 cm).
    - Rain: Moderate deletion (~10-20%), high ranging error (up to 5 cm), higher backscattering.
    """

    def __init__(self, R, d_min=0.63, seed=None):
        self.R = float(R)  # rain rate [mm/h]

        # -------------------------------------------------
        # 1) Modification probability (weaker in rain)
        # Fog epsilon ~ O(0.05–0.2)
        # Rain epsilon must be an order of magnitude lower
        # Sublinear scaling with rain rate
        self.epsilon_rain = 0.002 + 0.008 * (self.R / 100.0) ** 0.7
        self.epsilon_rain = np.clip(self.epsilon_rain, 0.002, 0.01)

        # -------------------------------------------------
        # 2) Backscatter distance (SHORTER than fog)
        # Rain backscatter is near-field dominated
        # Keep exponential model, but with small scale
        self.lambda_rain = 0.5 + 0.3 * (self.R / 100.0) ** 0.5
        self.lambda_rain = np.clip(self.lambda_rain, 0.5, 0.8)

        # -------------------------------------------------
        # 3) Deletion probability (SECONDARY effect)
        # Target: ~15–20% deletion at ~100 mm/h
        # Sublinear increase, hard cap
        self.a = -0.35
        self.b = -0.012

        # -------------------------------------------------
        # 4) Geometric distortion (DOMINANT rain effect)
        # Literature: up to ~4–5 cm ranging error in heavy rain
        # Stronger than fog, but still bounded
        sigma_max = 0.05 * (self.R / 100.0) ** 0.7
        sigma_max = np.clip(sigma_max, 0.0, 0.05)

        # Split into base + distance-scaled component
        self.c_turb = 0.005          # ~5 mm baseline
        self.a_turb = sigma_max      # distance-amplified part
        self.b_turb = 0.0            # rain ~ visibility-independent

        # -------------------------------------------------
        # Minimum measurable distance
        self.d_min = float(d_min)

        # RNG
        self.rng = np.random.default_rng(seed)

        # Stats
        self.stats = {
            "total": 0,
            "modified": 0,
            "deleted": 0,
            "backscattered": 0,
            "error_added": 0,
        }


    def apply_noise_rain(self, points: np.ndarray) -> np.ndarray:
        # Step 1: Validate input - normalize to numpy and avoid len() on scalars
        if points is None:
            return points

        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points.reshape((0, 3)) if points.ndim == 1 else points.reshape((0, 3))

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an (N,3) array of xyz coordinates")

        N = points.shape[0]
        self.stats["total"] += N

        # Step 2: Compute distances and directions for each point
        # This is needed for distance-dependent effects like modification probability
        dist = np.linalg.norm(points, axis=1)
        dist_safe = np.maximum(dist, 1e-6)
        dirs = points / dist_safe[:, None]

        # Step 3: Determine which points are modified (affected by rain)
        # Rain has lower modification probability than fog due to sparser droplets
        p_mod = 1.0 - np.exp(-dist * self.epsilon_rain)
        mod_mask = self.rng.random(N) < p_mod
        self.stats["modified"] += int(np.sum(mod_mask))

        # Step 4: Among modified points, decide deletions (extinction effect)
        # Rain has lower deletion rate than fog (less attenuation)
        p_delete = 1 + self.a * np.exp(self.b * self.R)
        p_delete = np.clip(p_delete, 0.0, 0.3)  # Cap at 30% for realism
        del_mask = mod_mask & (self.rng.random(N) < p_delete)
        self.stats["deleted"] += int(np.sum(del_mask))

        # Step 5: Create backscatters for modified but not deleted points
        # Rain has higher backscattering due to Mie scattering from larger droplets
        back_mask = mod_mask & ~del_mask & (self.rng.random(N) < 0.6)  # Higher prob than fog
        n_back = int(np.sum(back_mask))
        if n_back > 0:
            # Sample distances from exponential (similar to fog, but tuned)
            d_back = self.rng.exponential(1.0 / self.lambda_rain, size=n_back)
            d_back = np.maximum(d_back, self.d_min)
            points[back_mask] = dirs[back_mask] * d_back[:, None]
            self.stats["backscattered"] += n_back

        # Step 6: Apply geometric distortion to remaining surface returns
        # This is the key difference: rain causes significant ranging errors (up to ~5 cm)
        surf_mask = ~del_mask & ~back_mask
        n_surf = int(np.sum(surf_mask))
        if n_surf > 0:
            # Distance-dependent ranging noise, amplified for rain
            # --- Correct rain ranging error injection (MAE-based, asymmetric)
            dist_surf = dist[surf_mask]

            # Mean absolute error from empirical model
            mae = self.mae_rain(dist_surf)

            # Convert MAE -> sigma for half-normal
            sigma = mae / np.sqrt(2.0 / np.pi)

            # Sample half-normal (positive-only delay)
            dr = np.abs(self.rng.normal(0.0, sigma, size=n_surf))

            # Apply along beam direction (late return bias)
            points[surf_mask] += dirs[surf_mask] * dr[:, None]

            self.stats["error_added"] += n_surf

        # Step 7: Return the modified point cloud (exclude deleted points)
        return points[~del_mask].copy()

    def get_statistics(self):
        if self.stats["total"] == 0:
            return {}
        return {
            "rain_rate_mm_h": self.R,
            **self.stats,
            "delete_rate": self.stats["deleted"] / self.stats["total"],
            "backscatter_rate": self.stats["backscattered"] / self.stats["total"],
        }


    def mae_rain(self, dist):
        """
        Mean absolute ranging error [m]
        Empirically fitted from literature (Sensors 2023).
        """
        # normalize
        r = self.R / 100.0          # rain rate scaling
        d = dist / 20.0             # distance scaling (20 m reference)

        # ≈ 5 cm at 20 m, 100 mm/h
        mae = 0.05 * (r ** 0.7) * (d ** 1.0)

        return np.clip(mae, 0.0, 0.05)
