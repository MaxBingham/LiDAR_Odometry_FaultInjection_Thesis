#TO DO: Possibly add geometric distortion to fog model
# --> Dont know how to implement Intensity


import numpy as np


class FogSimulator:

    def __init__(self, distance, V, a=-0.7, b=-0.024, epsilon = None, lambda_ = None):
        
        self.distance = float(distance) #distance
        self.V = float(V) #visibility
        self.gamma = -np.log(0.05) / V

        #Constants determined with chafer metric 

        self.epsilon = 0.23 * np.exp(-0.0082 *V ) if epsilon is None else float(epsilon)
        self.lambda_ = -0.00600 * V + 2.31 if lambda_ is None else float(lambda_)
        self.a = float(a)
        self.b = float(b)

        self.stats = {
            'total': 0,
            'deleted': 0,
            'backscattered': 0
            }
        
    
    #Expand with Fog + Rain model 

    def apply_noise(self, points: np.ndarray)  -> np.ndarray:
            
        if points is None or len(points) == 0:
            return points
        
        
        points = np.asarray(points, dtype=float)
        N = points.shape[0]
        self.stats['total'] += N
        
        #Calculate Distance + Direction (points is already xyz only)
        distance = np.linalg.norm(points, axis=1)     #distance from sensor
        distance_safe = np.maximum(distance, 1e-6)  #avoid division by zero

        #probability of modification (per-point, based on distance)
        p_modify = 1.0 - np.exp(-distance * self.epsilon) 
        
        # Validate (check if any values are out of bounds)
        if np.any(p_modify < 0) or np.any(p_modify > 1):
            raise ValueError(f"Invalid modification probability range: [{p_modify.min()}, {p_modify.max()}]")
        
        #sample points for modification
        modify_mask = np.random.rand(N) < p_modify  # np.array of bools

        #probability of deletion 
        self.p_delete = 1 + self.a * np.exp(self.b * self.V)
        self.p_delete = np.clip(self.p_delete, 0, 1)  # Ensure valid probability 

        #sample points deletion  
        delete_mask = modify_mask & (np.random.rand(N) < self.p_delete)

        #delete points first
        kept_points = points[~delete_mask].copy()
        self.stats['deleted'] += np.sum(delete_mask)



        #backscatter points - remap to kept_points indices
        backscatter_mask = modify_mask & ~delete_mask  # Points modified but not deleted
        backscatter_mask_kept = backscatter_mask[~delete_mask]
        idx_move_kept = np.where(backscatter_mask_kept)[0]
        N_move = len(idx_move_kept)


        #Identify new positions in kept_points
        if N_move > 0:
            # Recalculate distances and directions for kept_points
            distance_kept = np.linalg.norm(kept_points, axis=1)
            distance_kept_safe = np.maximum(distance_kept, 1e-6)
            u_dir_kept = kept_points / distance_kept_safe[:, None]
            
            distance_orig = distance_kept[idx_move_kept]

            R_sample = np.random.exponential(scale=self.lambda_, size=N_move)

            d_backscatter = np.minimum(R_sample, distance_orig)

            xyz_move = u_dir_kept[idx_move_kept] * d_backscatter[:, None]

            kept_points[idx_move_kept] = xyz_move
            
            self.stats['backscattered'] += N_move

        #Intenstiy modification could be added here if needed
        
        #Output 
        return kept_points
        
        '''
        # ===== START: GEOMETRIC DISTORTION ADDITION =====
        # Apply very light fog-like geometric distortion to simulate atmospheric warping
        if len(kept_points) > 0:
            distortion_strength = 0.0  # Very subtle distortion (1500cm scale)
            
            # Create wave-based displacement using point coordinates as input
            x, y, z = kept_points[:, 0], kept_points[:, 1], kept_points[:, 2]
            
            # Generate smooth wave patterns for displacement
            wave_x = distortion_strength * np.sin(y * 0.1 + z * 0.05)
            wave_y = distortion_strength * np.cos(x * 0.1 + z * 0.05)
            wave_z = distortion_strength * np.sin(x * 0.05 + y * 0.1) * 0.5  # Less vertical distortion
            
            # Apply distortion (additive displacement)
            kept_points[:, 0] += wave_x
            kept_points[:, 1] += wave_y
            kept_points[:, 2] += wave_z
        # ===== END: GEOMETRIC DISTORTION ADDITION =====
        '''


    def get_statistics(self):
        if self.stats['total'] == 0:
            return {}
        
        return {
            'total': self.stats['total'],
            'movement_rate': self.stats['moved'] / self.stats['total']
        }  