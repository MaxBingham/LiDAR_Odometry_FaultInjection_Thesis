import numpy as np


class FogSimulator:

    
    def __init__(self, sigma):
        self.sigma = float(sigma)
        self.stats = {
            'total': 0,
            'moved': 0 }
        
    
    #Expand with Fog + Rain model 

    def apply_noise(self, points: np.ndarray)  -> np.ndarray:
            
        if points is None or len(points) == 0:
            return points
        
        self.stats['total'] += len(points)

        noise = np.random.normal(loc=0.0, scale=self.sigma, size=points.shape) 

        moved_points = points + noise 

        self.stats['moved'] += len(points)

        return moved_points

    def get_statistics(self):
        if self.stats['total'] == 0:
            return {}
        
        return {
            'total': self.stats['total'],
            'movement_rate': self.stats['moved'] / self.stats['total']
        }  