import math
import numpy as np

from orbit.core.bunch import Bunch


class Envelope:
    def __init__(self, cov_matrix: np.ndarray, centroid: np.ndarray = None, bunch: Bunch, intensity: float = 0.0) -> None:
        self.bunch = bunch
        self.sync_part = self.bunch.getSyncParticle()
        
        self.cov_matrix = cov_matrix
        self.centroid = centroid
        if self.centroid is None:
            self.centroid = np.zeros(6)

        self.intensity = intensity