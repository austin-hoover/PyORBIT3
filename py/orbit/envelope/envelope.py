import math

# Use NumPy for now; can switch to internal Matrix class later.
import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.bunch import SyncParticle
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


class Envelope:
    def __init__(self, sync_part: SyncParticle, cov_matrix: np.ndarray, centroid: np.ndarray = None, intensity: float = 0.0) -> None:
        """Constructor.
        
        sync_part: Synchronous particle.
        cov_matrix: Bunch covariance matrix, shape (6, 6).
        centroid: Bunch centroid vector, shape (6,).
        intensity: Number of particles in the bunch.
        """
        self.sync_part = sync_part
        
        self.cov_matrix = cov_matrix
        
        self.centroid = centroid
        if self.centroid is None:
            self.centroid = np.zeros(6)
        
        # Add extra dimension to centroid vector. (Always equal to 1).
        self.centroid = self.centroid[:6]
        self.centroid = np.append(self.centroid, 1.0)

        self.intensity = intensity

    def get_sc_matrix_2d(self, length: float) -> None:
        """Return matrix for linear space charge kick from uniform-density ellipse (2D)."""
        raise NotImplementedError

    def get_sc_matrix_3d(self, length: float) -> None:
        """Return matrix for linear space charge kick from uniform-density ellipsoid (3D)."""
        raise NotImplementedError


class EnvelopeTracker:
    def __init__(self, lattice: AccLattice, envelope: Envelope) -> None:
        self.lattice = lattice
        self.envelope = envelope

    def _apply_matrix(self, matrix: np.ndarray) -> None:
        # Track centroid: x -> Mx.
        self.centroid = np.matmul(matrix, self.centroid)
        
        # Track covariance matrix: S -> M S M^T.
        matrix_sub = matrix[:6, :6]
        self.cov_matrix = np.linalg.multi_dot([matrix_sub, self.cov_matrix, matrix_sub.T])

    