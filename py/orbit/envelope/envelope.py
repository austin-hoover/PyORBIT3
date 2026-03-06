import math

import numpy as np  # can switch to internal Matrix class later

from orbit.lattice import AccLattice
from orbit.lattice import AccNode


ENTRANCE = AccNode.ENTRANCE
BODY = AccNode.BODY
EXIT = AccNode.EXIT

BEFORE = AccNode.BEFORE
AFTER = AccNode.AFTER


class Envelope:
    def __init__(self, mass: float, kin_energy: float, cov_matrix: np.ndarray, centroid: np.ndarray = None, intensity: float = 0.0) -> None:
        """Constructor.
        
        Args:
            mass: Particle mass [GeV / c^2].
            kin_energy: Kinetic energy of synchronous particle [GeV].
            cov_matrix: Bunch covariance matrix, shape (6, 6).
            centroid: Bunch centroid vector, shape (6,).
            intensity: Number of particles in the bunch.
        """
        self.mass = mass
        self.kin_energy = kin_energy
        
        self.cov_matrix = cov_matrix
        
        self.centroid = centroid
        if self.centroid is None:
            self.centroid = np.zeros(6)
        
        # Add extra dimension to centroid vector. (Always equal to 1).
        self.centroid = self.centroid[:6]
        self.centroid = np.append(self.centroid, 1.0)

        self.intensity = intensity

    def propagate(self, matrix: np.ndarray) -> None:
        """Linear propagation of covariance matrix and centroid."""
        matrix_sub = matrix[0:6, 0:6]
        self.cov_matrix = np.linalg.multi_dot([matrix_sub, self.cov_matrix, matrix_sub.T])
        self.centroid = np.matmul(matrix, self.centroid)

    
class EnvelopeTracker:
    """Tracks beam envelope/centroid through linear lattice."""
    def __init__(self, lattice: AccLattice) -> None:
        self.lattice = lattice

    def add_sc_nodes(self, min_sep: float, max_sep: float) -> None:
        """Add space charge kicks to the lattice as child nodes."""
        raise NotImplementedError

    def track(self, envelope: Envelope) -> None:   
        """Track envelope through lattice.
        
        This assumes all nodes have `getMatrix` method.
        """
        for node in self.lattice.getNodes():
            for child_node in node.getChildNodes(ENTRANCE):
                envelope.propagate(child_node.getMatrix())

            for part_index in range(node.getnParts()):
                for child_node in node.getChildNodes(BODY, part_index, place_in_part=BEFORE):
                    envelope.propagate(child_node.getMatrix())

                envelope.propagate(node.getMatrix(part_index))

                for child_node in node.getChildNodes(BODY, part_index, place_in_part=AFTER):
                    envelope.propagate(child_node.getMatrix())
            
            for child_node in node.getChildNodes(EXIT):
                envelope.propagate(child_node.getMatrix())


class EnvelopeSpaceChargeKick(AccNode):
    """Base class for envelope space charge nodes."""
    def __init__(self, length: float) -> None:
        super().__init__()
        self.length = length

    def getMatrix(self, envelope: Envelope) -> None:
        raise NotImplementedError
    

class EnvelopeSpaceChargeKick2D(EnvelopeSpaceChargeKick):
    """Applies two-dimensional linear space charge kick to beam envelope."""
    def __init__(self, length: float) -> None:
        super().__init__(length)

    def getMatrix(self, envelope: Envelope) -> None:
        raise NotImplementedError


class EnvelopeSpaceChargeKick3D(EnvelopeSpaceChargeKick):
    """Applies three-dimensional linear space charge kick to beam envelope."""
    def __init__(self, length: float) -> None:
        super().__init__(length)

    def getMatrix(self, envelope: Envelope) -> None:
        raise NotImplementedError