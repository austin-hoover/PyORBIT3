import math

import numpy as np  # can switch to internal Matrix class later

from orbit.core.bunch import SyncParticle
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT
from orbit.teapot import TiltTEAPOT
from orbit.teapot import FringeFieldTEAPOT
from orbit.utils.consts import speed_of_light


ENTRANCE = AccNode.ENTRANCE
BODY = AccNode.BODY
EXIT = AccNode.EXIT

BEFORE = AccNode.BEFORE
AFTER = AccNode.AFTER


class Envelope:
    def __init__(self, sync_part: SyncParticle, cov_matrix: np.ndarray, centroid: np.ndarray = None, intensity: float = 0.0) -> None:
        """Constructor.
        
        Args:
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

    def getCentroid(self) -> np.ndarray:
        """Return copy of centroid."""
        return np.copy(self.centroid[:6])
    
    def getCovMatrix(self) -> np.ndarray:
        """Return copy of covariance matrix."""
        return np.copy(self.cov_matrix)

    def propagate(self, matrix: np.ndarray) -> None:
        """Linear propagation of covariance matrix and centroid."""
        matrix_sub = matrix[0:6, 0:6]
        self.cov_matrix = np.linalg.multi_dot([matrix_sub, self.cov_matrix, matrix_sub.T])
        self.centroid = np.matmul(matrix, self.centroid)


class EnvelopeTracker:
    """Tracks beam envelope/centroid through linear lattice."""
    def __init__(self, lattice: AccLattice) -> None:
        self.lattice = lattice

    def addSpaceChargeNodes(self, min_sep: float, max_sep: float) -> None:
        """Add space charge kicks to the lattice as child nodes."""
        raise NotImplementedError
    
    def getMatrix(self, node: AccNode, envelope: Envelope, part_index: int = None) -> np.ndarray:
        """Compute transfer matrix."""
        matrix = None

        if type(node) is DriftTEAPOT:
            length = node.getLength(part_index)

            sync_part = envelope.sync_part
            dp_p_coeff = 1.0 / (sync_part.momentum() * sync_part.beta())
            dp_p = envelope.centroid[5] * dp_p_coeff

            matrix = np.identity(7)
            matrix[0, 1] = length / (1.0 + dp_p)
            matrix[2, 3] = length / (1.0 + dp_p)
            # TO DO: longitudinal

        elif type(node) is TiltTEAPOT:
            angle = node.getTiltAngle()
            cs = np.cos(angle)
            sn = np.sin(angle)
            matrix = np.identity(7)
            matrix[0, 0] = matrix[1, 1] = +cs
            matrix[0, 2] = matrix[1, 3] = +sn
            matrix[2, 0] = matrix[3, 1] = -sn
            matrix[2, 2] = matrix[3, 3] = +cs

        elif type(node) is FringeFieldTEAPOT:
            matrix = np.identity(7)

        else:
            raise ValueError(f"No transfer matrix for node {node}")

        return matrix

    def track(self, envelope: Envelope) -> None:   
        """Track envelope through lattice."""
        for node in self.lattice.getNodes():
            for child_node in node.getChildNodes(ENTRANCE):
                matrix = self.getMatrix(child_node, envelope)
                envelope.propagate(matrix)

            for part_index in range(node.getnParts()):
                for child_node in node.getChildNodes(BODY, part_index, place_in_part=BEFORE):
                    matrix = self.getMatrix(child_node, envelope)
                    envelope.propagate(matrix)

                matrix = self.getMatrix(node, envelope, part_index)
                envelope.propagate(matrix)

                for child_node in node.getChildNodes(BODY, part_index, place_in_part=AFTER):
                    matrix = self.getMatrix(child_node, envelope)
                    envelope.propagate(matrix)
            
            for child_node in node.getChildNodes(EXIT):
                matrix = self.getMatrix(child_node, envelope)
                envelope.propagate(matrix)


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
    