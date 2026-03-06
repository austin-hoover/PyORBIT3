"""Test envelope tracker."""
import math

import numpy as np

from orbit.core.bunch import Bunch
from orbit.envelope import Envelope
from orbit.envelope import EnvelopeTracker
from orbit.envelope import EnvelopeSpaceChargeKick2D
from orbit.envelope import EnvelopeSpaceChargeKick3D
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT
from orbit.teapot import TEAPOT_Lattice
from orbit.utils.consts import mass_proton


# Create lattice
nodes = [
    DriftTEAPOT(length=1.0),
]

lattice = TEAPOT_Lattice()
for node in nodes:
    lattice.addNode(node)

# Create bunch
bunch = Bunch()
bunch.mass(mass_proton)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(1.0)

cov_matrix = np.zeros((6, 6))
cov_matrix[0, 0] = 0.010 ** 2
cov_matrix[1, 1] = 0.010 ** 2
cov_matrix[2, 2] = 0.010 ** 2
cov_matrix[3, 3] = 0.010 ** 2
cov_matrix[4, 4] = 10.0 ** 2

# Create envelope
envelope = Envelope(
    sync_part=sync_part,
    cov_matrix=cov_matrix,
)

# Track envelope
print(envelope.getCentroid() * 1e3)
print(envelope.getCovMatrix() * 1e6)

tracker = EnvelopeTracker(lattice)
tracker.track(envelope)

print(envelope.getCentroid() * 1e3)
print(envelope.getCovMatrix() * 1e6)
