from orbit.space_charge.scLatticeModifications import setSC_General_AccNodes
from .scAccNodes import SC2D_UniformEllipses_AccNode


def setUniformEllipses2DSCAccNodes(lattice, sc_path_length_min, space_charge_calculator):
    """Insert transverse ellipse kicks using sqrt(12)*rms(z) at each kick."""
    nodes = setSC_General_AccNodes(
        lattice, sc_path_length_min, space_charge_calculator, SC2D_UniformEllipses_AccNode
    ) or []
    for node in nodes:
        node.setName(node.getName() + "UnifEllsSC2D")
    lattice.initialize()
    return nodes
