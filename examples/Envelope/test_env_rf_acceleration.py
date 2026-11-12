"""A small quadrupole/RF linac: acceleration, focusing, and emittance."""

import numpy as np

from orbit.core.linac import MatrixRfGap
from orbit.envelope import Envelope
from orbit.py_linac.lattice import BaseRF_Gap, Drift, LinacAccLattice, Quad, RF_Cavity, Sequence

from example_helpers import (
    compare_final, make_bunch, output_dir, parser, particle_args, plot_series,
    positive_float, positive_int, save_histories, track_history,
)


def make_channel(args):
    lattice = LinacAccLattice("short accelerating channel")
    sequence = Sequence("channel")
    sequence.setLinacAccLattice(lattice)
    for cell in range(args.cells):
        nodes = []
        for label, gradient in (("qf", 0.5), ("qd", -0.5)):
            quad = Quad(f"{label}_{cell}")
            quad.setLength(0.05)
            quad.setnParts(args.nslice)
            quad.setParam("dB/dr", gradient)
            quad.setUsageFringeFieldIN(False)
            quad.setUsageFringeFieldOUT(False)
            drift = Drift(f"drift_{label}_{cell}")
            drift.setLength(0.1)
            drift.setnParts(args.nslice)
            nodes.extend([quad, drift])
        cavity = RF_Cavity(f"cavity_{cell}")
        cavity.setFrequency(args.frequency)
        cavity.setPhase(np.radians(args.phase))
        cavity.setAmp(1.0)
        gap = BaseRF_Gap(f"gap_{cell}")
        gap.setParam("E0TL", args.voltage)
        gap.setCppGapModel(MatrixRfGap())
        cavity.addRF_GapNode(gap)
        sequence.addRF_Cavity(cavity)
        nodes.append(gap)
        for node in nodes:
            sequence.addNode(node)
            lattice.addNode(node)
    lattice.initialize()
    return lattice


def main():
    cli = parser(__doc__)
    particle_args(cli)
    cli.add_argument("--cells", type=positive_int, default=6)
    cli.add_argument("--frequency", type=positive_float, default=402.5e6, help="Hz")
    cli.add_argument("--voltage", type=positive_float, default=1e-4, help="E0TL [GeV]")
    cli.add_argument("--phase", type=float, default=-30.0, help="synchronous phase [degrees]")
    args = cli.parse_args()
    path = output_dir(args, __file__)
    cov = np.diag(np.square([1e-3, 0.3e-3, 1e-3, 0.3e-3, 0.5e-3, 1e-6]))
    bunch = make_bunch(cov, args)
    envelope = Envelope(bunch=bunch).copy()
    lattice = make_channel(args)
    lattice.trackDesignBunch(bunch)
    # Dynamic maps are essential: each gap changes the reference energy.
    histories = {
        "envelope": track_history(lattice, envelope, fit=args.fit, static=False),
        "particles": track_history(lattice, bunch),
    }
    save_histories(path, histories)
    plot_series(path, "acceleration.png", histories, [
        ("kin_energy", None, 1e3, "kinetic energy [MeV]"),
        ("rms_z", None, 1e3, "z rms [mm]"),
        ("rms_x", None, 1e3, "x rms [mm]"),
    ])
    plot_series(path, "emittances.png", histories, [
        ("eps_x", None, 1e6, "geometric x [mm mrad]"),
        ("eps_y", None, 1e6, "geometric y [mm mrad]"),
        ("eps_x_n", None, 1e6, "normalized x [mm mrad]"),
        ("eps_y_n", None, 1e6, "normalized y [mm mrad]"),
    ])
    for name, history in histories.items():
        print(f"{name}: energy {1e3 * history['kin_energy'][0]:.4f} -> "
              f"{1e3 * history['kin_energy'][-1]:.4f} MeV")
        print(f"{name}: normalized x emittance relative change "
              f"{history['eps_x_n'][-1] / history['eps_x_n'][0] - 1:.3e}")
    compare_final(histories["envelope"], histories["particles"])


if __name__ == "__main__":
    main()
