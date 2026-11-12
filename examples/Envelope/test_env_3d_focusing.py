"""Finite-emittance focusing: 3D envelope, uniform ellipsoid, and PIC.

A 6D waterbag has a nonuniform spatial projection. The uniform-ellipsoid
closure is therefore an approximation; PIC differences need not vanish with
more particles. The ellipsoid particle solver assumes upright spatial axes.
"""

import numpy as np

from orbit.core.spacecharge import SpaceChargeCalc3D, SpaceChargeCalcUnifEllipse
from orbit.envelope import Envelope
from orbit.space_charge.sc3d import setSC3DAccNodes, setUniformEllipsesSCAccNodes
from orbit.teapot import DriftTEAPOT, QuadTEAPOT

from example_helpers import (
    compare_final, make_bunch, make_lattice, output_dir, parser, particle_args,
    plot_series, positive_float, positive_int, save_histories, track_history,
)


def make_channel(args):
    return make_lattice([
        QuadTEAPOT(length=0.05, kq=3.0, nparts=args.nslice),
        DriftTEAPOT(length=0.1, nparts=args.nslice),
        QuadTEAPOT(length=0.05, kq=-3.0, nparts=args.nslice),
        DriftTEAPOT(length=0.1, nparts=args.nslice),
    ])


def main():
    cli = parser(__doc__)
    particle_args(cli)
    cli.add_argument("--cells", type=positive_int, default=5)
    cli.add_argument("--intensity", type=positive_float, default=1e9)
    cli.add_argument("--sc", type=int, choices=(0, 1), default=1)
    cli.add_argument("--sc-grid", type=positive_int, default=32)
    cli.add_argument("--dist", choices=("waterbag", "gauss"), default="waterbag")
    cli.add_argument("--models", nargs="+", choices=("ellipsoid", "pic"), default=["ellipsoid", "pic"])
    args = cli.parse_args()
    if args.sc_grid < 4:
        cli.error("--sc-grid must be at least 4")
    path = output_dir(args, __file__)
    cov = np.diag(np.square([2e-3, 0.5e-3, 1.5e-3, 0.5e-3, 10e-3, 1e-6]))
    initial = make_bunch(cov, args, intensity=args.intensity, dist=args.dist)
    envelope = Envelope(bunch=initial).copy()
    histories = {"envelope": track_history(
        make_channel(args), envelope, cells=args.cells,
        sc="3d" if args.sc else None, fit=args.fit, static=True,
    )}
    for model in dict.fromkeys(args.models):
        # Fresh lattice and identical seeded particles for each solver.
        lattice = make_channel(args)
        bunch = make_bunch(cov, args, intensity=args.intensity, dist=args.dist)
        if args.sc:
            if model == "ellipsoid":
                calculator = SpaceChargeCalcUnifEllipse(1)
                setUniformEllipsesSCAccNodes(lattice, 1e-8, calculator)
            else:
                calculator = SpaceChargeCalc3D(args.sc_grid, args.sc_grid, args.sc_grid)
                setSC3DAccNodes(lattice, 1e-8, calculator)
        histories[model] = track_history(lattice, bunch, cells=args.cells)
        print(f"Envelope versus {model}:")
        compare_final(histories["envelope"], histories[model])
    save_histories(path, histories)
    plot_series(path, "rms.png", histories, [
        ("rms_x", None, 1e3, "x rms [mm]"),
        ("rms_y", None, 1e3, "y rms [mm]"),
        ("rms_z", None, 1e3, "z rms [mm]"),
    ])
    plot_series(path, "emittances.png", histories, [
        ("eps_x_n", None, 1e6, "normalized x [mm mrad]"),
        ("eps_y_n", None, 1e6, "normalized y [mm mrad]"),
        ("eps_z_n", None, 1e6, "normalized z [mm MeV]"),
    ])


if __name__ == "__main__":
    main()
