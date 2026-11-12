"""Solenoid coupling: projected emittances and transverse eigenemittances."""

import numpy as np

from orbit.envelope import Envelope
from orbit.teapot import DriftTEAPOT, SolenoidTEAPOT

from example_helpers import (
    compare_final, make_bunch, make_lattice, output_dir, parser, particle_args,
    plot_series, positive_int, save_histories, track_history,
)


def main():
    cli = parser(__doc__)
    particle_args(cli)
    cli.add_argument("--field", type=float, default=0.15, help="solenoid field [T]")
    cli.add_argument("--cells", type=positive_int, default=4)
    args = cli.parse_args()
    path = output_dir(args, __file__)
    # Unequal transverse emittances make exchange under coupling visible.
    cov = np.diag(np.square([2e-3, 0.5e-3, 1e-3, 0.25e-3, 1e-3, 1e-7]))
    bunch = make_bunch(cov, args)
    envelope = Envelope(bunch=bunch).copy()
    lattice = make_lattice([
        DriftTEAPOT(length=0.2, nparts=args.nslice),
        SolenoidTEAPOT(length=0.4, B=args.field, nparts=args.nslice),
        DriftTEAPOT(length=0.2, nparts=args.nslice),
    ])
    histories = {
        "envelope": track_history(lattice, envelope, cells=args.cells, fit=args.fit),
        "particles": track_history(lattice, bunch, cells=args.cells),
    }
    # x' and y' are canonical in the field-free observation planes. Calculate
    # the transverse invariants directly, avoiding a sort with longitudinal units.
    poisson = np.kron(np.eye(2), [[0.0, 1.0], [-1.0, 0.0]])
    for history in histories.values():
        eigenvalues = np.linalg.eigvals(history["cov"][:, :4, :4] @ poisson)
        history["eigen_xy"] = np.sort(np.abs(eigenvalues.imag), axis=1)[:, ::2]
        history["corr_xy"] = history["cov"][:, 0, 2] / np.sqrt(
            history["cov"][:, 0, 0] * history["cov"][:, 2, 2])
        history["corr_xyp"] = history["cov"][:, 0, 3] / np.sqrt(
            history["cov"][:, 0, 0] * history["cov"][:, 3, 3])
        drift = np.max(np.abs(history["eigen_xy"] / history["eigen_xy"][0] - 1))
        print(f"Transverse eigenemittance relative variation: {drift:.3e}")
    save_histories(path, histories)
    plot_series(path, "emittances.png", histories, [
        ("eps_x", None, 1e6, "projected x [mm mrad]"),
        ("eps_y", None, 1e6, "projected y [mm mrad]"),
        ("eigen_xy", 0, 1e6, "eigen 1 [mm mrad]"),
        ("eigen_xy", 1, 1e6, "eigen 2 [mm mrad]"),
    ])
    plot_series(path, "correlations.png", histories, [
        ("corr_xy", None, 1, "corr(x, y)"),
        ("corr_xyp", None, 1, "corr(x, y')"),
    ])
    compare_final(histories["envelope"], histories["particles"])


if __name__ == "__main__":
    main()
