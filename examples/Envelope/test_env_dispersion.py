"""Dipole transport with momentum spread and an off-momentum centroid."""

import numpy as np

from orbit.envelope import Envelope
from orbit.teapot import BendTEAPOT, DriftTEAPOT

from example_helpers import (
    compare_final, make_bunch, make_lattice, output_dir, parser, particle_args,
    plot_series, positive_float, reference_bunch, save_histories, track_history,
)


def main():
    cli = parser(__doc__)
    particle_args(cli)
    cli.add_argument("--sigma-dp", type=positive_float, default=1e-3, help="rms dp/p")
    cli.add_argument("--offset-dp", type=float, default=5e-4, help="centroid dp/p")
    cli.add_argument("--angle", type=positive_float, default=20.0, help="bend angle [degrees]")
    args = cli.parse_args()
    path = output_dir(args, __file__)
    reference = reference_bunch(args.kin_energy)
    sync = reference.getSyncParticle()
    # ORBIT's sixth coordinate is dE [GeV]; to first order dE = beta*p*dp/p.
    energy_per_delta = sync.beta() * sync.momentum()
    cov = np.diag(np.square([1e-3, 0.2e-3, 1e-3, 0.2e-3, 2e-3,
                            energy_per_delta * args.sigma_dp]))
    centroid = np.zeros(6)
    centroid[5] = energy_per_delta * args.offset_dp
    bunch = make_bunch(cov, args, centroid=centroid)
    envelope = Envelope(bunch=bunch).copy()
    lattice = make_lattice([
        DriftTEAPOT(length=0.25, nparts=args.nslice),
        BendTEAPOT(length=1.0, theta=np.radians(args.angle), nparts=args.nslice),
        DriftTEAPOT(length=1.0, nparts=args.nslice),
    ])
    histories = {
        "envelope": track_history(lattice, envelope, fit=args.fit),
        "particles": track_history(lattice, bunch),
    }
    for history in histories.values():
        history["dispersion_x"] = energy_per_delta * history["cov"][:, 0, 5] / history["cov"][:, 5, 5]
        history["dispersion_xp"] = energy_per_delta * history["cov"][:, 1, 5] / history["cov"][:, 5, 5]
        history["rms_x_betatron"] = np.sqrt(np.maximum(
            0.0, history["cov"][:, 0, 0] - history["cov"][:, 0, 5]**2 / history["cov"][:, 5, 5]))
    save_histories(path, histories)
    plot_series(path, "dispersion.png", histories, [
        ("dispersion_x", None, 1, "D x [m]"),
        ("dispersion_xp", None, 1, "D x' [rad]"),
        ("mean", 0, 1e3, "x centroid [mm]"),
    ])
    plot_series(path, "rms.png", histories, [
        ("rms_x", None, 1e3, "total x rms [mm]"),
        ("rms_x_betatron", None, 1e3, "betatron x rms [mm]"),
    ])
    compare_final(histories["envelope"], histories["particles"])


if __name__ == "__main__":
    main()
