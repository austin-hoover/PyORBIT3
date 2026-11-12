"""Match an uncoupled FODO envelope with 2D space charge."""

import numpy as np
from scipy.optimize import least_squares

from orbit.envelope import Envelope

from example_helpers import (
    fodo, output_dir, parser, plot_series, positive_float, positive_int,
    reference_bunch, save_histories, track_history,
)


def covariance(parameters, emittance, bunch_length):
    cov = np.zeros((6, 6))
    for axis, (log_beta, alpha) in zip((0, 2), parameters.reshape(2, 2)):
        beta = np.exp(log_beta)
        cov[axis:axis + 2, axis:axis + 2] = emittance * np.array([
            [beta, -alpha], [-alpha, (1.0 + alpha**2) / beta],
        ])
    cov[4, 4] = bunch_length**2 / 12.0
    return cov


def main():
    cli = parser(__doc__)
    cli.add_argument("--intensity", type=positive_float, default=5e9)
    cli.add_argument("--bunch-length", type=positive_float, default=5.0, help="uniform full length [m]")
    cli.add_argument("--emittance", type=positive_float, default=0.25e-6, help="geometric rms [m rad]")
    cli.add_argument("--cells", type=positive_int, default=25)
    cli.add_argument("--matching-steps", type=positive_int, default=4)
    cli.add_argument("--sc", type=int, choices=(0, 1), default=1)
    cli.add_argument("--mismatch", type=positive_float, default=1.3, help="x size multiplier; x' scales inversely")
    args = cli.parse_args()
    path = output_dir(args, __file__)
    lattice = fodo(args.nslice)
    reference = reference_bunch(args.kin_energy)
    template = Envelope(sync_part=reference.getSyncParticle(),
                        cov_matrix=np.eye(6), intensity=args.intensity).copy()

    # Start with the zero-current periodic Twiss parameters.
    matrix = lattice.trackEnvelope(template.copy(), fit=bool(args.fit), return_matrix=True)
    parameters = []
    for axis in (0, 2):
        block = matrix[axis:axis + 2, axis:axis + 2]
        cosine = np.trace(block) / 2.0
        if abs(cosine) >= 1.0:
            raise RuntimeError("The zero-current cell is unstable")
        sine = np.copysign(np.sqrt(1.0 - cosine**2), block[0, 1])
        parameters.extend([np.log(block[0, 1] / sine), (block[0, 0] - block[1, 1]) / (2.0 * sine)])
    parameters = np.asarray(parameters)
    sc = "2d" if args.sc else None

    def residual(values, intensity):
        env = template.copy()
        env.intensity = intensity
        env.cov_matrix = covariance(values, args.emittance, args.bunch_length)
        lattice.trackEnvelope(env, sc=sc, fit=bool(args.fit), static=True)
        result = []
        for axis, (log_beta, alpha) in zip((0, 2), values.reshape(2, 2)):
            result.extend([
                np.log(env.cov_matrix[axis, axis] / args.emittance) - log_beta,
                -env.cov_matrix[axis, axis + 1] / args.emittance - alpha,
            ])
        return np.asarray(result)

    # Continue from zero current to avoid starting far from the SC solution.
    for intensity in np.linspace(0.0, args.intensity, args.matching_steps + 1)[1:]:
        fit = least_squares(residual, parameters, args=(intensity,),
                            bounds=([-12, -100, -12, -100], [12, 100, 12, 100]),
                            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=500)
        parameters = fit.x
        error = np.max(np.abs(residual(parameters, intensity)))
        print(f"intensity={intensity:.6g}, periodic Twiss residual={error:.3e}")
        if not fit.success or error > 1e-6:
            raise RuntimeError(f"Matching did not converge: {fit.message}; residual={error}")

    matched = template.copy()
    matched.cov_matrix = covariance(parameters, args.emittance, args.bunch_length)
    mismatched = matched.copy()
    transform = np.diag([args.mismatch, 1.0 / args.mismatch, 1, 1, 1, 1])
    mismatched.cov_matrix = transform @ matched.cov_matrix @ transform.T
    np.savez(path / "match.npz", parameters=parameters, covariance=matched.cov_matrix,
             residual=residual(parameters, args.intensity))
    histories = {
        name: track_history(lattice, env, cells=args.cells, sc=sc, fit=args.fit, static=True)
        for name, env in (("matched", matched), ("mismatched", mismatched))
    }
    save_histories(path, histories)
    plot_series(path, "rms.png", histories, [
        ("rms_x", None, 1e3, "x rms [mm]"), ("rms_y", None, 1e3, "y rms [mm]"),
    ])
    # Compare cell boundaries, since the matched envelope breathes inside a cell.
    cov = histories["matched"]["cov"][::len(lattice.getNodes())]
    scale = np.sqrt(np.diag(cov[0])[:4])
    error = np.max(np.abs(cov[:, :4, :4] - cov[0, :4, :4]) / np.outer(scale, scale))
    print(f"Matched covariance repeatability at cell boundaries: {error:.3e}")


if __name__ == "__main__":
    main()
