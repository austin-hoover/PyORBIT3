"""Space-charge slicing convergence and cold-run cost of analytic/fitted maps."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from orbit.envelope import Envelope

from example_helpers import fodo, output_dir, parser, positive_float, positive_int, reference_bunch


def propagate(args, nslice, fit):
    lattice = fodo(nslice)
    reference = reference_bunch(args.kin_energy)
    env = Envelope(sync_part=reference.getSyncParticle(), intensity=args.intensity,
                   cov_matrix=np.diag(np.square([3e-3, 0.2e-3, 3e-3, 0.2e-3, 5 / np.sqrt(12), 0.0]))).copy()
    start = perf_counter()
    for _ in range(args.cells):
        lattice.trackEnvelope(env, sc="2d", fit=bool(fit), fallback=False, static=True)
    return env.cov_matrix.copy(), perf_counter() - start


def main():
    cli = parser(__doc__, tracking_options=False)
    cli.add_argument("--slices", type=positive_int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    cli.add_argument("--reference-slices", type=positive_int, default=256)
    cli.add_argument("--cells", type=positive_int, default=4)
    cli.add_argument("--intensity", type=positive_float, default=5e9)
    args = cli.parse_args()
    args.slices = sorted(set(args.slices))
    if args.reference_slices <= max(args.slices):
        cli.error("--reference-slices must exceed every value in --slices")
    path = output_dir(args, __file__)
    reference, _ = propagate(args, args.reference_slices, fit=False)
    scale = np.sqrt(np.diag(reference)[:4])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    results = {"reference_cov": reference, "slices": args.slices}
    for fit, name in ((False, "analytic"), (True, "fitted")):
        errors, times, covariances = [], [], []
        for nslice in args.slices:
            cov, elapsed = propagate(args, nslice, fit)
            error = np.max(np.abs(cov[:4, :4] - reference[:4, :4]) / np.outer(scale, scale))
            errors.append(error)
            times.append(elapsed)
            covariances.append(cov)
            print(f"{name}: slices={nslice}, covariance error={error:.6e}, seconds={elapsed:.6f}")
        results.update({f"{name}_error": errors, f"{name}_seconds": times,
                        f"{name}_cov": covariances})
        axes[0].loglog(args.slices, errors, ".-", label=name)
        axes[1].loglog(times, errors, ".-", label=name)
    for ax in axes:
        ax.set_ylabel("scaled transverse covariance error")
        ax.legend()
    axes[0].set_xlabel("Slices per element")
    axes[1].set_xlabel("Tracking time, including map preparation [s]")
    fig.savefig(path / "convergence.png")
    plt.close(fig)
    if not all(np.all(np.isfinite(value)) for value in results.values()):
        raise RuntimeError("Non-finite convergence result")
    np.savez(path / "convergence.npz", **results)


if __name__ == "__main__":
    main()
