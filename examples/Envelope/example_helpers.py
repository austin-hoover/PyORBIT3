"""Shared setup, diagnostics, and output for the small Envelope tutorials."""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit.core.bunch import Bunch
from orbit.envelope import Envelope, EnvelopeAnalysis
from orbit.lattice import AccActionsContainer
from orbit.teapot import DriftTEAPOT, QuadTEAPOT, TEAPOT_Lattice
from orbit.utils.consts import mass_proton


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parser(description, tracking_options=True):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--kin-energy", type=positive_float, default=0.0025, help="GeV")
    if tracking_options:
        result.add_argument("--nslice", type=positive_int, default=16)
        result.add_argument("--fit", type=int, choices=(0, 1), default=0)
    result.add_argument("--output-dir", type=Path)
    return result


def particle_args(result):
    result.add_argument("--nparts", type=positive_int, default=20_000)
    result.add_argument("--seed", type=int, default=42)


def output_dir(args, script):
    path = args.output_dir
    if path is None:
        path = Path(__file__).parent / "outputs" / Path(script).stem / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True, exist_ok=True)
    plt.style.use(Path(__file__).with_name("style.mplstyle"))
    print(f"Output: {path.resolve()}")
    return path


def make_lattice(nodes):
    lattice = TEAPOT_Lattice()
    for node in nodes:
        node.setUsageFringeFieldIN(False)
        node.setUsageFringeFieldOUT(False)
        lattice.addNode(node)
    lattice.initialize()
    return lattice


def fodo(nslice, kq=0.25):
    return make_lattice([
        QuadTEAPOT(length=0.5, kq=kq, nparts=nslice),
        DriftTEAPOT(length=1.0, nparts=nslice),
        QuadTEAPOT(length=1.0, kq=-kq, nparts=nslice),
        DriftTEAPOT(length=1.0, nparts=nslice),
        QuadTEAPOT(length=0.5, kq=kq, nparts=nslice),
    ])


def reference_bunch(kin_energy):
    bunch = Bunch()
    bunch.mass(mass_proton)
    bunch.charge(1.0)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    return bunch


def make_bunch(cov, args, intensity=0.0, centroid=None, dist="gauss"):
    """Seeded samples with exactly the requested population covariance.

    Whitening removes sampling error from the initial second moments so the
    comparisons measure transport error. This is a serial example (one MPI rank).
    """
    if args.nparts < 8:
        raise ValueError("--nparts must be at least 8 for a full-rank 6D sample")
    rng = np.random.default_rng(args.seed)
    coords = rng.normal(size=(args.nparts, 6))
    if dist == "waterbag":
        coords /= np.linalg.norm(coords, axis=1)[:, None]
        coords *= rng.random(args.nparts)[:, None] ** (1.0 / 6.0)
    coords -= coords.mean(axis=0)
    sample_cov = coords.T @ coords / args.nparts
    coords = np.linalg.solve(np.linalg.cholesky(sample_cov), coords.T).T
    coords = coords @ np.linalg.cholesky(cov).T
    if centroid is not None:
        coords += centroid
    bunch = reference_bunch(args.kin_energy)
    bunch.macroSize(intensity / args.nparts)
    for row in coords:
        bunch.addParticle(*row)
    return bunch


def snapshot(beam):
    env = beam if isinstance(beam, Envelope) else Envelope(bunch=beam)
    return EnvelopeAnalysis(env).to_dict()


def track_history(lattice, beam, *, cells=1, sc=None, fit=False, static=False):
    """Record the entrance and every top-level node exit, including RF gaps."""
    history = {"s": [0.0]}
    for key, value in snapshot(beam).items():
        history[key] = [value]
    positions = lattice.getNodePositionsDict()
    offset = 0.0

    def observe(params):
        if params["parentNode"] is not lattice:
            return
        history["s"].append(offset + positions[params["node"]][1])
        for key, value in snapshot(beam).items():
            history[key].append(value)

    actions = AccActionsContainer()
    actions.addAction(observe, AccActionsContainer.EXIT)
    for cell in range(cells):
        offset = cell * lattice.getLength()
        if isinstance(beam, Envelope):
            lattice.trackEnvelope(beam, sc=sc, fit=bool(fit), static=static,
                                  action_container=actions)
        else:
            lattice.trackBunch(beam, actionContainer=actions)
    return {key: np.asarray(value) for key, value in history.items()}


def save_histories(path, histories):
    for name, history in histories.items():
        if not all(np.all(np.isfinite(value)) for value in history.values()):
            raise RuntimeError(f"Non-finite diagnostics in {name}")
        np.savez(path / f"{name}.npz", **history)


def plot_series(path, filename, histories, panels):
    """Panels are (key, component or None, multiplier, ylabel)."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.5 * len(panels)),
                             sharex=True, squeeze=False, constrained_layout=True)
    for ax, (key, component, scale, label) in zip(axes[:, 0], panels):
        for name, history in histories.items():
            values = history[key]
            if component is not None:
                values = values[:, component]
            ax.plot(history["s"], scale * values, label=name)
        ax.set_ylabel(label)
        if key.startswith(("rms", "eps", "eigen")):
            # Keep conserved quantities visually flat instead of magnifying
            # floating-point or particle noise with an offset y axis.
            ax.set_ylim(bottom=0.0)
        ax.legend()
    axes[-1, 0].set_xlabel("s [m]")
    fig.savefig(path / filename)
    plt.close(fig)


def compare_final(reference, other):
    scale = np.sqrt(np.diag(reference["cov"][-1]))
    error = np.max(np.abs(reference["cov"][-1] - other["cov"][-1]) / np.outer(scale, scale))
    print(f"Final max covariance difference / (rms_i rms_j): {error:.6e}")
    return error
