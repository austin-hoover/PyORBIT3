"""Test 2D envelope tracker in FODO lattice."""

import argparse
import copy
import math
import os
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform

from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_utils import collect_bunch
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton

from plot import plot_corner
from utils import build_rotation_matrix_xy
from utils import gen_dist

plt.style.use("style.mplstyle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kin-energy", type=float, default=0.0025)
    parser.add_argument("--intensity", type=float, default=3e10)
    parser.add_argument("--dist", type=str, default="waterbag")

    parser.add_argument("--rms-x", type=float, default=0.010)
    parser.add_argument("--rms-y", type=float, default=0.010)
    parser.add_argument("--rms-z", type=float, default=0.010)

    parser.add_argument("--rot-x", type=float, default=0.0)
    parser.add_argument("--rot-y", type=float, default=0.0)
    parser.add_argument("--rot-z", type=float, default=0.0)

    parser.add_argument("--nslice", type=int, default=1)
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--length", type=float, default=0.1)

    parser.add_argument("--sc-grid-res", type=int, default=64)
    parser.add_argument("--sc-ellipse-n", type=int, default=1)

    parser.add_argument("--nparts", type=int, default=100_000)
    parser.add_argument("--sc", type=int, default=0)

    parser.add_argument("--plot-bins", type=int, default=64)
    parser.add_argument("--plot-mask", action="store_true")
    return parser.parse_args()


def make_lattice(args: argparse.Namespace) -> AccLattice:
    node = DriftTEAPOT(length=args.length)
    node.setLength(args.length)
    node.setnParts(args.nslice)

    lattice = TEAPOT_Lattice()
    lattice.addNode(node)
    lattice.initialize()
    return lattice


def rotation_matrix_3d(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    return scipy.spatial.transform.Rotation.from_euler(
        "xyz", [angle_x, angle_y, angle_z]
    ).as_matrix()


def build_cov_matrix_xyz(
    rms_sizes: np.ndarray,
    rotation_matrix: np.ndarray = None
) -> np.ndarray:
    cov_matrix = np.diag(np.square(rms_sizes))
    if rotation_matrix is None:
        return cov_matrix
    return rotation_matrix @ cov_matrix @ rotation_matrix.T


def make_empty_bunch(args: argparse.Namespace) -> Bunch:
    bunch = Bunch()
    bunch.mass(mass_proton)
    bunch.getSyncParticle().kinEnergy(args.kin_energy)
    return bunch


def make_cov_xyz(args: argparse.Namespace) -> np.ndarray:
    bunch = make_empty_bunch(args)
    sync_part = bunch.getSyncParticle()

    cov_matrix = np.zeros((6, 6))

    rotation_matrix = rotation_matrix_3d(
        math.radians(args.rot_x), math.radians(args.rot_y), math.radians(args.rot_z)
    )
    print(rotation_matrix)

    cov_matrix_xyz = build_cov_matrix_xyz(
        rms_sizes=[args.rms_x, args.rms_y, args.rms_z],
        rotation_matrix=rotation_matrix
    )

    lorentz_matrix = np.diag([1.0, 1.0, 1.0 / sync_part.gamma()])
    cov_matrix_xyz = lorentz_matrix @ cov_matrix_xyz @ lorentz_matrix.T
    return cov_matrix_xyz


def make_bunch(args: argparse.Namespace) -> Bunch:
    bunch = make_empty_bunch(args)
    sync_part = bunch.getSyncParticle()

    cov_matrix_xyz = make_cov_xyz(args)

    particles = np.zeros((args.nparts, 6))
    particles[:, (0, 2, 4)] = gen_dist(size=args.nparts, cov_matrix=cov_matrix_xyz, name=args.dist)
    for x, xp, y, yp, z, dE in particles:
        bunch.addParticle(x, xp, y, yp, z, dE)

    size_global = bunch.getSizeGlobal()
    bunch.macroSize(args.intensity / size_global)
    return bunch


def main(args: argparse.Namespace) -> None:
    lattice = make_lattice(args)
    bunch = make_bunch(args)


def get_bunch_cov(bunch: Bunch) -> np.ndarray:
    twiss_calc = BunchTwissAnalysis()
    twiss_calc.analyzeBunch(bunch)

    cov_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            cov_matrix[i, j] = twiss_calc.getCorrelation(j, i)
            cov_matrix[j, i] = cov_matrix[i, j]
    return cov_matrix


def track(lattice: TEAPOT_Lattice, bunch: Bunch, nsteps: int) -> dict:
    bunch_out = Bunch()
    bunch.copyBunchTo(bunch_out)

    history = {}
    for key in ["rms_x", "rms_y", "rms_z", "eps_x", "eps_y", "eps_z", "s"]:
        history[key] = []

    for step in range(nsteps):
        if step > 0:
            lattice.trackBunch(bunch_out)

        cov_matrix = 1e6 * get_bunch_cov(bunch_out)
        x_rms = math.sqrt(cov_matrix[0, 0])
        y_rms = math.sqrt(cov_matrix[2, 2])
        z_rms = math.sqrt(cov_matrix[4, 4])
        eps_x = np.sqrt(np.linalg.det(cov_matrix[0:2, 0:2]))
        eps_y = np.sqrt(np.linalg.det(cov_matrix[2:4, 2:4]))
        eps_z = np.sqrt(np.linalg.det(cov_matrix[4:6, 4:6]))
        s = step * lattice.getLength()

        history["rms_x"].append(x_rms)
        history["rms_y"].append(y_rms)
        history["rms_z"].append(z_rms)
        history["eps_x"].append(eps_x)
        history["eps_y"].append(eps_y)
        history["eps_z"].append(eps_z)
        history["s"].append(s)

        message = f"s={s:0.2f} xrms={x_rms:0.3f} yrms={y_rms:0.3f} epsx={eps_x:0.3f} epsy={eps_y:0.3f}"
        print(message)

    particles_out = collect_bunch(bunch_out)["coords"]
    particles_out[:, :] *= 1000.0
    return {
        "particles": particles_out.copy(),
        "history": history,
    }


def main(args: argparse.Namespace) -> None:
    path = pathlib.Path(__file__)
    output_dir = os.path.join("outputs", path.stem, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for key in ["SC3D", "SCUnifEllipse"]:
        results[key] = {}

    # Make initial bunch
    bunch = make_bunch(args)

    # Track bunch with SC3D space charge nodes
    lattice = make_lattice(args)
    if args.sc:
        sc_calc = SpaceChargeCalc3D(args.sc_grid_res, args.sc_grid_res, args.sc_grid_res)
        sc_path_length_min = 1.00e-06
        sc_nodes = setSC3DAccNodes(lattice, sc_path_length_min, sc_calc)

    print("TRACK SC3D")
    results["SC3D"] = track(lattice, bunch, nsteps=args.nsteps)

    # Track bunch with SCUnifEllipse space charge nodes
    lattice = make_lattice(args)
    if args.sc:
        sc_calc = SpaceChargeCalcUnifEllipse(args.sc_ellipse_n)
        sc_nodes = setUniformEllipsesSCAccNodes(lattice, sc_path_length_min, sc_calc)

    print("TRACK SCUnifEllipse")
    results["SCUnifEllipse"] = track(lattice, bunch, nsteps=args.nsteps)

    # Analysis
    # ------------------------------------------------------------------------------

    for model, result in results.items():
        history = result["history"]
        for key in history:
            history[key] = np.array(history[key])

    # Print errors
    for key in results["SCUnifEllipse"]["history"]:
        deltas = (
            results["SCUnifEllipse"]["history"][key] - results["SC3D"]["history"][key]
        )
        print("key:", key)
        print("max_abs_delta:", np.max(np.abs(deltas)))
        print("avg_abs_delta:", np.mean(np.abs(deltas)))

    # Plot rms size and emittance
    for key in ["eps_x", "eps_y", "eps_z", "rms_x", "rms_y", "rms_z"]:
        fig, ax = plt.subplots(figsize=(5, 3))
        for i, model in enumerate(["SC3D", "SCUnifEllipse"]):
            plot_kws = {}
            plot_kws["color"] = ["black", "red"][i]
            plot_kws["lw"] = [None, 0][i]
            ax.plot(results[model]["history"]["s"], results[model]["history"][key], marker=".", label=model, **plot_kws)
        ax.set_ylim(0.0, ax.get_ylim()[1] * 2.0)
        ax.set_xlabel("s [m]")
        ax.set_ylabel(key)
        ax.legend(loc="upper right")
        plt.savefig(os.path.join(output_dir, f"fig_{key}"))
        plt.close()

    # Set plot limits
    particles = results["SC3D"]["particles"]
    xmax = 4.0 * np.std(particles, axis=0)
    limits = list(zip(-xmax, xmax))
    dims = ["x", "xp", "y", "yp", "z", "dE"]
    labels = ["x [mm]", "xp [mrad]", "y [mm]", "yp [mrad]", "z", "dE"]

    # Plot x-x', y-y', x-y
    for axis in [(0, 1), (2, 3), (0, 2)]:
        fig, axs = plt.subplots(figsize=(6, 3), ncols=2, sharex=True, sharey=True)
        for ax, model in zip(axs, results):
            particles = results[model]["particles"]
            values, edges = np.histogramdd(
                particles[:, axis], bins=args.plot_bins, range=[limits[k] for k in axis]
            )
            if args.plot_mask:
                values = np.ma.masked_equal(values, 0.0)
            ax.pcolormesh(edges[0], edges[1], values.T)
            ax.set_xlabel(labels[axis[0]])
            ax.set_ylabel(labels[axis[1]])
            ax.set_title(model)
        plt.savefig(
            os.path.join(output_dir, f"fig_dist_{dims[axis[0]]}_{dims[axis[1]]}")
        )
        plt.close()

    # Plot corner
    for model in results:
        particles = results[model]["particles"]
        fig, axs = plot_corner(
            particles,
            limits=limits,
            bins=args.plot_bins,
            labels=labels,
            mask=args.plot_mask,
        )
        plt.savefig(os.path.join(output_dir, f"fig_dist_corner_{model}"))
        plt.close()


if __name__ == "__main__":
    main(parse_args())
