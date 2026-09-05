import argparse
import copy
import math
import os
import pathlib
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.bunch import SyncParticle
from orbit.core.linac import BaseRfGap
from orbit.core.linac import MatrixRfGap
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist3D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_utils import collect_bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.lattice import LinacAccLattice
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.utils.consts import mass_proton
from orbit.utils.consts import mass_electron
from orbit.utils.consts import charge_electron

from diagnostics import BunchMonitor

plt.style.use("../style.mplstyle")


mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
mpi_rank = orbit_mpi.MPI_Comm_rank(mpi_comm)
mpi_size = orbit_mpi.MPI_Comm_size(mpi_comm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="gauss")
    parser.add_argument("--nparts", type=int, default=100_000)
    parser.add_argument("--rot", type=float, default=0.0)
    parser.add_argument("--sc", type=int, default=0)
    parser.add_argument("--ds", type=float, default=0.1)
    parser.add_argument("--current", type=float, default=0.038)
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--seq-stop", type=str, default="CCL2")
    return parser.parse_args()


def make_lattice(args: argparse.Namespace) -> LinacAccLattice:
    seq_names = [
        "MEBT",
        "DTL1",
        "DTL2",
        "DTL3",
        "DTL4",
        "DTL5",
        "DTL6",
        "CCL1",
        "CCL2",
        "CCL3",
        "CCL4",
        "SCLMed",
        "SCLHigh",
        "HEBT1",
        "HEBT2",
    ]
    if args.seq_stop:
        index = seq_names.index(args.seq_stop) + 1
        seq_names = seq_names[:index]

    sns_linac_factory = SNS_LinacLatticeFactory()
    sns_linac_factory.setMaxDriftLength(args.ds)
    lattice = sns_linac_factory.getLinacAccLattice(seq_names, "inputs/sns_linac.xml")

    for node in lattice.getNodes():
        try:
            node.setUsageFringeFieldIN(False)
            node.setUsageFringeFieldOUT(False)
        except:
            pass

    rf_gaps = lattice.getRF_Gaps()
    for rf_gap in rf_gaps:
        rf_gap.setCppGapModel(MatrixRfGap())
    return lattice


def make_bunch(args: argparse.Namespace) -> Bunch:
    kin_energy = 0.0025  # [GeV]
    mass = mass_proton + 2.0 * mass_electron
    frequency = 402.5e06
    charge = -1.0
    intensity = args.current / frequency / (math.fabs(charge) * charge_electron)

    bunch = Bunch()
    bunch.mass(mass)
    bunch.macroSize(intensity / args.nparts)
    bunch.charge(charge)

    sync_part = bunch.getSyncParticle()
    sync_part.kinEnergy(kin_energy)
    sync_part.time(0.0)

    alpha_x, beta_x, eps_x = (-1.962, 0.183, 2.874e-06)
    alpha_y, beta_y, eps_y = (+1.768, 0.162, 2.874e-06)
    alpha_z, beta_z, eps_z = (-0.0196, 116.414, 1.651e-08)

    twiss_x = TwissContainer(alpha_x, beta_x, eps_x)
    twiss_y = TwissContainer(alpha_y, beta_y, eps_y)
    twiss_z = TwissContainer(alpha_z, beta_z, eps_z)

    if args.dist == "waterbag":
        dist = WaterBagDist3D(twiss_x, twiss_y, twiss_z)
    elif args.dist == "kv":
        dist = KVDist3D(twiss_x, twiss_y, twiss_z)
    elif args.dist == "gauss":
        dist = GaussDist3D(twiss_x, twiss_y, twiss_z)
    else:
        raise ValueError("Unknown distribution '{}'".format(args.dist))

    for i in range(args.nparts):
        x, xp, y, yp, z, dE = dist.getCoordinates()
        if args.rot:
            phi = math.radians(args.rot)
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            (x, y) = (cos_phi * x + sin_phi * y, -sin_phi * x + cos_phi * y)

        mpi_data_type = orbit_mpi.mpi_datatype.MPI_DOUBLE
        mpi_main_rank = 0
        (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast((x, xp, y, yp, z, dE), mpi_data_type, mpi_main_rank, mpi_comm)
        if i % mpi_size == mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch


def main(args: argparse.Namespace) -> None:

    path = pathlib.Path(__file__)
    output_dir = os.path.join("outputs", path.stem, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    random.seed(23)

    histories = {}

    # Track bunch SCUnifEllipse
    bunch = make_bunch(args)
    lattice = make_lattice(args)
    lattice.trackDesignBunch(bunch)
    if args.sc:
        n_ellipsoids = 5
        path_length_min = 0.010
        sc_calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
        sc_nodes = setUniformEllipsesSCAccNodes(
            lattice, path_length_min, sc_calc
        )

    monitor = BunchMonitor()
    action_container = AccActionsContainer()
    action_container.addAction(monitor, AccActionsContainer.ENTRANCE)
    action_container.addAction(monitor, AccActionsContainer.EXIT)

    params_dict = {"old_pos": -1.0, "count": 0, "pos_step": args.ds}
    lattice.trackBunch(bunch, paramsDict=params_dict, actionContainer=action_container)
    histories["SCUnifEllipse"] = copy.deepcopy(monitor.history)

    # Track bunch SC3D
    bunch = make_bunch(args)
    lattice = make_lattice(args)
    lattice.trackDesignBunch(bunch)
    if args.sc:
        sc_calc = SpaceChargeCalc3D(64, 64, 64)
        path_length_min = 0.010
        sc_nodes = setSC3DAccNodes(lattice, path_length_min, sc_calc)

    monitor = BunchMonitor()
    action_container = AccActionsContainer()
    action_container.addAction(monitor, AccActionsContainer.ENTRANCE)
    action_container.addAction(monitor, AccActionsContainer.EXIT)

    params_dict = {"old_pos": -1.0, "count": 0, "pos_step": args.ds}
    lattice.trackBunch(bunch, paramsDict=params_dict, actionContainer=action_container)
    histories["SC3D"] = copy.deepcopy(monitor.history)

    # Analysis
    # --------------------------------------------------------------------------------

    if mpi_rank == 0:
        for mode in histories:
            for key in histories[mode]:
                histories[mode][key] = np.array(histories[mode][key])

        plot_kws = {}
        plot_kws["SC3D"] = dict(
            color="black",
            lw=None,
            marker=".",
            ms=2
        )
        plot_kws["SCUnifEllipse"] = dict(
            color="red",
            lw=None,
            marker=".",
            ms=1
        )

        fig, axs = plt.subplots(
            nrows=3, figsize=(10, 5), sharex=True, constrained_layout=True
        )
        for mode in histories:
            history = histories[mode]
            for ax, key in zip(axs, ["rms_x", "rms_y", "rms_z"]):
                ax.plot(history["s"], history[key], **plot_kws[mode], label=mode)
        for ax in axs:
            ax.legend(loc="lower right")
        axs[0].set_ylabel("x rms [mm]")
        axs[1].set_ylabel("y rms [mm]")
        axs[2].set_ylabel("z rms [mm]")
        axs[2].set_xlabel("s [m]")
        plt.savefig(os.path.join(output_dir, "fig_history_rms.png"))
        if args.show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main(parse_args())
