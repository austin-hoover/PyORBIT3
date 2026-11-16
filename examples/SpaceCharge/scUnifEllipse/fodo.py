"""Compare uniform ellipses with free-space 2D FFT PIC in a repeated FODO cell.

Run from the repository root:
    python examples/SpaceCharge/scUnifEllipse/fodo.py --turns 20
    python examples/SpaceCharge/scUnifEllipse/fodo.py --distribution kv --ellipses 1

A turn is one traversal of the cell. Both bunches start from identical particles,
matched to the cell without space charge. With space charge enabled the initial
beam is mismatched and breathes. The PIC reference uses the transverse projected
density and sqrt(12)*rms(z), just like the ellipse model. NumPy is required.

MPI and independent timing runs:
    mpiexec -n 4 python examples/SpaceCharge/scUnifEllipse/fodo.py --turns 20
    python examples/SpaceCharge/scUnifEllipse/fodo.py --solver ellipse
    python examples/SpaceCharge/scUnifEllipse/fodo.py --solver pic

--particles is the global count; the same seeded sample is partitioned across
ranks. Each solver completes all turns before the next starts. Timing includes
tracking and its MPI collectives, excludes setup, diagnostics and printing, and
reports the maximum cumulative time across ranks. PIC uses Python particle
loops with C++ grid/FFT operations, while the ellipse kick runs in C++; timings
therefore compare these implementations, including the Python overhead.
"""

import argparse
import math
from time import perf_counter

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch, BunchTwissAnalysis
from orbit.core.spacecharge import Grid2D, PoissonSolverFFT2D, SpaceChargeCalcUnifEllipse2D
from orbit.space_charge.sc2d import setUniformEllipses2DSCAccNodes
from orbit.teapot import DriftTEAPOT, QuadTEAPOT, TEAPOT_Lattice


COMM = orbit_mpi.mpi_comm.MPI_COMM_WORLD
RANK = orbit_mpi.MPI_Comm_rank(COMM)
SIZE = orbit_mpi.MPI_Comm_size(COMM)
SUM = orbit_mpi.mpi_op.MPI_SUM
MAX = orbit_mpi.mpi_op.MPI_MAX


def reduce(value, op=SUM):
    # The MPI wrapper accepts scalars and tuples, not NumPy arrays/scalars.
    return orbit_mpi.MPI_Allreduce(value, orbit_mpi.mpi_datatype.MPI_DOUBLE, op, COMM)


def make_lattice(parts):
    lattice = TEAPOT_Lattice("FODO")
    for name, length, strength in [
        ("D1", 0.2, 0), ("QF", 0.1, 8), ("D2", 0.4, 0),
        ("QD", 0.1, -8), ("D3", 0.2, 0),
    ]:
        node = QuadTEAPOT(name) if strength else DriftTEAPOT(name)
        node.setLength(length)
        node.setnParts(parts)
        if strength:
            node.setParam("kq", strength)
        lattice.addNode(node)
    lattice.initialize()
    return lattice


def matched_twiss(sign):
    matrix = np.eye(2)
    for length, strength in [(0.2, 0), (0.1, 8*sign), (0.4, 0), (0.1, -8*sign), (0.2, 0)]:
        if strength == 0:
            element = np.array([[1, length], [0, 1]])
        else:
            k = math.sqrt(abs(strength))
            c = math.cos(k*length) if strength > 0 else math.cosh(k*length)
            s = math.sin(k*length) if strength > 0 else math.sinh(k*length)
            element = np.array([[c, s/k], [-math.copysign(k, strength)*s, c]])
        matrix = element @ matrix
    cosine = np.trace(matrix)/2
    if abs(cosine) >= 1:
        raise ValueError("FODO cell is unstable")
    sine = math.copysign(math.sqrt(1-cosine*cosine), matrix[0, 1])
    return matrix[0, 1]/sine, (matrix[0, 0]-matrix[1, 1])/(2*sine)


def make_bunch(args):
    rng = np.random.default_rng(args.seed)
    coordinates = rng.normal(size=(args.particles//2, 4))
    if args.distribution == "kv":
        coordinates *= 2/np.linalg.norm(coordinates, axis=1)[:, None]
    coordinates = np.concatenate((coordinates, -coordinates))
    emit = 2e-6  # geometric rms emittance [m rad]
    for offset, sign in [(0, 1), (2, -1)]:
        beta, alpha = matched_twiss(sign)
        u, v = coordinates[:, offset].copy(), coordinates[:, offset+1].copy()
        coordinates[:, offset] = math.sqrt(emit*beta)*u
        coordinates[:, offset+1] = math.sqrt(emit/beta)*(v-alpha*u)
    # Uniform longitudinal sample, rescaled to exactly 2 m rms-equivalent length.
    z = rng.uniform(-1, 1, len(coordinates))
    z -= z.mean()
    z /= math.sqrt(3)*z.std()
    bunch = Bunch()
    bunch.mass(0.938272)
    bunch.charge(1)
    bunch.getSyncParticle().kinEnergy(0.2)
    bunch.macroSize(args.intensity/len(coordinates))
    for row, zi in zip(coordinates[RANK::SIZE], z[RANK::SIZE]):
        bunch.addParticle(*row, zi, 0)
    return bunch


class TransversePIC:
    """Free-space PIC with the same projected-density model as the ellipses."""

    def __init__(self, size):
        self.size = size
        self.solver = PoissonSolverFFT2D(size, size, -1, 1, -1, 1)
        self.rho = Grid2D(size, size)
        self.phi = Grid2D(size, size)

    def trackBunch(self, bunch, length):
        xyz = np.array([(bunch.x(i), bunch.y(i), bunch.z(i)) for i in range(bunch.getSize())]).reshape(-1, 3)
        # A square mesh avoids repeated changes to the FFT Green function's
        # aspect ratio. Leave margin so every particle has a full stencil.
        moments = reduce((float(len(xyz)), *map(float, xyz.sum(axis=0))))
        count = moments[0]
        cx, cy, cz = np.array(moments[1:])/count
        local_radius = float(np.max(abs(xyz[:, :2]-[cx, cy]))) if len(xyz) else 0.0
        radius = 1.2*reduce(local_radius, MAX)
        for grid in (self.rho, self.phi):
            grid.setGridX(cx-radius, cx+radius)
            grid.setGridY(cy-radius, cy+radius)
        self.rho.setZero()
        self.rho.binBunch(bunch)
        self.rho.synchronizeMPI(COMM)
        self.solver.findPotential(self.rho, self.phi)
        sync = bunch.getSyncParticle()
        beam_length = math.sqrt(12*reduce(float(np.sum((xyz[:, 2]-cz)**2)))/count)
        factor = -2*length*bunch.classicalRadius()/(sync.beta()**2*sync.gamma()**3*beam_length)
        for i, (x, y, _) in enumerate(xyz):
            gx, gy = self.phi.calcGradient(x, y)
            bunch.xp(i, bunch.xp(i)+factor*gx)
            bunch.yp(i, bunch.yp(i)+factor*gy)


def statistics(bunch):
    analysis = BunchTwissAnalysis()
    analysis.analyzeBunch(bunch)
    return np.array([math.sqrt(analysis.getCorrelation(0, 0)),
                     math.sqrt(analysis.getCorrelation(2, 2)),
                     analysis.getEmittance(0), analysis.getEmittance(1)])


def run_solver(label, initial, args):
    bunch = Bunch()
    initial.copyBunchTo(bunch)
    lattice = make_lattice(args.parts)
    calculator = SpaceChargeCalcUnifEllipse2D(args.ellipses) if label == "ellipse" else TransversePIC(args.grid)
    nodes = setUniformEllipses2DSCAccNodes(lattice, 0, calculator)
    if RANK == 0:
        print(f"\n# {label}: kicks/turn={len(nodes)}", flush=True)
        print("turn    rms_x[mm] rms_y[mm] emit_x[mm mrad] emit_y[mm mrad] tracking_s", flush=True)
    history = []
    local_total = 0.0
    for turn in range(args.turns+1):
        elapsed = 0.0
        if turn:
            orbit_mpi.MPI_Barrier(COMM)
            start = perf_counter()
            lattice.trackBunch(bunch)
            elapsed = perf_counter()-start
            local_total += elapsed
        turn_seconds = reduce(float(elapsed), MAX)
        values = statistics(bunch)
        history.append(values)
        if RANK == 0:
            scaled = values*np.array([1e3, 1e3, 1e6, 1e6])
            print(f"{turn:4d} " + " ".join(f"{v:13.6f}" for v in scaled) + f" {turn_seconds:10.6f}", flush=True)
    total = reduce(float(local_total), MAX)
    if RANK == 0:
        per_turn = total/args.turns if args.turns else 0.0
        print(f"# {label}: tracking total={total:.6f} s, mean/turn={per_turn:.6f} s", flush=True)
    return np.array(history), total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--particles", type=int, default=10000, help="global even number of macroparticles")
    parser.add_argument("--ellipses", type=int, default=16)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--parts", type=int, default=8, help="integration parts per lattice element")
    parser.add_argument("--intensity", type=float, default=5e11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distribution", choices=["gaussian", "kv"], default="gaussian")
    parser.add_argument("--solver", choices=["both", "ellipse", "pic"], default="both")
    args = parser.parse_args()
    if args.particles < 4 or args.particles % 2 or args.ellipses < 1 or args.parts < 2 or args.grid < 8 or args.turns < 0 or args.intensity <= 0:
        parser.error("use an even particle count >=4, positive ellipses/intensity, parts >=2, grid >=8 and turns >=0")
    initial = make_bunch(args)
    if RANK == 0:
        print(f"# {args.distribution}, global N={args.particles}, MPI ranks={SIZE}, ellipses={args.ellipses}, PIC={args.grid}x{args.grid}")
        print("# Timings exclude setup, diagnostics and printing; total is maximum across ranks.")
        print("# PIC includes Python particle loops; ellipse kicks execute in C++.", flush=True)
    labels = ("ellipse", "pic") if args.solver == "both" else (args.solver,)
    results = {label: run_solver(label, initial, args) for label in labels}
    if RANK == 0 and args.solver == "both":
        left, ellipse_seconds = results["ellipse"]
        right, pic_seconds = results["pic"]
        print("\n# Differences: (ellipse/PIC-1)*100")
        print("turn    dx[%]    dy[%]   dex[%]   dey[%]")
        for turn, differences in enumerate(100*(left/right-1)):
            print(f"{turn:4d} " + " ".join(f"{v:8.3f}" for v in differences))
        if ellipse_seconds > 0:
            print(f"# PIC/ellipse tracking time ratio: {pic_seconds/ellipse_seconds:.3f}")


if __name__ == "__main__":
    main()
