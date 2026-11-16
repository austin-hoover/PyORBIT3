"""Analytic, quadrature, integration and physics benchmarks for the 2D solver."""

import math
from pathlib import Path
import runpy

import pytest

np = pytest.importorskip("numpy")

from orbit.core.bunch import Bunch
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse2D


def uniform_disk(a=0.01, b=0.006, angle=0.0, center=(0.0, 0.0)):
    r = np.sqrt((np.arange(32) + 0.5) / 32)
    t = 2 * np.pi * np.arange(64) / 64
    xy = np.column_stack(
        ((a * r[:, None] * np.cos(t)).ravel(), (b * r[:, None] * np.sin(t)).ravel())
    )
    c, s = math.cos(angle), math.sin(angle)
    return xy @ np.array([[c, s], [-s, c]]) + center


def make_bunch(xy, weighted=False):
    bunch = Bunch()
    bunch.getSyncParticle().kinEnergy(0.2)
    bunch.macroSize(1.0e11 / len(xy))
    if weighted:
        bunch.addPartAttr("macrosize")
    for i, (x, y) in enumerate(xy):
        bunch.addParticle(x, 0.001, y, -0.002, (-1)**i / math.sqrt(3), 0.004)
        if weighted:
            bunch.partAttrValue("macrosize", i, 0, bunch.macroSize())
    return bunch


@pytest.mark.parametrize("angle", [0.0, 0.37, -0.8])
def test_uniform_interior_rotation_translation_and_kick(angle):
    a, b, line_density, length = 0.01, 0.006, 5.0e10, 0.2
    center = (0.03, -0.02)
    xy = uniform_disk(a, b, angle, center)
    bunch = make_bunch(xy)
    calculator = SpaceChargeCalcUnifEllipse2D(nEllipses=1)
    calculator.trackBunch(bunch, length)
    assert calculator.getCenter() == pytest.approx(center, abs=1e-15)
    assert calculator.getAngle() == pytest.approx(angle, abs=1e-13)
    assert calculator.getEllipseParameters(0) == pytest.approx((a, b, line_density))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    strengths = np.array([2 * line_density / (a * (a + b)), 2 * line_density / (b * (a + b))])
    fields = ((xy - center) @ rotation * strengths) @ rotation.T
    sync = bunch.getSyncParticle()
    factor = 2 * length * bunch.classicalRadius() / (sync.beta() ** 2 * sync.gamma() ** 3)
    actual = np.array([(bunch.xp(i) - 0.001, bunch.yp(i) + 0.002) for i in range(len(xy))])
    np.testing.assert_allclose(actual, factor * fields, atol=1e-17, rtol=1e-11)
    for i in range(bunch.getSize()):
        assert (bunch.x(i), bunch.y(i)) == tuple(xy[i])
        assert (bunch.z(i), bunch.dE(i)) == ((-1)**i / math.sqrt(3), 0.004)


@pytest.mark.parametrize("a,b", [(0.01, 0.01), (0.01, 0.00999999999), (0.02, 0.001)])
def test_exterior_against_coulomb_quadrature(a, b):
    calculator = SpaceChargeCalcUnifEllipse2D()
    calculator.bunchAnalysis(make_bunch(uniform_disk(a, b)))
    # Independent area integral of (r-r')/|r-r'|^2 over a uniform ellipse.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    radial = np.sqrt((nodes + 1) / 2)
    theta = 2 * np.pi * np.arange(512) / 512
    sources = np.array([a * radial[:, None] * np.cos(theta), b * radial[:, None] * np.sin(theta)])
    for point in [(1.3 * a, 0.0), (0.0, 2 * b), (-1.2 * a, 1.8 * b), (3 * a, -2 * b)]:
        diff = np.array(point)[:, None, None] - sources
        expected = 5e10 * np.sum(
            np.mean(diff / np.sum(diff**2, axis=0), axis=2) * (weights / 2), axis=1
        )
        np.testing.assert_allclose(
            calculator.calculateField(*point), expected, rtol=2e-8, atol=0.02
        )


def test_surface_continuity_circular_limit_and_far_field():
    for a, b in [(0.01, 0.006), (0.01, 0.01), (0.01, 0.009999999999)]:
        calculator = SpaceChargeCalcUnifEllipse2D()
        calculator.bunchAnalysis(make_bunch(uniform_disk(a, b)))
        for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            point = np.array([a * math.cos(theta), b * math.sin(theta)])
            interior = calculator.calculateField(*(point * (1 - 1e-10)))
            exterior = calculator.calculateField(*(point * (1 + 1e-10)))
            np.testing.assert_allclose(interior, exterior, atol=1000, rtol=1e-8)
        point = np.array([1e6, -2e6])
        np.testing.assert_allclose(
            calculator.calculateField(*point),
            5e10 * point / (point @ point),
            rtol=2e-13,
        )
    # A circle has a radial 1/r field outside, including at either axis.
    for point in [(0.02, 0), (0, -0.03), (0.02, 0.03)]:
        p = np.array(point)
        np.testing.assert_allclose(
            calculator.calculateField(*point), 5e10 * p / (p @ p), rtol=1e-10, atol=0.02
        )


@pytest.mark.parametrize("count", [1, 4, 32])
def test_weights_and_zero_weight_probes(count):
    xy = uniform_disk()
    plain, weighted = make_bunch(xy), make_bunch(xy, weighted=True)
    # An uncharged probe must not change the fit or the maximum source radius.
    probe = weighted.addParticle(0.3, 0, -0.2, 0, 0, 0)
    weighted.partAttrValue("macrosize", probe, 0, 0.0)
    left, right = (
        SpaceChargeCalcUnifEllipse2D(count),
        SpaceChargeCalcUnifEllipse2D(count),
    )
    left.bunchAnalysis(plain)
    right.bunchAnalysis(weighted)
    for p in [(0.002, 0.001), (0.03, 0.02)]:
        np.testing.assert_allclose(left.calculateField(*p), right.calculateField(*p), rtol=1e-12)
    assert sum(right.getEllipseParameters(i)[2] for i in range(count)) == pytest.approx(
        5e10, rel=1e-12
    )
    # Splitting each charged macro-particle into two unequal weights preserves
    # the physical density, exercising genuinely nonuniform particle weights.
    split = make_bunch(np.repeat(xy, 2, axis=0), weighted=True)
    for i in range(len(xy)):
        split.z(2*i, plain.z(i))
        split.z(2*i+1, plain.z(i))
        split.partAttrValue("macrosize", 2 * i, 0, plain.macroSize() * 0.25)
        split.partAttrValue("macrosize", 2 * i + 1, 0, plain.macroSize() * 0.75)
    right.bunchAnalysis(split)
    np.testing.assert_allclose(
        left.calculateField(0.03, 0.02), right.calculateField(0.03, 0.02), rtol=1e-12
    )


def test_hollow_distribution_and_radial_charge_conservation():
    theta = 2 * np.pi * np.arange(128) / 128
    xy = np.column_stack((0.01 * np.cos(theta), 0.01 * np.sin(theta)))
    calculator = SpaceChargeCalcUnifEllipse2D(8)
    calculator.bunchAnalysis(make_bunch(xy))
    assert any(calculator.getEllipseParameters(i)[2] < 0 for i in range(8))
    np.testing.assert_allclose(calculator.calculateField(0.001, 0.002), (0, 0), atol=0.1)
    np.testing.assert_allclose(
        calculator.calculateField(0.02, 0), (5e10 / 0.02, 0), rtol=1e-12, atol=0.1
    )


def test_clear_on_empty_zero_charge_and_deleted_particles():
    calculator = SpaceChargeCalcUnifEllipse2D(4)
    bunch = make_bunch(uniform_disk())
    calculator.bunchAnalysis(bunch)
    bunch.macroSize(0.0)
    calculator.bunchAnalysis(bunch)
    assert calculator.calculateField(0.01, 0.02) == (0.0, 0.0)
    bunch.macroSize(1.0)
    bunch.deleteAllParticles()
    calculator.trackBunch(bunch, 0.1)
    assert calculator.calculateField(0.01, 0.02) == (0.0, 0.0)
    bunch = make_bunch(uniform_disk())
    probe = bunch.addParticle(1000, 0, 1000, 0, 0, 0)
    bunch.deleteParticleFast(probe)
    calculator.bunchAnalysis(bunch)
    assert bunch.getSize() == 2048
    assert calculator.getCenter() == pytest.approx((0, 0), abs=1e-15)


def test_weighted_rms_longitudinal_length():
    bunch = make_bunch([(-0.01, -0.006), (0.01, -0.006), (-0.01, 0.006), (0.01, 0.006)], weighted=True)
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    z = np.array([100.0, 101.0, 104.0, 109.0])
    for i in range(4):
        bunch.partAttrValue("macrosize", i, 0, weights[i])
        bunch.z(i, z[i])
    mean = np.average(z, weights=weights)
    expected_length = np.sqrt(12 * np.average((z - mean)**2, weights=weights))
    calculator = SpaceChargeCalcUnifEllipse2D()
    calculator.bunchAnalysis(bunch)
    assert calculator.getEllipseParameters(0)[2] == pytest.approx(weights.sum() / expected_length)


def test_line_density_length_and_relativistic_scaling():
    base = make_bunch(uniform_disk())
    calculator = SpaceChargeCalcUnifEllipse2D()
    for energy, length, bunch_length in [(0.2, 0.1, 2.0), (1.0, -0.2, 4.0)]:
        bunch = Bunch()
        base.copyBunchTo(bunch)
        bunch.getSyncParticle().kinEnergy(energy)
        for i in range(bunch.getSize()):
            bunch.z(i, 10.0 + base.z(i) * bunch_length / 2.0)
        calculator.trackBunch(bunch, length)
        sync = bunch.getSyncParticle()
        field = 2 * (1e11 / bunch_length) * bunch.x(0) / (0.01 * 0.016)
        expected = (
            2 * length * bunch.classicalRadius() * field / (sync.beta() ** 2 * sync.gamma() ** 3)
        )
        assert bunch.xp(0) - 0.001 == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_constructor(bad):
    with pytest.raises(ValueError):
        SpaceChargeCalcUnifEllipse2D(bad)


def test_invalid_inputs_raise_python_exceptions():
    calculator = SpaceChargeCalcUnifEllipse2D()
    bunch = make_bunch(uniform_disk())
    for z in [0.0, float("nan"), float("inf")]:
        invalid = make_bunch(uniform_disk())
        for i in range(invalid.getSize()):
            invalid.z(i, z)
        with pytest.raises(ValueError):
            calculator.bunchAnalysis(invalid)
    with pytest.raises(TypeError):
        calculator.trackBunch(None, 0.1)
    with pytest.raises(IndexError):
        calculator.getEllipseParameters(-1)
    with pytest.raises(IndexError):
        calculator.getEllipseParameters(1)
    with pytest.raises(ValueError):
        calculator.bunchAnalysis(make_bunch([(0, 0), (0, 1), (0, -1)]))
    bunch.macroSize(-1)
    with pytest.raises(ValueError):
        calculator.bunchAnalysis(bunch)


def test_lattice_insertion_and_switch():
    from orbit.teapot import TEAPOT_Lattice, DriftTEAPOT
    from orbit.space_charge.sc2d import setUniformEllipses2DSCAccNodes

    lattice = TEAPOT_Lattice()
    drift = DriftTEAPOT("drift")
    drift.setLength(0.2)
    drift.setnParts(4)
    lattice.addNode(drift)
    lattice.initialize()
    calculator = SpaceChargeCalcUnifEllipse2D()
    nodes = setUniformEllipses2DSCAccNodes(lattice, 0.04, calculator)
    assert sum(node.getLengthOfSC() for node in nodes) == pytest.approx(0.2)
    bunch = make_bunch(uniform_disk())
    before = bunch.xp(0)
    nodes[0].setCalculationOn(False)
    nodes[0].track({"bunch": bunch})
    assert bunch.xp(0) == before
    nodes[0].setCalculationOn(True)
    nodes[0].track({"bunch": bunch})
    assert bunch.xp(0) != before


def test_mpi_partitioned_fit():
    # Run this test alone under mpiexec too. The last rank is empty for any
    # multi-rank run, while the nonempty ranks carry unequal source weights.
    from orbit.core import orbit_mpi

    comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    rank = orbit_mpi.MPI_Comm_rank(comm)
    size = orbit_mpi.MPI_Comm_size(comm)
    xy = uniform_disk(angle=0.4, center=(0.03, -0.02))
    bunch = Bunch()
    bunch.getSyncParticle().kinEnergy(0.2)
    bunch.addPartAttr("macrosize")
    for i, (x, y) in enumerate(xy):
        if i % max(1, size - 1) == rank:
            j = bunch.addParticle(x, 0, y, 0, (-1)**i / math.sqrt(3), 0)
            bunch.partAttrValue("macrosize", j, 0, (1 + (i // 64) % 2) * 1e11 / len(xy))
    calculator = SpaceChargeCalcUnifEllipse2D(16)
    calculator.bunchAnalysis(bunch)
    assert calculator.getCenter() == pytest.approx((0.03, -0.02), abs=1e-14)
    assert calculator.getAngle() == pytest.approx(0.4)
    assert sum(calculator.getEllipseParameters(i)[2] for i in range(16)) == pytest.approx(7.5e10)
    # Independent far-field normalization is unchanged by rank decomposition.
    point = np.array([100.0, -200.0])
    np.testing.assert_allclose(
        calculator.calculateField(*point), 7.5e10 * point / (point @ point), rtol=1e-8
    )
    calculator.trackBunch(bunch, 0.1)
    if size > 1 and rank == size - 1:
        assert bunch.getSize() == 0
    if rank == 0:
        bunch.partAttrValue("macrosize", 0, 0, -1.0)
    with pytest.raises(ValueError):
        calculator.bunchAnalysis(bunch)


@pytest.fixture(scope="module")
def benchmarks():
    pytest.importorskip("scipy")
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/SpaceCharge/scUnifEllipse/benchmark.py"
    )
    return runpy.run_path(str(path))


def test_kv_envelope_benchmark(benchmarks):
    coarse = benchmarks["kv_benchmark"](steps=50)
    fine = benchmarks["kv_benchmark"](steps=100)
    assert fine["max_relative_envelope_error"] < 2e-5
    assert fine["max_relative_envelope_error"] < coarse["max_relative_envelope_error"] / 3.5


def test_elliptical_gaussian_pic_benchmark(benchmarks):
    results = benchmarks["pic_benchmark"]()["results"]
    assert results[-1]["relative_rms_field_error"] < 0.015
    assert results[-1]["relative_rms_kick_error"] < 0.015
    assert results[-1]["relative_rms_field_error"] < results[0]["relative_rms_field_error"] / 10
