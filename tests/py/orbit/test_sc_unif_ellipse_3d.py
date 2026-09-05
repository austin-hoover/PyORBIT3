import math

import pytest

from orbit.core.bunch import Bunch
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse
from orbit.core.spacecharge import UniformEllipsoidFieldCalculator


def _rotation_matrix(rx=0.31, ry=-0.43, rz=0.57):
    """Return Rz(rz) Ry(ry) Rx(rx); columns are the principal axes."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def _rotate(rotation, vector):
    return tuple(sum(rotation[i][j] * vector[j] for j in range(3)) for i in range(3))


def _make_bunch(principal_points, weights, probe_points, rotation, center, kinetic_energy=1.0, use_particle_weights=True):
    bunch = Bunch()
    bunch.mass(0.93827231)
    bunch.charge(1.0)
    bunch.getSyncParticle().kinEnergy(kinetic_energy)
    gamma = bunch.getSyncParticle().gamma()
    if use_particle_weights:
        bunch.addPartAttr("macrosize")

    for point, weight in zip(principal_points, weights):
        x_rest, y_rest, z_rest = _rotate(rotation, point)
        index = bunch.addParticle(
            center[0] + x_rest,
            0.0,
            center[1] + y_rest,
            0.0,
            center[2] + z_rest / gamma,
            0.0,
        )
        if use_particle_weights:
            bunch.partAttrValue("macrosize", index, 0, weight)

    if not use_particle_weights:
        assert all(weight == weights[0] for weight in weights)
        bunch.macroSize(weights[0])

    probe_indices = []
    for point in probe_points:
        x_rest, y_rest, z_rest = _rotate(rotation, point)
        index = bunch.addParticle(
            center[0] + x_rest,
            0.0,
            center[1] + y_rest,
            0.0,
            center[2] + z_rest / gamma,
            0.0,
        )
        if use_particle_weights:
            bunch.partAttrValue("macrosize", index, 0, 0.0)
        probe_indices.append(index)
    return bunch, probe_indices


def _relative_vector_error(actual, expected):
    difference = math.sqrt(sum((x - y) ** 2 for x, y in zip(actual, expected)))
    scale = math.sqrt(sum(value**2 for value in expected))
    return difference / scale


@pytest.mark.parametrize("rotation", [_rotation_matrix(0.0, 0.0, 0.0), _rotation_matrix()])
@pytest.mark.parametrize("use_particle_weights", [False, True])
def test_oriented_ellipsoid_matches_principal_axis_field(rotation, use_particle_weights):
    axes = (0.0060, 0.0035, 0.0020)
    total_macrosize = 7.5e9
    center = (0.0013, -0.0007, 0.0021)

    # These six weighted points have exactly <u_i^2> = axes[i]^2 / 5.
    principal_points = []
    scale = math.sqrt(3.0 / 5.0)
    for axis_index, axis in enumerate(axes):
        point = [0.0, 0.0, 0.0]
        point[axis_index] = scale * axis
        principal_points.extend((tuple(point), tuple(-value for value in point)))
    weights = [total_macrosize / len(principal_points)] * len(principal_points)
    probe_points = (
        (0.20 * axes[0], -0.15 * axes[1], 0.10 * axes[2]),
        (-0.30 * axes[0], 0.10 * axes[1], 0.20 * axes[2]),
        (0.15 * axes[0], 0.25 * axes[1], -0.20 * axes[2]),
    )
    bunch, _ = _make_bunch(principal_points, weights, (), rotation, center, use_particle_weights=use_particle_weights)

    calculator = SpaceChargeCalcUnifEllipse(1)
    calculator.trackBunch(bunch, 0.0)
    reference = UniformEllipsoidFieldCalculator()
    reference.setEllipsoid(*axes, 10.0 * max(axes))

    for point in probe_points:
        reference_principal = reference.calcField(*point)
        expected = _rotate(rotation, tuple(total_macrosize * value for value in reference_principal))
        rest_point = _rotate(rotation, point)
        actual = calculator.calculateField(*rest_point)
        assert actual == pytest.approx(expected, rel=2.0e-12)


def _uniform_ellipsoid_grid(axes, grid_size=31):
    points = []
    for ix in range(grid_size):
        ux = -1.0 + 2.0 * ix / (grid_size - 1)
        for iy in range(grid_size):
            uy = -1.0 + 2.0 * iy / (grid_size - 1)
            for iz in range(grid_size):
                uz = -1.0 + 2.0 * iz / (grid_size - 1)
                if ux * ux + uy * uy + uz * uz <= 1.0:
                    points.append((axes[0] * ux, axes[1] * uy, axes[2] * uz))
    return points


def test_nested_ellipsoid_shells_rotate_with_distribution():
    axes = (0.0060, 0.0040, 0.0025)
    principal_points = _uniform_ellipsoid_grid(axes, grid_size=13)
    weights = []
    for x, y, z in principal_points:
        radius_squared = (x / axes[0]) ** 2 + (y / axes[1]) ** 2 + (z / axes[2]) ** 2
        weights.append(1.0e7 * (1.0 + 2.0 * radius_squared))
    probe_points = (
        (0.20 * axes[0], 0.10 * axes[1], -0.10 * axes[2]),
        (-0.25 * axes[0], 0.15 * axes[1], 0.20 * axes[2]),
    )
    identity = _rotation_matrix(0.0, 0.0, 0.0)
    rotation = _rotation_matrix(rx=0.29, ry=-0.37, rz=0.51)
    reference_bunch, _ = _make_bunch(principal_points, weights, probe_points, identity, (0.0, 0.0, 0.0))
    rotated_bunch, _ = _make_bunch(principal_points, weights, probe_points, rotation, (0.001, -0.002, 0.003))

    reference = SpaceChargeCalcUnifEllipse(4)
    rotated = SpaceChargeCalcUnifEllipse(4)
    reference.trackBunch(reference_bunch, 0.0)
    rotated.trackBunch(rotated_bunch, 0.0)

    for point in probe_points:
        expected = _rotate(rotation, reference.calculateField(*point))
        actual = rotated.calculateField(*_rotate(rotation, point))
        assert actual == pytest.approx(expected, rel=2.0e-12)


def test_arbitrarily_oriented_ellipsoid_agrees_with_pic():
    axes = (0.0060, 0.0040, 0.0025)
    total_macrosize = 2.0e10
    rotation = _rotation_matrix(rx=0.29, ry=-0.37, rz=0.51)
    center = (0.0, 0.0, 0.0)
    principal_points = _uniform_ellipsoid_grid(axes)
    weights = [total_macrosize / len(principal_points)] * len(principal_points)
    probe_points = (
        (0.20 * axes[0], 0.10 * axes[1], -0.10 * axes[2]),
        (-0.25 * axes[0], 0.15 * axes[1], 0.20 * axes[2]),
        (0.10 * axes[0], -0.25 * axes[1], 0.15 * axes[2]),
    )
    ellipse_bunch, _ = _make_bunch(principal_points, weights, probe_points, rotation, center)
    pic_bunch, probe_indices = _make_bunch(principal_points, weights, probe_points, rotation, center)

    length = 1.0
    ellipse = SpaceChargeCalcUnifEllipse(1)
    ellipse.trackBunch(ellipse_bunch, length)
    pic = SpaceChargeCalc3D(64, 64, 64)
    pic.trackBunch(pic_bunch, length)

    sync_part = pic_bunch.getSyncParticle()
    transverse_factor = length * pic_bunch.classicalRadius() / (sync_part.beta() ** 2 * sync_part.gamma() ** 2)
    longitudinal_factor = length * pic_bunch.classicalRadius() * pic_bunch.mass()
    errors = []
    for point, index in zip(probe_points, probe_indices):
        rest_point = _rotate(rotation, point)
        ellipse_field = ellipse.calculateField(*rest_point)
        pic_field = (
            pic_bunch.xp(index) / transverse_factor,
            pic_bunch.yp(index) / transverse_factor,
            pic_bunch.dE(index) / longitudinal_factor,
        )
        errors.append(_relative_vector_error(pic_field, ellipse_field))

    # The finite particle sample and PIC mesh give about 2.1% error here.
    assert max(errors) < 0.04
