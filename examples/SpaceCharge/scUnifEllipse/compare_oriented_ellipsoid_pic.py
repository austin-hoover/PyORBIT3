"""Compare a rotated uniform-ellipsoid model with the 3D PIC solver."""

import argparse
import math

from orbit.core.bunch import Bunch
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse


def rotation_matrix(rx, ry, rz):
    """Return Rz(rz) Ry(ry) Rx(rx); columns are the principal axes."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def rotate(rotation, vector):
    return tuple(sum(rotation[i][j] * vector[j] for j in range(3)) for i in range(3))


def uniform_ellipsoid_grid(axes, grid_size):
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


def make_bunch(principal_points, probe_points, rotation, total_macrosize, kinetic_energy):
    bunch = Bunch()
    bunch.mass(0.93827231)
    bunch.charge(1.0)
    bunch.getSyncParticle().kinEnergy(kinetic_energy)
    gamma = bunch.getSyncParticle().gamma()
    bunch.addPartAttr("macrosize")

    weight = total_macrosize / len(principal_points)
    for point in principal_points:
        x_rest, y_rest, z_rest = rotate(rotation, point)
        index = bunch.addParticle(x_rest, 0.0, y_rest, 0.0, z_rest / gamma, 0.0)
        bunch.partAttrValue("macrosize", index, 0, weight)

    probe_indices = []
    for point in probe_points:
        x_rest, y_rest, z_rest = rotate(rotation, point)
        index = bunch.addParticle(x_rest, 0.0, y_rest, 0.0, z_rest / gamma, 0.0)
        bunch.partAttrValue("macrosize", index, 0, 0.0)
        probe_indices.append(index)
    return bunch, probe_indices


def relative_vector_error(actual, expected):
    difference = math.sqrt(sum((x - y) ** 2 for x, y in zip(actual, expected)))
    scale = math.sqrt(sum(value**2 for value in expected))
    return difference / scale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-grid", type=int, default=31, help="points per axis used to sample the uniform density")
    parser.add_argument("--pic-grid", type=int, default=64, help="PIC cells per axis")
    return parser.parse_args()


def main():
    args = parse_args()
    axes = (0.0060, 0.0040, 0.0025)
    total_macrosize = 2.0e10
    kinetic_energy = 1.0
    rotation = rotation_matrix(rx=0.29, ry=-0.37, rz=0.51)
    principal_points = uniform_ellipsoid_grid(axes, args.sample_grid)
    probe_points = (
        (0.20 * axes[0], 0.10 * axes[1], -0.10 * axes[2]),
        (-0.25 * axes[0], 0.15 * axes[1], 0.20 * axes[2]),
        (0.10 * axes[0], -0.25 * axes[1], 0.15 * axes[2]),
    )
    ellipse_bunch, _ = make_bunch(principal_points, probe_points, rotation, total_macrosize, kinetic_energy)
    pic_bunch, probe_indices = make_bunch(principal_points, probe_points, rotation, total_macrosize, kinetic_energy)

    ellipse = SpaceChargeCalcUnifEllipse(1)
    ellipse.trackBunch(ellipse_bunch, 0.0)
    pic = SpaceChargeCalc3D(args.pic_grid, args.pic_grid, args.pic_grid)
    pic.trackBunch(pic_bunch, 1.0)

    sync_part = pic_bunch.getSyncParticle()
    transverse_factor = pic_bunch.classicalRadius() / (sync_part.beta() ** 2 * sync_part.gamma() ** 2)
    longitudinal_factor = pic_bunch.classicalRadius() * pic_bunch.mass()
    print(f"sample particles: {len(principal_points)}, PIC grid: {args.pic_grid}^3")
    print("probe     relative field error       analytic field (Ex, Ey, Ez)       PIC field (Ex, Ey, Ez)")
    for probe_number, (point, index) in enumerate(zip(probe_points, probe_indices), start=1):
        ellipse_field = ellipse.calculateField(*rotate(rotation, point))
        pic_field = (
            pic_bunch.xp(index) / transverse_factor,
            pic_bunch.yp(index) / transverse_factor,
            pic_bunch.dE(index) / longitudinal_factor,
        )
        error = relative_vector_error(pic_field, ellipse_field)
        analytic_text = " ".join(f"{value: .5e}" for value in ellipse_field)
        pic_text = " ".join(f"{value: .5e}" for value in pic_field)
        print(f"{probe_number:5d} {error:23.3%}   ({analytic_text})   ({pic_text})")


if __name__ == "__main__":
    main()
