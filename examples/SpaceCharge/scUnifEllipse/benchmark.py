"""Benchmark 2D uniform ellipse space charge solver.

One ellipse: track KV distribution through drift and compare full envelope
history with KV equations.

Multiple ellipses: compare fields and kicks with free-space FFT PIC for a
truncated elliptical Gaussian.
"""

import json
import math

import numpy as np
import scipy.integrate

from orbit.core.bunch import Bunch
from orbit.core.spacecharge import Grid2D
from orbit.core.spacecharge import PoissonSolverFFT2D
from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse2D


def make_bunch(particles: np.ndarray, intensity: float = 2.0e12) -> Bunch:
    bunch = Bunch()
    bunch.mass(0.938272)
    bunch.charge(1.0)
    bunch.getSyncParticle().kinEnergy(0.2)
    bunch.macroSize(intensity / len(particles))
    for i, particle in enumerate(particles):
        particle = particle.copy()
        particle[4] = (-1)**i / math.sqrt(3)
        bunch.addParticle(*particle)
    return bunch


def kv_benchmark(steps: int = 200) -> dict:
    # A quiet quadrature on the 4D KV surface. Its projection is a uniform
    # ellipse and the edge emittances below are four times the rms emittances.
    a0, b0, emit_x, emit_y = 0.010, 0.006, 8.0e-6, 5.0e-6
    u, phi, psi = np.meshgrid(
        (np.arange(12) + 0.5) / 12,
        2 * np.pi * np.arange(12) / 12,
        2 * np.pi * np.arange(12) / 12,
        indexing="ij",
    )
    particles = np.zeros((u.size, 6))
    particles[:, 0] = (a0 * np.sqrt(u) * np.cos(phi)).ravel()
    particles[:, 1] = (emit_x / a0 * np.sqrt(u) * np.sin(phi)).ravel()
    particles[:, 2] = (b0 * np.sqrt(1 - u) * np.cos(psi)).ravel()
    particles[:, 3] = (emit_y / b0 * np.sqrt(1 - u) * np.sin(psi)).ravel()
    bunch = make_bunch(particles)
    bunch_length = math.sqrt(12) * np.std([bunch.z(i) for i in range(bunch.getSize())])
    sync = bunch.getSyncParticle()
    perveance = (
        2 * bunch.classicalRadius() * 2.0e12 / (bunch_length * sync.beta() ** 2 * sync.gamma() ** 3)
    )

    def rhs(s, state):
        a, ap, b, bp = state
        return [
            ap,
            emit_x**2 / a**3 + 2 * perveance / (a + b),
            bp,
            emit_y**2 / b**3 + 2 * perveance / (a + b),
        ]

    positions = np.linspace(0.0, 2.0, steps + 1)
    exact = scipy.integrate.solve_ivp(
        rhs,
        (0.0, positions[-1]),
        [a0, 0.0, b0, 0.0],
        t_eval=positions,
        rtol=2e-12,
        atol=2e-14,
    )
    if not exact.success:
        raise RuntimeError(exact.message)
    calculator = SpaceChargeCalcUnifEllipse2D(1)
    history = [[a0, b0]]
    ds = positions[1]

    def drift(length):
        # Paraxial drift, the same independent-variable convention as KV.
        for i in range(bunch.getSize()):
            bunch.x(i, bunch.x(i) + length * bunch.xp(i))
            bunch.y(i, bunch.y(i) + length * bunch.yp(i))

    for _ in range(steps):
        drift(0.5 * ds)
        calculator.trackBunch(bunch, ds)
        drift(0.5 * ds)
        history.append(
            [
                2
                * math.sqrt(sum(bunch.x(i) ** 2 for i in range(bunch.getSize())) / bunch.getSize()),
                2
                * math.sqrt(sum(bunch.y(i) ** 2 for i in range(bunch.getSize())) / bunch.getSize()),
            ]
        )
    history = np.asarray(history)
    reference = exact.y[[0, 2]].T
    return {
        "steps": steps,
        "max_relative_envelope_error": float(np.max(abs(history / reference - 1))),
        "final_tracked_axes_m": history[-1].tolist(),
        "final_kv_axes_m": reference[-1].tolist(),
    }


def pic_benchmark(ellipse_counts: list[int] = [1, 4, 16, 64], grid_size: int = 256) -> dict:
    # Inverse-CDF radial quadrature of a Gaussian truncated at 4 sigma,
    # with a full angular quadrature on each ring. Equal particle weights.
    sigma_x, sigma_y = 0.004, 0.002
    r = np.sqrt(-2 * np.log(1 - (np.arange(256) + 0.5) / 256 * (1 - np.exp(-8))))
    theta = 2 * np.pi * np.arange(256) / 256
    particles = np.zeros((r.size * theta.size, 6))
    particles[:, 0] = (sigma_x * r[:, None] * np.cos(theta)).ravel()
    particles[:, 2] = (sigma_y * r[:, None] * np.sin(theta)).ravel()
    bunch = make_bunch(particles, intensity=1.0e11)
    bunch_length = math.sqrt(12) * np.std([bunch.z(i) for i in range(bunch.getSize())])
    kick_length = 0.1
    # The 2D PIC Green function is -log(r), so -grad(phi)/bunch_length
    # has exactly the field normalization of the ellipse calculator.
    extent_x, extent_y = 6 * sigma_x, 6 * sigma_y
    args = (grid_size, grid_size, -extent_x, extent_x, -extent_y, extent_y)
    rho, potential = Grid2D(*args), Grid2D(*args)
    rho.binBunch(bunch)
    PoissonSolverFFT2D(*args).findPotential(rho, potential)
    # Probe both the core and tails, with several off-axis angles. A global
    # vector norm avoids ill-conditioned component errors on symmetry axes.
    points = np.array(
        [
            (sigma_x * radius * math.cos(t), sigma_y * radius * math.sin(t))
            for radius in np.linspace(0.2, 4.8, 32)
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ]
    )
    pic = -np.array([potential.calcGradient(*p) for p in points]) / bunch_length
    sync = bunch.getSyncParticle()
    factor = 2 * kick_length * bunch.classicalRadius() / (sync.beta() ** 2 * sync.gamma() ** 3)
    # Compare kicks on actual particles as well as fields at external probes.
    indices = np.arange(0, bunch.getSize(), 97)
    pic_kicks = (
        -factor
        * np.array([potential.calcGradient(bunch.x(int(i)), bunch.y(int(i))) for i in indices])
        / bunch_length
    )
    results = []
    for count in ellipse_counts:
        calculator = SpaceChargeCalcUnifEllipse2D(count)
        for i in range(bunch.getSize()):
            bunch.xp(i, 0.0)
            bunch.yp(i, 0.0)
        calculator.trackBunch(bunch, kick_length)
        fields = np.array([calculator.calculateField(*p) for p in points])
        kicks = np.array([(bunch.xp(int(i)), bunch.yp(int(i))) for i in indices])
        results.append(
            {
                "ellipses": count,
                "relative_rms_field_error": float(
                    np.linalg.norm(fields - pic) / np.linalg.norm(pic)
                ),
                "relative_rms_kick_error": float(
                    np.linalg.norm(kicks - pic_kicks) / np.linalg.norm(pic_kicks)
                ),
            }
        )
    return {"grid_size": grid_size, "particles": bunch.getSize(), "results": results}


if __name__ == "__main__":
    results = {}
    results["kv"] = kv_benchmark()
    results["pic"] = pic_benchmark()
    print(json.dumps(results, indent=2))
