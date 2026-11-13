"""Analytic matrices accepting Envelope or SyncParticleParams reference state.

Transport maps advance the supplied reference time and kinetic energy in place.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np

from orbit.utils.consts import speed_of_light

from .sync_particle import SyncParticleParams

if TYPE_CHECKING:
    from .envelope import Envelope


def get_dp_p_coeff(sync_params: Envelope | SyncParticleParams) -> float:
    # dE/E = (beta^2) * dp/p
    # dE = (beta^2 * E) * dp/p
    # dE = (beta^2 * gamma * m * c^2) * dp/p
    beta = sync_params.beta
    gamma = sync_params.gamma
    rest_energy = sync_params.mass  # GeV
    return 1.0 / (beta**2 * gamma * rest_energy)


def get_zp_coeff(sync_params: Envelope | SyncParticleParams) -> float:
    # dE/E = (beta^2) * dp/p = (beta^2) * (gamma^2) z'
    # dE = (beta^2 * gamma^2 * E) * z'
    # dE = (beta^2 * gamma^3 * m * c^2) * z'
    beta = sync_params.beta
    gamma = sync_params.gamma
    rest_energy = sync_params.mass
    return 1.0 / (beta**2 * gamma**3 * rest_energy)


def convert_matrix_dp_p_to_dE(matrix: np.ndarray, sync_params: Envelope | SyncParticleParams) -> np.ndarray:
    # v = [x, x', y, y', z, dp/p]
    # w = [x, x', y, y', z, dE]
    # v = A w
    # v -> M v
    # w -> A M A^-1
    dp_p_coeff = get_dp_p_coeff(sync_params)
    matrix[:5, 5] *= dp_p_coeff
    matrix[5, :5] /= dp_p_coeff
    matrix[5, 6] /= dp_p_coeff  # driving term
    return matrix


def convert_matrix_zp_to_dE(matrix: np.ndarray, sync_params: Envelope | SyncParticleParams) -> np.ndarray:
    zp_coeff = get_zp_coeff(sync_params)
    matrix[:5, 5] *= zp_coeff
    matrix[5, :5] /= zp_coeff
    matrix[5, 6] /= zp_coeff  # driving term
    return matrix


def get_matrix_tilt(angle: float) -> np.ndarray:
    cos_phi = math.cos(angle)
    sin_phi = math.sin(angle)

    M = np.identity(7)
    M[0, 0] = M[1, 1] = +cos_phi
    M[0, 2] = M[1, 3] = -sin_phi
    M[2, 0] = M[3, 1] = +sin_phi
    M[2, 2] = M[3, 3] = +cos_phi
    return M


def get_matrix_kick(kx: float = 0.0, ky: float = 0.0, kE: float = 0.0) -> np.ndarray:
    M = np.identity(7)
    M[1, -1] = kx
    M[3, -1] = ky
    M[5, -1] = kE
    return M


def get_matrix_drift(sync_params: Envelope | SyncParticleParams, length: float) -> np.ndarray:
    M = np.identity(7)
    M[0, 1] = length
    M[2, 3] = length
    M[4, 5] = length / (sync_params.gamma ** 2)
    M[4, 5] *= get_dp_p_coeff(sync_params)  # convert_matrix_dp_p_to_dE(M, sync_params)

    sync_params.time += length / (sync_params.beta * speed_of_light)
    return M


def get_matrix_quad(sync_params: Envelope | SyncParticleParams, length: float, kq: float) -> np.ndarray:
    if abs(kq) == 0:
        return get_matrix_drift(sync_params, length)

    sqrt_abs_kq = math.sqrt(abs(kq))

    M = np.identity(7)
    if kq > 0:
        cx = np.cos(sqrt_abs_kq * length)
        sx = np.sin(sqrt_abs_kq * length)
        cy = np.cosh(sqrt_abs_kq * length)
        sy = np.sinh(sqrt_abs_kq * length)
        M[0, 0] = cx
        M[0, 1] = +sx / sqrt_abs_kq
        M[1, 0] = -sx * sqrt_abs_kq
        M[1, 1] = cx
        M[2, 2] = cy
        M[2, 3] = sy / sqrt_abs_kq
        M[3, 2] = sy * sqrt_abs_kq
        M[3, 3] = cy
    elif kq < 0:
        cx = np.cosh(sqrt_abs_kq * length)
        sx = np.sinh(sqrt_abs_kq * length)
        cy = np.cos(sqrt_abs_kq * length)
        sy = np.sin(sqrt_abs_kq * length)
        M[0, 0] = cx
        M[0, 1] = sx / sqrt_abs_kq
        M[1, 0] = sx * sqrt_abs_kq
        M[1, 1] = cx
        M[2, 2] = cy
        M[2, 3] = +sy / sqrt_abs_kq
        M[3, 2] = -sy * sqrt_abs_kq
        M[3, 3] = cy

    M[4, 5] = length / (sync_params.gamma**2)
    M[4, 5] *= get_dp_p_coeff(sync_params)  # convert_matrix_dp_p_to_dE(M, sync_params)

    sync_params.time += length / (sync_params.beta * speed_of_light)
    return M


def get_matrix_bend(sync_params: Envelope | SyncParticleParams, length: float, theta: float) -> np.ndarray:
    rho = length / theta
    cx = math.cos(theta)
    sx = math.sin(theta)

    M = np.identity(7)
    M[0, 0] = cx
    M[0, 1] = rho * sx
    M[0, 5] = rho * (1.0 - cx)
    M[1, 0] = -sx / rho
    M[1, 1] = cx
    M[1, 5] = sx
    M[2, 3] = length
    M[4, 0] = -sx
    M[4, 1] = -rho * (1.0 - cx)
    M[4, 5] = -(sync_params.beta ** 2) * length + rho * sx
    M[:5, 5] *= get_dp_p_coeff(sync_params)  # convert_matrix_dp_p_to_dE(M, sync_params)

    sync_params.time += length / (sync_params.beta * speed_of_light)
    return M


def get_matrix_solenoid(sync_params: Envelope | SyncParticleParams, length: float, B: float) -> np.ndarray:
    if B == 0:
        return get_matrix_drift(sync_params, length)

    phase = B * length

    V = np.identity(7)
    V[:4, :4] = 0.0
    V[0, 1] = -1.0 / B
    V[0, 2] = 0.5
    V[1, 0] = 0.5 * B
    V[1, 3] = 1.0
    V[2, 1] = 1.0 / B
    V[2, 2] = 0.5
    V[3, 0] = -0.5 * B
    V[3, 3] = 1.0

    M = np.identity(7)
    M[0, 0] = +1.0
    M[1, 1] = -1.0
    M[2, 2] = math.cos(phase)
    M[2, 3] = math.sin(phase) / B
    M[3, 2] = math.sin(phase) * B * -1.0
    M[3, 3] = math.cos(phase)
    M[4, 5] = length / (sync_params.gamma**2)

    M = np.linalg.inv(V) @ M @ V
    M[4, 5] *= get_dp_p_coeff(sync_params)  # convert_matrix_dp_p_to_dE(M, sync_params)

    sync_params.time += length / (sync_params.beta * speed_of_light)
    return M


def get_matrix_cf(sync_params: Envelope | SyncParticleParams, length: float, kq: float) -> np.ndarray:
    if kq == 0:
        return get_matrix_drift(sync_params, length)

    sqrt_abs_kq = math.sqrt(abs(kq))

    cx = math.cos(sqrt_abs_kq * length)
    sx = math.sin(sqrt_abs_kq * length)

    M = np.identity(7)
    M[0, 0] = M[2, 2] = cx
    M[0, 1] = M[2, 3] = +sx / sqrt_abs_kq
    M[1, 0] = M[3, 2] = -sx * sqrt_abs_kq
    M[1, 1] = M[3, 3] = cx
    M[4, 5] = length / (sync_params.gamma**2)
    M[4, 5] *= get_dp_p_coeff(sync_params)

    sync_params.time += length / (sync_params.beta * speed_of_light)
    return M


def get_matrix_rf_gap(sync_params: Envelope | SyncParticleParams, frequency: float, E0TL: float, phase: float) -> np.ndarray:
    gamma = sync_params.gamma
    beta = sync_params.beta
    mass = sync_params.mass
    charge = sync_params.charge

    kin_energy_in = sync_params.kin_energy
    charge_E0TL_sin = charge * E0TL * math.sin(phase)
    kin_energy_delta = charge * E0TL * math.cos(phase)

    # Calculate parameters in the center of the gap.
    sync_params.kin_energy = kin_energy_in + kin_energy_delta / 2.0
    gamma_gap = sync_params.gamma
    beta_gap = sync_params.beta

    # Move to the end of the gap.
    kin_energy_out = kin_energy_in + kin_energy_delta
    sync_params.kin_energy = kin_energy_out

    # The base RF gap is simple - no phase correction.
    gamma_out = sync_params.gamma
    beta_out = sync_params.beta
    prime_coeff = (beta * gamma) / (beta_out * gamma_out)

    # Wave momentum
    k = 2.0 * math.pi * frequency / speed_of_light
    phase_time_coeff = k / beta

    # Transverse focusing coefficient
    kappa = -charge * E0TL * k / (2.0 * mass * beta_gap**2 * beta_out * gamma_gap**2 * gamma_out)
    d_rp = kappa * math.sin(phase)

    M = np.eye(7)
    M[5, 4] = charge_E0TL_sin * phase_time_coeff
    M[4, 4] = beta_out / beta
    M[1, 1] = prime_coeff
    M[3, 3] = prime_coeff
    M[1, 0] = d_rp
    M[3, 2] = d_rp
    return M


def get_node_matrix(node, sync_params: Envelope | SyncParticleParams, part_index: int = -1) -> np.ndarray | None:
    """Return an analytic node map and advance its reference energy/time.

    Custom handlers can be registered on orbit.envelope.node_matrices.get_node_matrix.
    """
    from .node_matrices import get_node_matrix as dispatch

    return dispatch(node, sync_params, part_index=part_index)
