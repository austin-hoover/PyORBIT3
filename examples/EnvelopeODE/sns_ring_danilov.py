"""Calculate matched envelope in SNS ring."""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from tqdm import trange

from orbit.envelope_ode import DanilovEnvelope
from orbit.envelope_ode import DanilovEnvelopeTracker
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton

plt.style.use("style.mplstyle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kin-energy", type=float, default=0.8)
    parser.add_argument("--max-node-length", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--emittance", type=float, default=4.25e-6)
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1])

    parser.add_argument("--pulse-intensity", type=float, default=1.5e11)
    parser.add_argument("--pulse-width", type=float, default=38.0)
    parser.add_argument("--npulse", type=int, default=1)
    return parser.parse_args()


def calc_tune_fft(x: np.ndarray, window: bool = True) -> np.ndarray:
    x = np.copy(x)
    if x.ndim == 1:
        x = x[:, None]

    n = x.shape[0]

    if window:
        shape = [1] * x.ndim
        shape[0] = n
        x *= scipy.signal.windows.hann(n).reshape(shape)

    amplitudes = np.fft.rfft(x, axis=0)
    frequencies = np.fft.rfftfreq(n, d=1.0)
    index = np.argmax(np.abs(amplitudes), axis=0)
    return np.squeeze(frequencies[index])


def main(args: argparse.Namespace) -> None:
    lattice = TEAPOT_Lattice()
    lattice.readMADX("inputs/sns_ring_madx.lat", "rnginjsol")
    lattice.initialize()

    for name in ["scbdsol_c13a", "scbdsol_c13b"]:
        node = lattice.getNodeForName(name)
        node.setParam("B", 0.15 / (2.0 * node.getLength()))

    for node in lattice.getNodes():
        node_length = node.getLength()
        if node_length > args.max_node_length:
            node.setnParts(1 + int(node_length/  args.max_node_length))

    intensity = args.npulse * args.pulse_intensity
    length = lattice.getLength() * (args.pulse_width / 64.0)

    if args.mode == 0:
        eps_1 = args.emittance
        eps_2 = 0.0
    else:
        eps_1 = 0.0
        eps_2 = args.emittance

    envelope = DanilovEnvelope(
        eps_1=eps_1,
        eps_2=eps_2,
        mass=mass_proton,
        kin_energy=args.kin_energy,
        length=length,
        line_density=0.0,
        params=None,
    )
    tracker = DanilovEnvelopeTracker(lattice)
    tracker.match_zero_sc(envelope, method="2d")
    envelope.set_line_density(intensity / length)
    tracker.match(envelope, method="replace_avg", verbose=1)

    envelope_copy = envelope.copy()
    for turn in range(20):
        cov_matrix = envelope_copy.cov()
        x_rms = 1000.0 * np.sqrt(cov_matrix[0, 0])
        y_rms = 1000.0 * np.sqrt(cov_matrix[2, 2])
        print("turn={} xrms={:0.2f} yrms={:0.2f}".format(turn, x_rms, y_rms))

        tracker.track(envelope_copy)

    # Calculate tune (integration through lattice)
    tune = tracker.calc_tune(envelope)
    print("tune =", tune)

    # Calculate tune (FFT)
    turns = 1000
    particles = np.zeros((1, 6))
    particles[0, :4] = 0.5 * envelope.get_coordinates()

    particles_tbt = np.zeros((turns, particles.shape[0], particles.shape[1]))
    for turn in trange(turns):
        particles_tbt[turn] = particles.copy()
        envelope, particles = tracker.track_particles(envelope, particles)

    particles_tbt = np.squeeze(particles_tbt)
    particles_tbt = particles_tbt[:, :4]

    frac_tune = calc_tune_fft(particles_tbt[:, 0])  # nux = nuy = nu1
    print("frac_tune_fft =", frac_tune)


if __name__ == "__main__":
    main(parse_args())