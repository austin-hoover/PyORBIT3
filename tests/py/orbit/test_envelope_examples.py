"""Run the small tutorials and check their physical diagnostics and outputs."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "Envelope"


@pytest.mark.parametrize("name,extra", [
    ("2d_matching", ["--cells", "3"]),
    ("solenoid", ["--cells", "2", "--nparts", "1000"]),
    ("dispersion", ["--nparts", "1000"]),
    ("rf_acceleration", ["--cells", "3", "--nparts", "1000"]),
    ("convergence", ["--cells", "2", "--slices", "2", "4", "8", "--reference-slices", "32"]),
    ("3d_focusing", ["--cells", "2", "--nparts", "1000", "--sc-grid", "16"]),
])
def test_envelope_tutorial(tmp_path, tmp_path_factory, name, extra):
    args = [] if name == "convergence" else ["--nslice", "4"]
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / f"test_env_{name}.py"), *args, *extra,
         "--output-dir", str(tmp_path)],
        cwd=tmp_path,  # Verify scripts and styles do not depend on the CWD.
        env={**os.environ, "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path_factory.getbasetemp() / "mpl")},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    figures = list(tmp_path.glob("*.png"))
    assert figures and all(path.stat().st_size > 1000 for path in figures)
    histories = {}
    for path in tmp_path.glob("*.npz"):
        with np.load(path) as data:
            histories[path.stem] = dict(data)
    assert histories
    for data in histories.values():
        assert all(np.all(np.isfinite(value)) for value in data.values())

    if name == "convergence":
        data = histories["convergence"]
        for mode in ("analytic", "fitted"):
            assert np.all(np.diff(data[f"{mode}_error"]) < 0)
            assert np.all(data[f"{mode}_seconds"] > 0)
        np.testing.assert_allclose(data["analytic_cov"], data["fitted_cov"], rtol=1e-6, atol=1e-12)
        return

    if name == "2d_matching":
        assert np.max(np.abs(histories["match"]["residual"])) < 1e-6
        matched = histories["matched"]["cov"][::5, :4, :4]
        np.testing.assert_allclose(matched, np.broadcast_to(matched[0], matched.shape), atol=1e-12)
        # The intentional mismatch preserves the transverse RMS emittance.
        np.testing.assert_allclose(histories["matched"]["eps_x"][0],
                                   histories["mismatched"]["eps_x"][0], rtol=1e-12)
        return

    env = histories["envelope"]
    for model, history in histories.items():
        np.testing.assert_allclose(history["s"], env["s"])
        np.testing.assert_allclose(history["cov"][0], env["cov"][0], atol=1e-18)
        assert np.all(history["rms_x"] > 0)
        assert np.all(history["rms_y"] > 0)
        assert np.all(history["rms_z"] > 0)
        if model != "envelope" and name != "3d_focusing":
            scale = np.sqrt(np.diag(env["cov"][-1]))
            error = np.max(np.abs(env["cov"][-1] - history["cov"][-1]) / np.outer(scale, scale))
            assert error < 0.005

    if name == "solenoid":
        np.testing.assert_allclose(env["eigen_xy"] / env["eigen_xy"][0], 1.0, atol=1e-12)
        assert np.ptp(env["eps_x"]) > 0.001 * env["eps_x"][0]
    elif name == "dispersion":
        assert abs(env["dispersion_x"][-1]) > 0.1
        assert env["rms_x"][-1] > env["rms_x_betatron"][-1]
        np.testing.assert_allclose(env["mean"][-1, 0], env["dispersion_x"][-1] * 5e-4, rtol=1e-8)
    elif name == "rf_acceleration":
        assert env["kin_energy"][-1] > env["kin_energy"][0]
        for axis in ("x", "y"):
            np.testing.assert_allclose(env[f"eps_{axis}_n"] / env[f"eps_{axis}_n"][0], 1.0, atol=1e-12)
    elif name == "3d_focusing":
        assert {"ellipsoid", "pic"} <= histories.keys()
        # All three models include a finite longitudinal expansion.
        assert all(history["rms_z"][-1] > history["rms_z"][0] for history in histories.values())
