# Envelope examples

Run these scripts with an installed PyORBIT build and NumPy, SciPy, and
Matplotlib. The six small tutorials below construct their own lattices and
require no SNS input files. Run them on **one MPI rank**: each particle example
generates the complete bunch locally.

From the repository root:

```sh
python examples/Envelope/test_env_2d_matching.py
python examples/Envelope/test_env_solenoid.py
python examples/Envelope/test_env_dispersion.py
python examples/Envelope/test_env_rf_acceleration.py
python examples/Envelope/test_env_convergence.py
python examples/Envelope/test_env_3d_focusing.py
```

Each script prints its output directory and writes PNG figures and NPZ data to
`examples/Envelope/outputs/<script>/<timestamp>/`. Use `--output-dir PATH` to
choose a destination (files with the same names are overwritten there).
Use `MPLBACKEND=Agg` in a headless environment. Styles resolve relative to the
scripts, so these tutorials can run from any working directory.

All six accept `--kin-energy` in GeV. Except for the convergence scan, they
accept `--nslice` (parts per element) and `--fit 0|1` (analytic or fitted
external transport maps). Particle comparisons accept `--nparts` (at least 8)
and `--seed`. Seeded samples are centered and whitened before applying the
target covariance, eliminating initial second-moment sampling differences.
Envelope and particles also have independent reference-particle state.

## Periodic matching with space charge

`test_env_2d_matching.py` solves for the four uncoupled transverse Twiss
parameters of a periodic FODO envelope at fixed geometric RMS emittances.
It starts from zero-current optics and increases intensity in steps, solving
the periodicity conditions with SciPy's bounded least-squares optimizer.
The script fails explicitly if matching does not converge.

```sh
python examples/Envelope/test_env_2d_matching.py --intensity 5e9 --cells 40 --mismatch 1.3
python examples/Envelope/test_env_2d_matching.py --sc 0
```

The mismatched beam is stretched in x and compressed inversely in x',
preserving its emittance. `rms.png` compares matched and mismatched beam
sizes throughout the cells. The matched envelope varies inside each cell;
periodicity is measured at cell boundaries. `match.npz` stores the matched
covariance, parameters `[log(beta_x), alpha_x, log(beta_y), alpha_y]`, and
periodicity residual. `--bunch-length` is the full uniform longitudinal
length in meters; the coasting-beam model has zero energy spread.

## Solenoid coupling

`test_env_solenoid.py` transports an unequal-emittance beam through repeated
drift/solenoid/drift cells without space charge. Change `--field` in tesla
or `--cells` to explore the coupling.

`emittances.png` compares projected transverse emittances with the two
transverse eigenemittances; `correlations.png` shows correlations between
x and y, and x and y'. Observation planes are outside the solenoid, with
fringe tracking disabled, using the symplectic TEAPOT solenoid map.
The transverse eigenemittances are calculated from the 4D transverse block,
independently of the longitudinal coordinate units.

## Dispersion and an off-momentum centroid

`test_env_dispersion.py` uses a drift, sector bend, and exit drift with finite
momentum spread and a centroid momentum offset. Options `--sigma-dp` and
`--offset-dp` specify RMS and mean dp/p; `--angle` is in degrees.

ORBIT stores dE in GeV, so the script converts using the first-order relation
`dE = beta * p * (dp/p)`. Dispersion is inferred from `cov(x, dp/p) / var(dp/p)`
and similarly for x'. `dispersion.png` shows dispersion and centroid motion;
`rms.png` separates total and dispersion-subtracted horizontal RMS size.
The envelope uses a linear map; increasing momentum spread or offset can
increase its difference from nonlinear particle tracking.

## RF acceleration

`test_env_rf_acceleration.py` builds a short linac with quadrupoles and one
independent RF cavity per cell. A design-bunch pass initializes cavity
arrival times. Gaps use `MatrixRfGap`, and envelope maps are recomputed as
the reference energy changes.

```sh
python examples/Envelope/test_env_rf_acceleration.py --cells 8 --phase -30 --voltage 0.0001
```

`--voltage` is E0TL in GeV, `--frequency` is in Hz, and `--phase` is in
degrees. `acceleration.png` shows energy and transverse/longitudinal sizes;
`emittances.png` compares geometric and normalized transverse emittances.
Normalized transverse emittance is conserved by the linear model even as
geometric emittance changes with acceleration. Particle transport through
the surrounding quadrupoles/drifts can introduce small nonlinear differences.

## Slicing convergence and cost

`test_env_convergence.py` compares both analytic and fitted maps for a FODO
envelope with 2D space charge. Each point starts with a fresh lattice and
envelope; tracking time includes map preparation and subsequent reuse across
cells, and excludes lattice construction and plotting. Analytic mode disables
the fallback to fitted maps.

```sh
python examples/Envelope/test_env_convergence.py --slices 2 4 8 16 32 64 --reference-slices 256
```

`convergence.png` plots transverse covariance error against slicing and runtime.
The error is the maximum elementwise covariance difference divided by the
reference RMS products. The reference is a finer analytic-map calculation
with the same space-charge closure, rather than an exact physical solution.
Use more `--reference-slices` to check reference resolution. `convergence.npz`
stores errors, timings, final covariances, and the reference covariance.
Timings of short runs can be noisy.

## Finite-emittance 3D focusing

`test_env_3d_focusing.py` compares the 3D envelope against uniform-ellipsoid
particle space charge and 3D PIC in a short quadrupole channel. Initial
transverse slopes and energy spread are nonzero. Both particle solvers use
identical initial samples and fresh lattices.

```sh
python examples/Envelope/test_env_3d_focusing.py --sc-grid 64 --nparts 100000
python examples/Envelope/test_env_3d_focusing.py --models ellipsoid
python examples/Envelope/test_env_3d_focusing.py --sc 0
```

`rms.png` and `emittances.png` compare all three models. The uniform-ellipsoid
particle solver assumes upright spatial axes. A finite-emittance 6D waterbag
(the default), or Gaussian (`--dist gauss`), has a nonuniform spatial density;
the uniform-ellipsoid envelope closure is approximate. Differences from PIC
can therefore persist with finer slicing, larger particle counts, and finer
grids. This is a model comparison, not an exact 3D equilibrium benchmark.

## Saved diagnostics and existing examples

Each history NPZ contains `s`, `mean`, `cov`, kinetic energy, beta/gamma,
RMS sizes, projected emittances, normalized emittances, and eigenemittances.
Coordinates are `(x, x', y, y', z, dE)` in meters, radians, and GeV; saved
arrays retain those units. Figures apply the indicated unit conversions.
Solenoid and dispersion histories also include their specialized diagnostics.
Printed covariance differences use RMS products to compare coordinates with
different units.

The existing `test_env_2d_fodo.py`, `test_env_3d_drift.py`, `sns_linac/`, and
`sns_ring/` provide larger particle benchmarks and speed comparisons.
`run.sh` runs those benchmarks and the six tutorials from this directory.
