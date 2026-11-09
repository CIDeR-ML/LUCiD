# Examples

Short scripts that run the main uses of LUCiD. The documentation pages walk through them and
include their code. Install LUCiD and the example data first
([installation](../docs/getting-started/install.md)), then run them from the repository root,
since they read `config/` and `data/`.

| script | what it does | small run (as in CI) |
|--------|--------------|----------------------|
| `hello_simulate.py` | one 1 GeV muon in SK-like water: the expected charge on each PMT, drawn as an unrolled-cylinder display (`hello_simulate.png`). [Quickstart](../docs/getting-started/quickstart.md) | `--n-rays 20000` |
| `hello_gradient.py` | `jax.grad` of the forward with respect to energy and vertex, and `jax.vmap` over energies. [Quickstart](../docs/getting-started/quickstart.md) | defaults |
| `hello_calibrate.py` | the published calibration estimator recovers seven optical parameters while profiling all 10764 per-PMT gains, and prints the Cramér-Rao bound for each. [Calibration](../docs/guides/calibration.md) | `--n-photons 100000 --steps 3` |
| `hello_calibrate_own_optimiser.py` | one calibration problem solved four ways: the default Gauss-Newton, `optax.adam` through `calibrate(tx=...)`, a hand-written optax loop on `problem.grad_metric_loss`, and scipy's `least_squares`. [Calibration](../docs/guides/calibration.md) | `--n-photons 100000 --steps 3` |
| `seed_reconstruct.py` | the published track reconstruction: seeds from the charge and first-arrival times, a pick by data loss, one Fisher-Gauss-Newton fit. The event is sampled from SIREN, or with `--geant4` taken from the Geant4 muon slice; `--particle` and `--hypothesis` choose muon or electron. [Reconstruction](../docs/guides/reconstruction.md) | `--n-rays 10000 --steps 3 --quick` |
| `hello_telescope.py` | an IceCube-style string array in ice: the expected DOM charge from a through-going muon and from a cascade | defaults |
| `hello_multiparticle.py` | two Geant4 muons from one vertex, `--opening` degrees apart (160 by default), transported together and drawn combined, per primary (`--split`) and coloured by primary. `--file` reads a PhotonSim file with several primaries per event | `--split` |

Each script takes `--help`. They use a GPU when JAX sees one and the CPU otherwise
(`JAX_PLATFORMS=cpu` forces the CPU). `hello_simulate`, `hello_gradient`, `hello_telescope` and
`hello_multiparticle` are practical on a CPU at their defaults. The calibration and
reconstruction examples are sized for a GPU; on an 11 GB card `seed_reconstruct` needs
`XLA_PYTHON_CLIENT_MEM_FRACTION=.95`.

`seed_reconstruct` ends by printing the data loss at the fit. Fitting the same event under both
hypotheses gives a muon/electron test: the lower loss names the particle. With `--quick` the fit
is not converged, so the numbers are only a smoke test.

The PMT overlap tables the examples use come with the example data. A sensor radius or
temperature without a shipped table is computed on first use and cached in
`spatial_overlap_integrals/`, or in `~/.cache/lucid/spatial_overlap_integrals/` when the checkout
is not writable.

`python examples/X.py` imports the installed `lucid`, not necessarily the checkout the script is
in. To run the examples of a second checkout, use `PYTHONPATH=<that checkout> python examples/X.py`.
