# LUCiD examples

Seven short runnable scripts against the current API: the tutorials. The docs pages narrate them
and include their code, so what a page shows is what runs. Install LUCiD and the example data
first ([installation](../docs/getting-started/install.md)), then run them from the repo root
(they read `config/` and `data/`). With a second checkout, run its examples as
`PYTHONPATH=<that checkout> python examples/X.py` ([why](../scripts/README.md)).

They run on a GPU when one is visible, else on the CPU (`JAX_PLATFORMS=cpu` forces it). Each takes
a small-size switch; CI's `examples` job (`.github/workflows/ci.yml`) runs every one at the size
in the last column, on CPU.

| script | what it shows | small size (CI) |
|--------|---------------|-----------------|
| `hello_simulate.py` | the differentiable forward `ParticleParams → expected charge per PMT`, drawn with the unrolled-cylinder event display (writes `hello_simulate.png`). [Quickstart](../docs/getting-started/quickstart.md) | `--n-rays 20000` |
| `hello_gradient.py` | `jax.grad` of the forward with respect to the energy and the vertex, and `jax.vmap` over energies. [Quickstart](../docs/getting-started/quickstart.md#differentiate-it) | (default) |
| `hello_calibrate.py` | the published calibration estimator recovering seven optical parameters, with every per-PMT gain profiled, against the Cramér-Rao bound. [Calibration](../docs/guides/calibration.md) | `--n-photons 100000 --steps 3` |
| `hello_calibrate_own_optimiser.py` | one calibration problem, four solvers: the default Gauss-Newton, `optax.adam` through `calibrate(tx=)`, a plain optax loop over `problem.grad_metric_loss`, and scipy's `least_squares`. [Calibration](../docs/guides/calibration.md#choosing-a-solver) | `--n-photons 100000 --steps 3` |
| `seed_reconstruct.py` | the published reconstruction: data-driven seeds, a pre-fit pick, one Fisher-Gauss-Newton fit, on a SIREN-sampled event or (`--geant4`) a GEANT4 muon of the bundled slice; a muon or electron event and model (`--particle`, `--hypothesis`). [Reconstruction](../docs/guides/reconstruction.md) | `--n-rays 10000 --steps 3 --quick` |
| `hello_telescope.py` | a neutrino-telescope string array (IceCube, ice): a through-going muon track and a cascade, and the expected charge each puts on the DOMs | (default) |
| `hello_multiparticle.py` | two GEANT4 muons from one vertex (two events of the bundled muon slice, one rotated by `--opening`; `--file` reads a multi-primary file): the truth per primary, the transport, and displays (combined, one per primary with `--split`, and coloured by which primary lit each PMT) | `--split` |

At the default sizes `hello_simulate` and `hello_gradient` take about a minute on 4 CPU cores; the
calibration and reconstruction examples are meant for a GPU (`hello_calibrate` is about 4 minutes on
one; `seed_reconstruct` about 8 on an 11 GB card with `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`, without
which it runs out of memory there), and are much slower on a CPU. The PMT overlap tables they need
come with the example data (`scripts/download_data.py` installs them into
`spatial_overlap_integrals/`); a sensor radius or temperature with no shipped table costs a minute
or two on first use, then is cached there (or in `~/.cache/lucid/spatial_overlap_integrals/` when
the checkout is not writable).

## Notes

- **The calibration bound.** `hello_calibrate` prints, per parameter, the truth, the start, the
  fit, its error and the Cramér-Rao bound, then the RMS error of the 10764 profiled per-PMT
  gains. The bound is a toy demonstration, not a calibration budget: it is built from the
  simulator's autodiff gradients at this small photon count, and the wide `wall`/`sensor` bound
  reflects that charge alone weakly constrains reflectivity. The data are averaged over 8 draws,
  as the paper's calibration does: the Neyman residual weights each sensor by its observed
  charge, and a single draw at ~1.6 PE per sensor biases it.
- **Particle ID by likelihood.** `seed_reconstruct` ends with the data loss at the fit, so one
  event reconstructed under both hypotheses is a μ/e test: the lower loss names the particle
  ([reconstruction](../docs/guides/reconstruction.md#minimal-fit) has the loop). At the small
  size the fit is not converged; the numbers mean something only at the defaults.
- **Multi-particle events** are not a special case for the transport: `hello_multiparticle`
  feeds the data path a flat photon list, so an event with N primaries costs what its photon
  count costs. The split is what the event carries: `particles[i]['photon_indices']` partitions
  its photons by primary (no overlap, nothing dropped), so each particle can be transported and
  drawn on its own. The default event is two muons of the bundled slice 160° apart (`--opening 40`
  overlaps the rings); `--file` takes a PhotonSim ROOT file with more than one primary per event,
  read with `read_particle_data_from_photonsim`, and `--event` picks its event.
