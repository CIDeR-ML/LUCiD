# LUCiD hello-world examples

Seven short runnable scripts against the **real, current** API — the fastest way to see what
LUCiD does. They auto-detect the platform (GPU if present, else CPU) and write a figure or
table. `hello_simulate` is quick anywhere (~15 s warm); `hello_calibrate` (1e6 photons, data
averaged over 8 draws, Fisher/CRB, 100 GN steps) is ~4 min on a GPU, `hello_calibrate_adam` ~1.5
min, and `hello_reconstruct` (100+-step Fisher-GN) ~1–3 min; all are much slower on CPU. Most use a reduced detector grid for speed; `hello_reconstruct` runs the full production grid so it matches the campaign recon.

```bash
pip install -e .
python examples/hello_simulate.py      # forward: muon -> per-PMT charge (event display)
python examples/hello_calibrate.py     # calibration: recover optical params + Cramer-Rao bound
python examples/hello_calibrate_adam.py  # calibration, minimal: two params, jax.jacfwd + optax.adam
python examples/hello_reconstruct.py   # reconstruction: 9-param track fit; --closure/--data-fit, --data/--hypothesis mu|e
python examples/seed_reconstruct.py    # reconstruction from a data-driven seed (seed -> fit)
python examples/hello_telescope.py     # neutrino telescope: muon track + cascade in an ice string array
python examples/hello_multiparticle.py --split   # multi-particle GEANT4 event: 2 muons, one vertex
```

Expected `hello_calibrate` output (7 globals recovered, all 10764 per-PMT gains profiled):

```
param                     truth    start      fit     err    CRB
g                         0.900    0.938    0.946  +5.1%  23.1%
scatter_length           70.000   65.329   69.953  -0.1%   5.4%
mie_scatter_length     3000.000 2614.059 2597.409 -13.4% 178.5%
absorption_length        60.000   51.899   60.914  +1.5%  13.7%
wall_reflection_rate      0.200    0.220    0.197  -1.7%  35.6%
sensor_reflection_rate    0.200    0.226    0.192  -3.9%  53.4%
qe                        0.070    0.072    0.069  -0.9%  11.7%
```
(The CRB column is a toy demonstration, not a calibration budget: it is built from the
simulator's autodiff gradients at this small photon count, and those gradients are somewhat worse
than finite differences for the scattering-type parameters, so the bound is loose there. The wide
`wall`/`sensor` bound also reflects that charge alone weakly constrains reflectivity. The data are
averaged over 8 draws, as the paper's calibration does: the Neyman residual weights each sensor by
its observed charge, and a single draw at ~1.6 PE per sensor biases it.)

| script | what it shows | API it calls |
|--------|---------------|--------------|
| `hello_simulate.py` | the differentiable forward `ParticleParams → per-sensor charge`, drawn with the canonical unrolled-cylinder event display (Cherenkov ring on the barrel) | `setup_event_simulator`, `create_detector_display` |
| `hello_calibrate.py` | the Gauss-Newton fit recovering optical scales from calibration sources (Neyman chi2, per-PMT gains profiled in closed form), vs the Fisher/CRB bound | `lucid.fitting`: `build_calibration_problem`, `fit`, `crb` |
| `hello_calibrate_adam.py` | the smallest calibration to set up yourself: a squared-charge loss, a forward-mode gradient through the simulator, and `optax.adam`, recovering absorption length and QE (a closure test); says when to move to the recipe | `setup_event_simulator`, `build_calibration_problem`, `jax.jacfwd`, `optax` |
| `hello_reconstruct.py` | 9-parameter track reconstruction (E, vertex, direction, t0) by Fisher-Gauss-Newton — muon **or** electron, closure (prediction-vs-prediction) or a real GEANT4 data fit, and cross-particle (fit an electron event with the muon model, etc.), all via CLI | `setup_event_simulator`, `lucid.fitting`: `ReconModel`, `fit_track` |
| `seed_reconstruct.py` | the honest pipeline: a data-driven initial guess (energy scan → vertex/t0 grid → cone direction) from `lucid.optimization`, then refine with `fit_track` | `lucid.optimization.grid_search` / `utils.functions`, `lucid.fitting` |
| `hello_multiparticle.py` | loading a **multi-particle** PhotonSim event (two muons sharing a vertex, independent isotropic directions): the truth per primary, the transport, and displays — combined, one per primary, and one coloured by which primary lit each PMT | `read_particle_data_from_photonsim`, `pad_photon_data`, `setup_event_simulator(is_data=True)`, `create_detector_display` / `unroll_layout` |
| `hello_telescope.py` | a neutrino-telescope string array (IceCube, ice): a through-going muon **track** and a **cascade** shower, DOM counts for each | `setup_event_simulator(detector_type='string')`, `lucid.sources.cascade` |

## Notes

- **Calibration is the gradient-fitting showcase.** `hello_calibrate.py` exercises the real
  optimizer (`lucid.fitting`) — the same recipe the calibration campaign ran on.
- **Reconstruction** is a 9-parameter `[E, x, y, z, dir, t0]` Fisher-Gauss-Newton fit on a
  Poisson-charge + first-arrival **order-statistic-time** loss, run in SCALE9-preconditioned
  coordinates with finite-difference Jacobians (the DiCE `custom_vjp` blocks `jacfwd`, and the
  autodiff track-Hessian is indefinite, so a PSD Fisher metric is built and stepped against).
  SK_like geometry at the full production grid (80/120/80), 1000 MeV, from an aggressive start
  (up to 3 m / 200 MeV / 10° / 3 ns off truth). The per-fit engine (`fit_track`) is the same as
  the campaign recon — only the multi-start seeding differs. Two study axes, set by CLI:

  ```bash
  # closure (default): the "data" event is SIREN-generated, fit with the SIREN model.
  python examples/hello_reconstruct.py                            # muon vs muon (self-consistent)
  python examples/hello_reconstruct.py --data e --hypothesis e    # electron vs electron
  python examples/hello_reconstruct.py --data e --hypothesis mu   # muon model on an electron event

  # data-fit: the "data" event is a real GEANT4 event (PhotonSim ROOT), fit with the SIREN model.
  python examples/hello_reconstruct.py --data-fit --data mu --hypothesis mu
  python examples/hello_reconstruct.py --data-fit --data e  --hypothesis mu   # cross-particle
  ```

  `--data`/`--hypothesis` each take `mu` or `e`. **Closure** with `--data == --hypothesis` is
  self-consistent — the optimizer/loss machinery with no model mismatch (~10 cm vertex from a
  ~3 m start). The interesting studies are **cross-particle** and **`--data-fit`**: a
  wrong-particle hypothesis, or the real GEANT4-vs-SIREN emission mismatch, blows the vertex out
  to ~1 m and drifts t0 by several ns — so vertex and t0 discriminate μ/e. `--data-fit` needs the
  ROOT files (`./scripts/download_data.sh`); `--event N` selects the GEANT4 event, `--seed N` the
  start offset, `--photons N` the model ray count (default 250k; more sharpens the closure fit but
  needs more GPU memory).
- **Particle ID by likelihood.** Each run also prints the **final data loss** (the metric the
  multistart uses to arbitrate), so reconstructing one event under both hypotheses is a μ/e test —
  the correct hypothesis gives the lower loss. E.g. 5 electron GEANT4 events under both models:

  ```bash
  for ev in 0 1 2 3 4; do
    for hyp in e mu; do
      python examples/hello_reconstruct.py --data-fit --data e --hypothesis "$hyp" --event "$ev"
    done
  done
  ```

  On electron data the electron hypothesis wins the loss on every event, while the muon hypothesis
  can't fit the shower (vertex ~1.5–2 m off, t0 drifting several ns).
- **Multi-particle events** are not a special case for the transport: `hello_multiparticle.py`
  feeds the data path a flat photon list, so an event with N primaries costs what its photon
  count costs. The reader is what makes the split possible — `particles[i]['photon_indices']`
  partitions the event's photons by primary (exactly: no overlap, nothing dropped), so each
  particle can be transported and drawn on its own. `--split` writes the per-primary displays
  plus a per-particle colouring; `--file` takes any PhotonSim ROOT with >1 primary per event,
  and `--event` picks the topology (93 back-to-back, 0 overlapping rings).
- A `fit(forward, residual=, solver=)` / `SimParams` / `Field` interface has been
  proposed but is **not built**. These examples call the canonical API that exists
  today.
