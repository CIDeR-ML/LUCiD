# LUCiD hello-world examples

Five short runnable scripts against the **real, current** API — the fastest way to see what
LUCiD does. They auto-detect the platform (GPU if present, else CPU) and write a figure or
table. `hello_simulate` is quick anywhere (~15 s warm); `hello_calibrate` (1e6 photons +
Fisher/CRB + 100 GN steps) and `hello_reconstruct` (100+-step Fisher-GN) are ~1–3 min on a
GPU, much slower on CPU. They use a reduced detector grid for speed; production uses the full grid.

```bash
pip install -e .
python examples/hello_simulate.py      # forward: muon -> per-PMT charge (event display)
python examples/hello_calibrate.py     # calibration: recover optical params + Cramer-Rao bound
python examples/hello_reconstruct.py   # reconstruction: 9-param track fit; --closure/--data-fit, --data/--hypothesis mu|e
python examples/seed_reconstruct.py    # reconstruction from a data-driven seed (seed -> fit)
python examples/hello_telescope.py     # neutrino telescope: muon track + cascade in an ice string array
```

Expected `hello_calibrate` output (7 globals recovered, all 10764 per-PMT gains marginalised):

```
param                     truth    start      fit     err    CRB
g                         0.900    0.938    0.937  +4.1%   0.4%
scatter_length           70.000   65.329   65.583  -6.3%   0.1%
mie_scatter_length     3000.000 2614.059 2621.090 -12.6%   0.4%
absorption_length        60.000   51.899   61.524  +2.5%  13.4%
wall_reflection_rate      0.200    0.220    0.175 -12.4%  32.5%
sensor_reflection_rate    0.200    0.226    0.206  +3.2%  52.8%
qe                        0.070    0.072    0.069  -1.0%   7.6%
```
(Errors above CRB on `scatter`/`mie`/`g` are the reduced-grid forward-noise floor, not the
Fisher bound; the wide `wall`/`sensor` CRB reflects that charge alone weakly constrains
reflectivity — the timing-observable frontier.)

| script | what it shows | API it calls |
|--------|---------------|--------------|
| `hello_simulate.py` | the differentiable forward `ParticleParams → per-sensor charge`, drawn with the canonical unrolled-cylinder event display (Cherenkov ring on the barrel) | `setup_event_simulator`, `create_detector_display` |
| `hello_calibrate.py` | the validated Gauss-Newton + per-PMT-Schur fit recovering optical scales from calibration sources, vs the Fisher/CRB bound | `lucid.fitting`: `build_calibration_problem`, `fit`, `crb` |
| `hello_reconstruct.py` | 9-parameter track reconstruction (E, vertex, direction, t0) by Fisher-Gauss-Newton — muon **or** electron, closure (prediction-vs-prediction) or a real GEANT4 data fit, and cross-particle (fit an electron event with the muon model, etc.), all via CLI | `setup_event_simulator`, `lucid.fitting`: `ReconModel`, `fit_track` |
| `seed_reconstruct.py` | the honest pipeline: a data-driven initial guess (energy scan → vertex/t0 grid → cone direction) from `lucid.optimization`, then refine with `fit_track` | `lucid.optimization.grid_search` / `utils.functions`, `lucid.fitting` |
| `hello_telescope.py` | a neutrino-telescope string array (IceCube, ice): a through-going muon **track** and a **cascade** shower, DOM counts for each | `setup_event_simulator(detector_type='string')`, `lucid.sources.cascade` |

## Notes

- **Calibration is the gradient-fitting showcase.** `hello_calibrate.py` exercises the real
  optimizer (`lucid.fitting`) — the same recipe the calibration campaign ran on.
- **Reconstruction** is a 9-parameter `[E, x, y, z, dir, t0]` Fisher-Gauss-Newton fit on a
  Poisson-charge + first-arrival **order-statistic-time** loss, run in SCALE9-preconditioned
  coordinates with finite-difference Jacobians (the DiCE `custom_vjp` blocks `jacfwd`, and the
  autodiff track-Hessian is indefinite, so a PSD Fisher metric is built and stepped against).
  SK_like geometry, 1000 MeV, from an aggressive start (up to 3 m / 200 MeV / 10° / 3 ns off
  truth). Two study axes, set by CLI:

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
  start offset.
- The `fit(forward, residual=, solver=)` / `SimParams` / `Field` interface described in
  `docs/internal/MAIN_BRANCH_PLAN.md` is a **proposal**, not yet built. These examples call the
  canonical API that exists today.
