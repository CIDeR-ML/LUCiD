# LUCiD hello-world examples

Three short runnable scripts against the **real, current** API — the fastest way to see what
LUCiD does. They auto-detect the platform (GPU if present, else CPU) and write one figure or
table each. `hello_simulate` / `hello_reconstruct` are quick anywhere (~10–20 s warm); on a
GPU `hello_calibrate` (1e6 photons + Fisher/CRB + 100 GN steps) is ~1–2 min — much slower on
CPU. They use a reduced detector grid for speed; production uses the full grid.

```bash
pip install -e .
python examples/hello_simulate.py      # forward: muon -> per-PMT charge (event display)
python examples/hello_calibrate.py     # calibration: recover optical params + Cramer-Rao bound
python examples/hello_reconstruct.py   # reconstruction: muon energy from observed light
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
| `hello_simulate.py` | the differentiable forward `ParticleParams → per-sensor charge`, drawn as an unrolled-cylinder event display | `setup_event_simulator` |
| `hello_calibrate.py` | the validated Gauss-Newton + per-PMT-Schur fit recovering optical scales from calibration sources, vs the Fisher/CRB bound | `lucid.fitting`: `build_calibration_problem`, `fit`, `crb` |
| `hello_reconstruct.py` | reconstructing muon energy by scanning the forward (Stage-0 of the production pipeline); the loss landscape is smooth with a clean minimum at truth | `setup_event_simulator` |

## Notes

- **Calibration is the gradient-fitting showcase.** `hello_calibrate.py` exercises the real
  optimizer (`lucid.fitting`) — the same recipe the calibration campaign ran on.
- **Reconstruction here is a forward scan, by design.** Full multi-parameter track
  reconstruction (vertex/direction/t0/energy by gradient descent) lives in
  `lucid.optimization` and is the subject of `docs/RECON_CONSOLIDATION.md`. The energy
  gradient through the SIREN emitter carries DiCE-score noise, so robust *gradient* recon
  needs the conditioning + loss treatment documented there — out of scope for a 20-line demo.
- The `fit(forward, residual=, solver=)` / `SimParams` / `Field` interface described in
  `docs/MAIN_BRANCH_PLAN.md` is a **proposal**, not yet built. These examples call the
  canonical API that exists today.
