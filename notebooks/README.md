# Notebooks

This directory has no notebooks yet. The tutorials are the scripts in
[`../examples/`](../examples/README.md). A notebook belongs here when interactive exploration is
the point (a display to rotate, a scan to re-run with other numbers), not as a copy of an example.

Notebooks that would be worth adding, each starting from an example script and built on
`lucid.simulation.setup_event_simulator`, `lucid.fitting` and `lucid.visualization`:

| notebook | content | start from | needs |
|---|---|---|---|
| event displays | one event as an unrolled 2D map, a time-evolution animation and an interactive 3D view; the display suited to each geometry (cylinder, sphere, box, strings) | `hello_simulate.py` | the `[viz]` extra; CPU |
| track gradients | the reconstruction loss and its gradient along each track parameter, and a 2D loss landscape | `hello_gradient.py`, `lucid.gradient_analysis` | CPU at reduced photon count; GPU for full grids |
| data vs prediction | a Geant4 event against the prediction, PMT by PMT, under the charge and first-arrival-time likelihood of the fit | `seed_reconstruct.py --geant4` | CPU |
| reconstruction anatomy | one fit in detail: seeds A, B and F, the convergence, the residuals | `seed_reconstruct.py`, `lucid.fitting.analysis` | GPU for the published fit |
| calibration | the optical parameters recovered with per-PMT gains profiled, the Cramér-Rao bound, and the Fisher matrix showing which parameters are degenerate | `hello_calibrate.py` | CPU at reduced size; GPU at the paper's |

Notebooks built on an earlier API are archived as release assets ([studies](../docs/studies.md));
they do not run against the current code.

A notebook added here needs a CI job that executes it at a small size, as the `examples` job does
for the scripts. Commit it without outputs: `nbstripout` runs in pre-commit, and CI rejects a
tracked notebook that carries outputs.
