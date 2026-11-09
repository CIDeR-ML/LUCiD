# notebooks/

Empty today. The tutorials are the scripts in [`../examples/`](../examples/), which the docs pages
include and CI runs. A notebook earns a place where interactive exploration is the point
(something to look at, turn, or re-run with other numbers), not as a second copy of an example.

## Notebooks worth adding

Each of these existed before the API it used was rewritten. The old versions are in git history
(`tutorials/` and `good_notebooks/` at `1fb6cb67^`; `notebooks/_source/` at `9e02c407`) and in the
release assets `studies-archive-notebooks-2026` and `studies-archive-notebooks-legacy-2026`
([`docs/studies.md`](../docs/studies.md)). They call functions that no longer exist
(`lucid.optimization`, `build_calibration_problem`, `fit_track_multistart`), so port the idea, not
the code: build on `lucid.simulation.setup_event_simulator`, `lucid.fitting` and
`lucid.visualization`, starting from the matching example script.

| notebook | what it shows | needs |
|---|---|---|
| quickstart | one SK-like muon simulated and displayed, and the `PhotonList` the forward returns (was `tutorials/00_quickstart`; start from `hello_simulate.py`) | SIREN weights (`scripts/download_data.py`); CPU |
| event displays | one event three ways (unrolled 2D map, time-evolution animation, interactive 3D), then the display that fits each geometry: discs on cylinder/sphere/box walls, a 3D scatter of DOMs on strings (was `tutorials/event_displays`, `good_notebooks/displays/event_displays`) | SIREN weights; the `[viz]` extra (plotly, kaleido, imageio); CPU |
| track gradients and loss landscapes | the reconstruction loss and its gradient along each track parameter (smooth basins, zero crossings at truth), energy constrained by the total charge, a 2D landscape (was `tutorials/track_gradients`, `good_notebooks/gradients/loss_landscapes`; start from `hello_gradient.py` and `lucid.gradient_analysis`) | SIREN weights; CPU at a reduced photon count, GPU for full grids |
| data vs prediction | a GEANT4 event against the differentiable prediction under the likelihood the fit uses (Poisson charge and first-arrival time NLL), PMT by PMT (was `tutorials/data_vs_prediction`) | SIREN weights and a GEANT4 slice (`data/water/muon/1000MeV_5events.root`); CPU |
| reconstruction anatomy | one fit opened up: seeds A, B and F, the longitudinal basin, the convergence, the residual decomposed (was `good_notebooks/reconstruction/recon_anatomy`, `tutorials/track_optimization`; start from `seed_reconstruct.py` and `lucid.fitting.analysis`) | SIREN weights and a GEANT4 slice; a GPU for the published fit (minutes there, much longer on a CPU), or `--quick`-sized settings on a CPU |
| calibration | the published estimator recovering the optical parameters with the per-PMT QE map profiled, against the Cramér-Rao bound; then the gradient and the Fisher matrix (which parameters are degenerate) (was `tutorials/calibration_optimization`, `tutorials/calibration_gradients`, `good_notebooks/calibration/calibrate_optics`; start from `hello_calibrate.py`) | no downloaded data (lasers and a flasher, no emitter); CPU at a reduced size, GPU at the paper's |
| wavelength calibration | the optics at each SK laser line and how identifiable each parameter is per wavelength (was `good_notebooks/calibration/wavelength_calibration`) | no downloaded data; GPU |

## Rules for a notebook

- It comes with a CI job that executes it at a small size, as the `examples` job does for the
  scripts.
- It is committed with its outputs stripped: `nbstripout` runs in pre-commit, and the CI `size`
  job rejects a notebook with outputs.
