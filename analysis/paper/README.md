# Paper figures

Self-contained, reproducible scripts for the paper's figures. Each figure is one script
(`fig_<name>.py`) with a uniform command-line contract; shared logic lives in `utils/`.

The tree covers **two halves of the paper**:

| half | figures | recipe doc | source of truth |
|---|---|---|---|
| tracking / reconstruction | `fig_nrays`, `fig_energy_scan`, `fig_convergence`, `fig_loss_landscape`, … | `TRACKING_STUDIES.md` | `utils/studies.py` |
| calibration | `fig_calib_convergence`, `fig_calib_loss_geometry` | [below](#calibration-figures) | `utils/calibration.py` |

⚠️ `fig_convergence` and `fig_loss_landscape` are the **reconstruction** figures (per-event p68
vs iteration; 2D landscape over track parameters). Their calibration counterparts carry the
`fig_calib_` prefix and are different figures over different parameters. Do not conflate them.

The figures are named descriptively for now (`fig_nrays`, `fig_energy_scan`, …); the
final paper numbering will be applied later.

## Contract

Every figure script supports:

```
python fig_<name>.py                    # reproduce end-to-end (generate locally, then plot)
python fig_<name>.py --generate-data    # (re)generate the reconstructed data only
python fig_<name>.py --plot-results      # plot from whatever data already exists
```

Data-generation backend:

```
--backend local   (default)  reconstruct a small subset in-process — no cluster, no
                             site assumptions; for checking reproducibility on a laptop.
--backend s3df               submit the published production to the S3DF GPU partitions
                             via utils/submit_job.py (nrays 500 events/point;
                             energy and geom 100).
```

Common options: `--events N`, `--tags 5k,50k,250k`, `--root-base <dir>`, `--out <dir>`.

Figures that only re-plot existing reconstruction output (e.g. convergence, Polyak)
have no `--generate-data` step and are correspondingly simpler.

## Layout

**`fig_*.py` are the only entry points at top level.** Everything else is machinery and
lives in `utils/`; everything generated is git-ignored. Running a figure script is the
whole interface — there are no prerequisite steps to remember.

```
analysis/paper/
  fig_<name>.py         one figure (or a small group) per script — THE entry points
  utils/
    studies.py          ★ single source of truth: the published recipe + the sweeps
    pipeline.py         TrackingPipeline — the reconstruction engine the figures build on
    run.py              run_local() in-process + submit_s3df() SLURM wrapper
    run_study.py        reconstruct one study config -> per-event HDF5 (the SLURM worker)
    submit_job.py       submit configs to S3DF SLURM (one GPU job per config)
    make_geometries.py  SK_like geometry JSONs at varying PMT counts (auto-called)
    generate_data.py    generate the energy-sweep PhotonSim ROOTs (the inputs)
    paths.py            default output locations (repo-local; s3df area opt-in)
    plotting.py         thin wrappers over utils/ovv_plots
    ovv_plots.py        the nrays/energy/geom/convergence plot modes
    -- calibration --
    calibration.py      ★ single source of truth for the CALIBRATION campaign: detector,
                          truth model, source layout, CALIB_RECIPE, LANDSCAPE_RECIPE
    calib_run.py        ★ runs one seed of the 19-parameter joint optical fit, through
                          lucid.fitting. Typed arguments, no environment variables; the
                          seeds share one compiled forward.
    calib_plots.py      rendering for both calibration figures (plot-only)
    damping.py          the damped GN step the 2D loss-geometry figure's two halves share
                          (numpy only; see tests/test_paper_damping.py for why it is not
                          lucid.fitting.transforms.damped_matrix)
    compute_2d_neyman.py / combine_2d.py / traj_2d_neyman.py
                        the three data stages behind fig_calib_loss_geometry
    p68_evolution.py    per-iteration p68 convergence helper (used by ovv_plots)
  configs/              generated study config JSONs (git-ignored)
  geometries/           generated SK_like geometry JSONs (git-ignored)
  output/               generated data + figures (default target; git-ignored)
```

`utils/studies.py` defines the reconstruction recipe and every sweep once; the figure
scripts build their configs from it at run time. Nothing else may define a sweep or a
recipe value — that duplication is what let the scripts drift from the published data.

## Calibration figures

Both are closure fits: truth data from the same differentiable simulator, fitted from a displaced
start. The estimator is `lucid.fitting`'s (Neyman residual, per-PMT gains profiled); the knobs live
in one table each in `utils/calibration.py`, and every run writes the knobs it used into its `.npz`.

- **`fig_calib_convergence`** — the joint 19-parameter fit over iterations, 3 seeds.
  `utils/calib_run.py` runs each seed through `lucid.fitting` with `CALIB_RECIPE`;
  `utils/calib_plots.convergence` renders. The published run is 600 Gauss-Newton steps per seed on
  the SK-like geometry; `--steps` shortens it for a smoke test (not the paper's figure). The
  trajectories are Polyak-smoothed at plot time (`--polyak-window`), so changing the window needs
  no re-run.
- **`fig_calib_loss_geometry`** — `log10 L` over (scattering, absorption length) with one laser,
  gradient-descent and Gauss-Newton paths overlaid. `compute_2d_neyman.py` (sharded) →
  `combine_2d.py` → `traj_2d_neyman.py` → `utils/calib_plots.loss_geometry`, all reading
  `LANDSCAPE_RECIPE`, so the surface and the paths share `K` / `N_PH`. The shards need
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (set by the driver), and the driver refuses to combine an
  incomplete or stale set of shards.

## Inputs

Local reproduction reconstructs from PhotonSim ROOT files
(`<root-base>/<mu-|e->/<E>MeV_<N>events.root`). Point `--root-base` at a directory that
has them; generating them from scratch is `analysis/paper/utils/generate_data.py`.

## Reconstruction defaults

The published working point, defined in `utils/studies.py` and applied by every figure
script with no flags needed:

| | |
|---|---|
| TTS | **2.1 ns**, with a matched likelihood kernel (`gn.sigma == tts`) |
| charge resolution | **`Abe_2013`** per-PMT charge model on the pseudo-data |
| Fisher-GN | `fisher_mode='ad'`, `lr=4.0`, `nkeys=8`, `niters=150` |
| vertices | contained, random true t0 (±15 ns), uncorrected |
| reflection | `scalar_mix` — a **code** default (`lucid/simulation/simulator.py`), not a config key |

⚠️ `fisher_mode` and `lr` are a **tuned pair**. The legacy `('fd', lr=8.0)` set is not
interchangeable with `('ad', lr=4.0)`: the AD metric is smaller per parameter, so the
FD-tuned step overshoots. Change one and you must retune the other.

These values are duplicated once, in `utils/pipeline.py:DEFAULT_CONFIG`, because a study
config only overrides what varies — so anything it omits must still land on the paper
recipe. Keep the two in step.
