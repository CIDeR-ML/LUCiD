# Paper figures

Self-contained, reproducible scripts for the paper's figures. Each figure is one script
(`fig_<name>.py`) with the command-line contract below; shared logic lives in `utils/`.

The tree covers **two halves of the paper**, tracking/reconstruction (source of truth
`utils/studies.py`, [below](#tracking-studies)) and calibration (`utils/calibration.py`,
[below](#calibration-figures)). The scripts and what each needs beyond the configs:

| script | draws | input |
|---|---|---|
| `fig_charge_displays` | data-like vs predicted charge on a cylinder, a sphere and a box | `data/water/muon/1000MeV_5events.root` (`lucid-data-v1`) |
| `fig_cylinder_2d_displays` | unrolled-cylinder charge and first-arrival time of one muon | event 5 of `data/water/muon/1500MeV_6events.root` (`lucid-data-v1`) |
| `fig_photon_event_display` | 3D Cherenkov photons coloured by creation time | `data/water/muon/1000MeV_5events.root` |
| `fig_detector_gallery` | the four geometry families with scale bars | configs only |
| `fig_siren_profile` | SIREN Cherenkov emission profiles | the shipped emitters |
| `fig_t0_correction` | photon arrival delay vs distance, with the shipped t0 fit | emitters; the points need a PhotonSim sample (`--generate-data` or `--data-dir`) |
| `fig_juno_wbls` | Cherenkov vs scintillation on a WbLS sphere, and a scintillation parameter's loss | a PhotonSim ROOT file (`--root`; the default is entry 0 of the shipped 1 GeV muon slice) |
| `fig_compute_performance` | forward and fit-step timing vs number of rays, per K | the shipped emitters; a GPU for the published numbers |
| `fig_nrays` | tracking resolution vs number of rays, and the fit convergence at the largest | PhotonSim ROOT sweep (`--root-base`) |
| `fig_energy_scan` | tracking resolution vs energy | PhotonSim ROOT sweep (`--root-base`) |
| `fig_geometry_scan` | tracking resolution vs PMT count | PhotonSim ROOT sweep (`--root-base`) |
| `fig_loss_landscape` | 2D reconstruction loss landscapes with fit trajectories | PhotonSim ROOT sweep (`--root-base`) |
| `fig_calib_convergence` | the 19-parameter joint calibration over iterations | none (closure fit) |
| `fig_calib_loss_geometry` | the 2D calibration loss with descent and Gauss-Newton paths | none (closure fit) |

The `--root-base` default of the tracking scans is an S3DF directory (`utils/paths.py`); the
500-event inputs are not distributed ([Inputs](#inputs)).

## Redraw the published figures

The reconstructed scans and every other figure's cached inputs are the `lucid-paper-results-v1`
release asset (299 MB unpacked, `tools/assets/lucid-paper-results-v1.manifest`). Install it and
redraw, on a CPU:

```bash
python scripts/download_data.py --paper          # into analysis/paper/output/data/
python analysis/paper/fig_nrays.py --plot-results    # likewise each fig_*.py
```

Its index says how the files were made: at commit `7a913a09`, the scans at 16 candidates per ray
where the shipped recipe now uses 8, and the event count of each point.

## Contract

Most figure scripts take the same three modes:

```
python fig_<name>.py                    # reproduce end-to-end (generate locally, then plot)
python fig_<name>.py --generate-data    # (re)generate the figure's data only
python fig_<name>.py --plot-results     # plot from whatever data already exists
```

Two figures only draw shipped inputs and run in one step with their own flags (`--help`):
`fig_photon_event_display` and `fig_siren_profile`.

Data-generation backend, for `fig_nrays`, `fig_energy_scan`, `fig_geometry_scan` (and
`fig_compute_performance`):

```
--backend local   (default)  reconstruct a small subset in-process — no cluster; it reads
                             the 500-event inputs from --root-base (default
                             paths.S3DF_ROOT_FILES), so off S3DF pass --root-base.
--backend s3df               submit the published production to the S3DF GPU partitions
                             via utils/submit_job.py (nrays 500 events/point;
                             energy and geom 100).
```

Options shared across scripts: `--out <dir>` (the figure's output dir). The three reconstruction
scans also take `--events N`, `--root-base <dir>` and their sweep (`fig_nrays --tags 5k,50k,250k`,
`fig_energy_scan --energies`, `fig_geometry_scan --sensors`); the calibration, t0 and
loss-landscape figures take `--data-dir`. Every other flag is per figure: `python fig_<name>.py --help`.

Each step writes `<script>.json` next to what it wrote: the arguments it ran with and
`utils/provenance.run_provenance()` (commit, dirty flag, package versions). The plot step writes
it into the figure directory; the data step into the cache directory, where the cache does not
already carry its provenance (the study `.h5` and the calibration `.npz` do). The rendering style
is module constants in each script, so the commit fixes it.

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
    paths.py            default output locations (repo-local) and the S3DF_* constants: the
                          productions, the PhotonSim inputs, the image, the user site and the
                          binds, the only site-specific paths in analysis/paper
    events.py           shared event preparation for the two charge-display figures
    plotting.py         thin wrappers over utils/ovv_plots
    ovv_plots.py        the nrays/energy/sensors/convergence plot modes
    -- calibration --
    calibration.py      ★ single source of truth for the CALIBRATION campaign: detector,
                          truth model, source layout, CALIB, LANDSCAPE
    calib_run.py        ★ runs one seed of the 19-parameter joint optical fit, through
                          lucid.fitting. Typed arguments, no environment variables; the
                          seeds share one compiled forward.
    calib_plots.py      rendering for both calibration figures (plot-only)
    landscape.py        the 2D Neyman surface, defined once for both stages below
    compute_2d_neyman.py / combine_2d.py / traj_2d_neyman.py
                        the three data stages behind fig_calib_loss_geometry
    provenance.py       commit, dirty flag and package versions a run records
    p68_evolution.py    per-iteration p68 convergence of one study file (a CLI)
  geometries/           generated SK_like geometry JSONs (git-ignored)
  output/               generated data + figures (default target; git-ignored)
```

`utils/studies.py` defines the study settings and every sweep once; the figure
scripts build their configs from it at run time (on `--backend s3df` they are written to
`<data-dir>/configs/` for provenance). Nothing else may define a sweep or a
recipe value — that duplication is what let the scripts drift from the published data.

## Tracking studies

The single-track reconstruction campaigns run the three-start Fisher-GN of
[`lucid.fitting`](../../lucid/fitting) across the **nrays**, **energy**, and **geom** axes for
**muon** and **electron**. The reconstruction itself is the method in
[`utils/pipeline.py`](utils/pipeline.py); the campaign it was validated in
(`scripts/campaign_recon/` and its `RESULTS.md`) is archived in the release asset
`studies-archive-geometry-v2-2026-10` ([`docs/studies.md`](../../docs/studies.md)).

Per event: load the PhotonSim photons, place the event (`lucid.sources.placement`: a uniform
rotation about the recorded primary vertex, the vertex uniform in the fiducial cylinder, redrawn
until the track end point is contained), make realistic per-PMT (charge, first-arrival time) with
**2.1 ns TTS** and the **`Abe_2013`** charge-resolution model, seed from two complementary
vertices plus their fusion (first-light grid ‖ time-multilateration) each with a cone direction,
pick one start by its data loss (prefer the first-light-grid seed unless another beats it by
`pick_margin`), then run one `fit_track` keeping the full per-iteration trajectory. There is no
scalar charge normalisation.

> **The study settings live in exactly one place: [`utils/studies.py`](utils/studies.py)**
> (with the library's `FIT_RECIPE`, `MODEL_RECIPE` and `SEED_RECIPE`). The figure scripts build
> every config from it at run time, so the commands below reproduce the data by construction.
> Do not hand-write study configs — that is how the campaigns drifted from the published set in
> the first place.

### Reconstruction defaults

The published working point: the library's `FIT_RECIPE`, `MODEL_RECIPE` and `SEED_RECIPE`
(`lucid.fitting`) plus the study settings in `utils/studies.py` (`TTS`, `PIPELINE`), applied by
every figure script with no flags needed:

| | |
|---|---|
| TTS | **2.1 ns**, with a matched likelihood kernel (the model's sigma is the config's `tts`) |
| charge resolution | **`Abe_2013`** per-PMT charge model on the pseudo-data |
| Fisher-GN | autodiff Fisher metric, `lr=4.0`, `nkeys=8`, `steps=150` |
| vertices | contained, random true t0 (±15 ns), uncorrected |
| propagation grid | `PIPELINE['grid']`: 80 × 120 × 80 cells, `max_candidates_per_ray=8` |
| reflection | `scalar_mix` — a **code** default (`lucid/simulation/simulator.py`), not a config key |

Every study config is complete: `studies.base_config` writes the whole working point (the
`model`, `fit` and `seed` records in full, and `PIPELINE`), and `TrackingPipeline` refuses a
config that lacks a key. `tests/paper/test_paper_configs.py` pins the published values, so an
edit to a library recipe fails the paper's gate rather than moving its figures.

### What gets logged

One **HDF5 per config**, `<output>/<config-name>.h5`:

- **root attrs** — the full config JSON plus provenance (`n_sensors`, `n_rays`,
  `energy_nominal_MeV`, `particle`, `study`, `root_file`, `lucid_commit`, `git_dirty`, `versions`,
  host, timestamps) and completion (`n_events_requested`, `n_events_done`, `finished`). A failed
  event fails the job and leaves `finished` unset; the figures refuse such a file, a missing
  point, or one still being written.
- **`events/ev%04d`** — per event, as 9-vectors `[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]`
  (`lucid.fitting.track_from_vec9` / `vec9_dir` decode them):
  - `truth_vec9` — **true** track parameters (exact, from the applied transform), and `tdir` its
    unit direction;
  - `seedA_vec9`, `seedB_vec9`, `seedF_vec9` — the **initial guesses** (first-light-grid,
    time-multilateration, and their fusion — the validated third start);
  - `traj_win` `(steps+1, 9)` — **reco parameters at each optimization step** (picked seed), and
    `fit_vec9` the fit's Polyak readout.

  Plus the gradient norms (`gnorm_win`), the fit's errors (`fit_err` = `lucid.fitting.track_errors`:
  `[vtx cm, dir deg, dE MeV, dt0 ns]`), `which` picked seed (0=A, 1=B, 2=fused),
  `best_iter_win`, `losses` (each start's pre-fit data loss), `n_rejected`, and metadata attrs
  (`energy_true`, `n_hit`, `q_tot`, `seconds`).

### Output layout

Each figure writes under its own data dir, `utils/paths.data_dir(figure, backend)`:
`output/data/<figure>` in the repo by default (`LUCID_PAPER_OUTPUT` moves the `output/` root),
`paths.S3DF_DATA / 'paper' / <figure>` with `--backend s3df`:

```
<data-dir>/
├── configs/                  # the exact configs submitted (provenance)
├── <config-name>/
│   └── <config-name>.h5      # one HDF5 per config, e.g. nrays_mu_250k/
├── logs/                     # SLURM stdout/err
└── slurm/                    # generated sbatch scripts
```

### Running a campaign on S3DF

One command per study. It builds the configs from `utils/studies.py`, writes them next to the
output for provenance, and queues one GPU job per config. Geometry configs for the geom axis are
generated automatically — there is no prerequisite step. The S3DF side (the image, the CUDA user
site, the SLURM partitions) is in the
[cluster deployment guide](../../docs/guides/production/clusters.md#s3df-slac-slurm).

```bash
cd LUCiD
python analysis/paper/fig_nrays.py         --generate-data --backend s3df
python analysis/paper/fig_energy_scan.py   --generate-data --backend s3df
python analysis/paper/fig_geometry_scan.py --generate-data --backend s3df
```

Each defaults to the published sweep and event count:

- **nrays** — `n_rays` ∈ {5k,10k,25k,50k,100k,150k,250k} at 1000 MeV / SK_like,
  muon **and** electron, 500 events/point.
- **energy** — 400..1800 MeV (step 100) at a flat 250k rays, muon, 100 events/point. Two
  passes: `time_weight=1.0` throughout plus `0.75` at/above 1600 MeV (`studies.W_CROSSOVER`);
  the figure draws each energy from the appropriate pass. Reads `<E>MeV_500events.root`
  ([below](#energy-sweep-photonsim-data)).
- **geom** — SK_like PMT count 2k..18k (step 2k) at 1000 MeV, muon, 100 events/point.

Narrow the scope with `--tags`, `--energies`, `--sensors`, `--events`, `--root-base`.
Then plot:

```bash
python analysis/paper/fig_nrays.py --plot-results --backend s3df
```

### Submitting configs by hand (debugging)

The campaign commands above call this for you; reach for it directly only when re-submitting a
single failed point. The container's baked jaxlib is CPU-only, so the CUDA build is layered in via
`APPTAINERENV_PYTHONUSERBASE` + `--nv`
(["Running the JAX stack on a GPU"](../../docs/guides/production/clusters.md#running-the-jax-stack-on-a-gpu));
the checkout's own `lucid` wins on `sys.path`. `D` is the figure's S3DF data dir,
`paths.S3DF_DATA / 'paper' / 'nrays'`:

```bash
D=<S3DF_DATA>/paper/nrays
# one config
python analysis/paper/utils/submit_job.py \
    --config $D/configs/nrays_mu_250k.json --output $D --submit
# every config in a directory (one job each)
python analysis/paper/utils/submit_job.py --config $D/configs --output $D --submit
```

Omit `--submit` for a dry run (writes the sbatch scripts to `<output>/slurm/` without queuing).
Useful flags: `--partition` (default `turing`; also `ampere`/`ada`/`hopper`; the campaign
commands above submit to `ampere`, `utils/run.py`), `--time`, `--mem`, `--image` (default
`paths.S3DF_IMAGE`), `--env-base` (default `paths.S3DF_ENV_BASE`).

Run one config locally (e.g. to debug), still inside the container, with the CUDA user site of
`paths.S3DF_ENV_BASE`:

```bash
APPTAINERENV_PYTHONUSERBASE=<S3DF_ENV_BASE> \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs <S3DF_IMAGE> \
    /opt/conda/bin/python3 analysis/paper/utils/run_study.py \
        --config $D/configs/nrays_mu_250k.json \
        --output /tmp/trk_test --events 0,1
```

### Energy-sweep PhotonSim data

The energy study needs `<E>MeV_500events.root` for 400..1800 MeV (the generator covers
300..2100), both particles. These are made by the PhotonSim monoenergetic / individual-photon
recipe (the same one behind the distributed files), on CPU nodes (`milano`):

```bash
# write macros + sbatch, inspect first
python analysis/paper/utils/generate_data.py
# write + queue everything, dropping stale files in the target dir
python analysis/paper/utils/generate_data.py --submit --clean
```

Output: `<base>/<mu-|e->/<E>MeV_500events.root`, `--base` defaulting to `paths.S3DF_ROOT_FILES`, which
is also the default `--root-base` of every scan in `utils/studies.py`. Muon macros disable the
muon-decay processes; electrons are stable so that block is dropped. The nrays and geom studies
reuse the 1000 MeV file from this same tree.

### Reading the output

```python
import h5py, numpy as np
with h5py.File('nrays_mu_250k.h5', 'r') as f:
    print(dict(f.attrs))                      # config + provenance
    ev = f['events/ev0000']
    true  = ev['truth_vec9'][:]               # [E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]
    guess = ev['seedA_vec9'][:]               # initial guess (first-light-grid seed)
    steps = ev['traj_win'][:]                 # (steps+1, 9) reco per step
    print('vtx/dir/dE/dt0 err:', ev['fit_err'][:])
```

### Minimal local validation (no SLURM)

Write a small config from a published one, so events are placed as in the study:

```bash
python3 -c "import json; from analysis.paper.utils import studies
c = studies.named_config('nrays_mu_250k', n_events=2); c['model']['n_rays'] = 30000; c['fit']['steps'] = 15
json.dump(c, open('/tmp/trk_smoke.json', 'w'))"
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs <S3DF_IMAGE> \
    /opt/conda/bin/python3 analysis/paper/utils/run_study.py \
        --config /tmp/trk_smoke.json --output /tmp/trk_smoke --events 0,1
```

This confirms the pipeline runs end-to-end and writes a readable HDF5 before launching a full
campaign.

## Calibration figures

Both are closure fits: truth data from the same differentiable simulator, fitted from a displaced
start. The estimator is `lucid.fitting`'s (Neyman residual, per-PMT gains profiled); the recipe of
each figure is one dict in `utils/calibration.py` (`CALIB`, `LANDSCAPE`), and every `.npz` records
the recipe it was computed with. No environment variable reaches either figure's physics.

- **`fig_calib_convergence`** — the joint 19-parameter fit over iterations, 3 seeds.
  `utils/calib_run.py` runs each seed as `run_seed(**CALIB)` through `lucid.fitting` and records
  the commit, package versions and device mapping (`pmap` on >= 7 devices, else serial; the two
  agree to ~1e-3, not bit for bit);
  `utils/calib_plots.convergence` renders. The paper's run is 1200 Gauss-Newton steps per seed on
  the SK-like geometry; `--steps` shortens it for a smoke test (not the paper's figure). The
  trajectories are Polyak-smoothed at plot time (`--polyak-window`), so changing the window needs
  no re-run.
- **`fig_calib_loss_geometry`** — `log10 L` over (scattering, absorption length) with one laser,
  gradient-descent and Gauss-Newton paths overlaid. `compute_2d_neyman.py` (sharded) →
  `combine_2d.py` → `traj_2d_neyman.py` → `utils/calib_plots.loss_geometry`. Every stage takes
  `--data-dir`; the surface is `utils/landscape.py` built from `LANDSCAPE`, and the trajectories
  read the recipe from the surface's own npz, so the paths lie on the plotted surface. The shards need
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (set by the driver), and the driver refuses to combine an
  incomplete or stale set of shards.

## Inputs

Local reproduction reconstructs from PhotonSim ROOT files
(`<root-base>/<mu-|e->/<E>MeV_<N>events.root`). They are not distributed: point `--root-base`
at a directory that has them. `analysis/paper/utils/generate_data.py` regenerates them on S3DF
(Slurm on milano, the PhotonSim binary of the LUCiD image); elsewhere, run the same macros with
your own PhotonSim. To redraw the published figures without them, install
`lucid-paper-results-v1` ([above](#redraw-the-published-figures)).
