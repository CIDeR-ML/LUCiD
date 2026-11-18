# Paper figures

The scripts that make the figures of *End-to-end Differentiable Calibration and Reconstruction
for Optical Particle Detectors* ([arXiv:2602.24129](https://arxiv.org/abs/2602.24129)). Each
figure is one script, `fig_<name>.py`; shared code is in `utils/`. Everything they write goes
under `output/` (git-ignored; `LUCID_PAPER_OUTPUT` moves it).

## Redraw the published figures

The reconstructed scans and every other figure's cached inputs are the `lucid-paper-results-v1`
release asset (299 MB unpacked). Install it from the repository root and redraw, on a CPU:

```bash
python scripts/download_data.py --paper              # into analysis/paper/output/data/
python analysis/paper/fig_nrays.py --plot-results    # likewise each fig_*.py
```

`fig_photon_event_display` and `fig_siren_profile` draw the example data
(`python scripts/download_data.py`) and need nothing from the asset.

The asset's `INDEX.md` records how its files were made: at commit `7a913a09`, and with 16
candidates per ray in the propagation grid where `utils/studies.py` now uses 8. A scan rerun with
this code is the same study at 8 candidates, not a byte-identical copy of the published files.

## Figures

| script | draws | input |
|---|---|---|
| `fig_charge_displays` | data-like vs predicted charge on a cylinder, a sphere and a box | `data/water/muon/1000MeV_5events.root` |
| `fig_cylinder_2d_displays` | unrolled-cylinder charge and first-arrival time of one muon | event 5 of `data/water/muon/1500MeV_6events.root` |
| `fig_photon_event_display` | 3D Cherenkov photons coloured by creation time | `data/water/muon/1000MeV_5events.root` |
| `fig_detector_gallery` | the four geometry families with scale bars | configs only |
| `fig_siren_profile` | SIREN Cherenkov emission profiles | the shipped emitters |
| `fig_t0_correction` | photon arrival delay vs distance, with the shipped t0 fit | the emitters; the points need a PhotonSim sample (`--generate-data`, or `--data-dir` for a full scan) |
| `fig_juno_wbls` | Cherenkov vs scintillation on a WbLS sphere, and the loss along a scintillation parameter | a PhotonSim ROOT file (`--root`; default entry 0 of the 1 GeV muon slice) |
| `fig_compute_performance` | forward and fit-step time vs number of rays, per K | the shipped emitters; a GPU for the published numbers (`--full`) |
| `fig_nrays` | tracking resolution vs number of rays, and the fit convergence | PhotonSim ROOT files (`--root-base`) |
| `fig_energy_scan` | tracking resolution vs energy | PhotonSim ROOT files (`--root-base`) |
| `fig_geometry_scan` | tracking resolution vs PMT count | PhotonSim ROOT files (`--root-base`) |
| `fig_loss_landscape` | 2D reconstruction loss landscapes with fit trajectories | PhotonSim ROOT files (`--root-base`) |
| `fig_calib_convergence` | the 19-parameter joint calibration over iterations | none (closure fit) |
| `fig_calib_loss_geometry` | the 2D calibration loss with descent and Gauss-Newton paths | none (closure fit) |

## Regenerate

Every script except `fig_photon_event_display` and `fig_siren_profile` (which draw in one step)
has three modes:

```bash
python analysis/paper/fig_<name>.py                   # generate the data, then plot
python analysis/paper/fig_<name>.py --generate-data   # generate the data only
python analysis/paper/fig_<name>.py --plot-results    # plot the data already there
```

All take `--out <dir>` for the figure. The reconstruction scans also take `--events N`,
`--root-base <dir>`, their sweep (`fig_nrays --tags 5k,50k,250k`, `fig_energy_scan --energies`,
`fig_geometry_scan --sensors`) and `--backend`:

- `--backend local` (default) reconstructs in-process, 20 events per point, and `fig_nrays` a
  subset of its ray counts.
- `--backend s3df` submits the published production to the S3DF GPU partitions, one job per
  config ([below](#running-a-campaign-on-s3df)).

The calibration, t0 and loss-landscape figures take `--data-dir`. Other flags are per figure:
`python analysis/paper/fig_<name>.py --help`.

Each run writes `fig_<name>.json` next to its output: the arguments and
`utils/provenance.run_provenance()` (commit, dirty flag, package versions). The plot style is
fixed in each script.

### Inputs

The tracking scans read PhotonSim ROOT files `<root-base>/<mu-|e->/<E>MeV_500events.root`,
which are not distributed. The default `--root-base` is the S3DF directory
`paths.S3DF_ROOT_FILES` (`utils/paths.py`, which holds every site-specific path in this
directory); elsewhere, pass `--root-base`. `utils/generate_data.py` makes them on S3DF with the
PhotonSim of the LUCiD image (500 events per energy, 300 to 2100 MeV, muon and electron, Slurm
on `milano`; muon decay is disabled):

```bash
python analysis/paper/utils/generate_data.py                    # write macros and sbatch scripts
python analysis/paper/utils/generate_data.py --submit --clean   # queue them; --clean removes files outside the sweep
```

## Tracking studies

The scans run the published reconstruction of `lucid.fitting` (`utils/pipeline.py`) over the
number of rays, the energy and the PMT count. Per event: load the PhotonSim photons; place the
event (`lucid.sources.placement`: a uniform rotation about the primary vertex, the vertex uniform
in the fiducial cylinder, redrawn until the track end is contained); make per-PMT charge and
first-arrival time with 2.1 ns TTS and the `Abe_2013` charge-resolution model; build seeds A
(first-light grid), B (time multilateration) and F (their fusion); pick one by data loss; run
one `fit_track`, keeping the trajectory.

`utils/studies.py` defines the study settings and every sweep; the figure scripts build their
configs from it at run time. Do not write study configs by hand.

### Reconstruction defaults

The library's `FIT_RECIPE`, `MODEL_RECIPE` and `SEED_RECIPE` plus `TTS` and `PIPELINE` in
`utils/studies.py`:

| | |
|---|---|
| TTS | 2.1 ns; the likelihood's timing sigma is the same |
| charge resolution | `Abe_2013` on the pseudo-data |
| Fisher-Gauss-Newton | autodiff Fisher metric, `lr=4.0`, `nkeys=8`, `steps=150` |
| vertices | contained, true t0 uniform in ±15 ns |
| propagation grid | 80 × 120 × 80 cells, `max_candidates_per_ray=8` |
| reflection | `scalar_mix`, the simulator default |

`studies.base_config` writes the whole working point into each config, and `TrackingPipeline`
refuses a config that lacks a key. `tests/paper/test_paper_configs.py` pins these values.

### Running a campaign on S3DF

One command per study builds its configs, writes them to `<data-dir>/configs/`, and queues one
GPU job per config on `ampere` (geometry files for the PMT-count scan are generated
automatically). The image, the CUDA user site and the partitions are described in
[clusters](../../docs/guides/production/clusters.md).

```bash
python analysis/paper/fig_nrays.py         --generate-data --backend s3df
python analysis/paper/fig_energy_scan.py   --generate-data --backend s3df
python analysis/paper/fig_geometry_scan.py --generate-data --backend s3df
python analysis/paper/fig_nrays.py         --plot-results  --backend s3df
```

The defaults are the published sweeps:

- rays: 5k, 10k, 25k, 50k, 100k, 150k and 250k at 1000 MeV in SK_like, muon and electron, 500
  events per point;
- energy: 400 to 1800 MeV in steps of 100 at 250k rays, muon, 100 events per point, run with
  `time_weight=1.0` and again with 0.75 at and above 1600 MeV (`studies.W_CROSSOVER`); the figure
  takes each energy from the matching pass;
- PMT count: SK_like with 2k to 18k PMTs in steps of 2k, 1000 MeV muon, 100 events per point.

`--tags`, `--energies`, `--sensors`, `--events` and `--root-base` narrow a campaign.

Each figure's data directory is `utils/paths.data_dir(figure, backend)`: `output/data/<figure>/`
locally, `paths.S3DF_DATA/paper/<figure>/` with `--backend s3df`. It holds `configs/` and one
directory per config with `<config-name>.h5`, `logs/` and `slurm/`.

### Submitting configs by hand

To resubmit one failed point. The image's `jaxlib` is CPU-only, so the job layers in a CUDA
build through `APPTAINERENV_PYTHONUSERBASE` and `--nv`.

```bash
D=<S3DF_DATA>/paper/nrays
python analysis/paper/utils/submit_job.py \
    --config $D/configs/nrays_mu_250k.json --output $D/nrays_mu_250k --submit
python analysis/paper/utils/submit_job.py --config $D/configs --output $D --submit   # every config, one output dir
```

Without `--submit` the sbatch scripts are written to `<output>/slurm/` and not queued. Other
flags: `--partition` (default `turing`), `--time`, `--mem`, `--image` (default
`paths.S3DF_IMAGE`), `--env-base` (default `paths.S3DF_ENV_BASE`); see `--help`.

To run one config in the container on a GPU node:

```bash
APPTAINERENV_PYTHONUSERBASE=<S3DF_ENV_BASE> \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs <S3DF_IMAGE> \
    /opt/conda/bin/python3 analysis/paper/utils/run_study.py \
        --config $D/configs/nrays_mu_250k.json --output <out-dir> --events 0,1
```

### Minimal local validation

A two-event run of a published config at reduced size, to check the pipeline end to end:

```bash
python3 -c "import json; from analysis.paper.utils import studies
c = studies.named_config('nrays_mu_250k', n_events=2); c['model']['n_rays'] = 30000; c['fit']['steps'] = 15
json.dump(c, open('trk_smoke.json', 'w'))"
python3 analysis/paper/utils/run_study.py --config trk_smoke.json --output trk_smoke --events 0,1
```

The config's `root_file` must exist: off S3DF, pass `root_base=` to `named_config`.

### Study files

One HDF5 per config, `<config-name>.h5`.

Root attributes: `config_json` (the full config), `name`, `particle`, `study`,
`energy_nominal_MeV`, `n_rays`, `root_file`, `n_sensors`, `hostname`, `lucid_commit`,
`git_dirty`, `versions`, `started`, and on completion `n_events_requested`, `n_events_done` and
`finished`. A failed event fails the job and leaves `finished` unset; the figures refuse such a
file.

`events/ev%04d`, one group per event. Track parameters are 9-vectors
`[E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]` (`lucid.fitting.track_from_vec9` and `vec9_dir`
decode them).

| dataset | content |
|---|---|
| `truth_vec9`, `tdir` | the true track and its unit direction |
| `seedA_vec9`, `seedB_vec9`, `seedF_vec9` | the three starts: first-light grid, time multilateration, their fusion |
| `losses` | the data loss of each start before the fit |
| `traj_win` | `(steps+1, 9)`: the fitted start's parameters at each step |
| `gnorm_win` | the gradient norm at each step |
| `fit_vec9` | the fitted track |
| `fit_err` | `lucid.fitting.track_errors`: vertex (cm), direction (deg), energy (MeV), t0 (ns) |

Event attributes: `ev`, `energy_true`, `which` (the picked start: 0 = A, 1 = B, 2 = F),
`best_iter_win`, `n_rejected`, `n_hit`, `q_tot`, `seconds`.

```python
import h5py
with h5py.File('nrays_mu_250k.h5', 'r') as f:
    ev = f['events/ev0000']
    truth, start, traj = ev['truth_vec9'][:], ev['seedA_vec9'][:], ev['traj_win'][:]
    print(ev['fit_err'][:], ev.attrs['which'])
```

`utils/p68_evolution.py <file.h5>` prints the 68% quantile of each error per iteration.

## Calibration figures

Both are closure fits: data from the same differentiable simulator, fitted from a displaced
start with `lucid.fitting`'s estimator (Neyman residual, per-PMT gains profiled). Each figure's
recipe is one dict in `utils/calibration.py` (`CALIB`, `LANDSCAPE`), and every `.npz` it writes
records the recipe.

- `fig_calib_convergence`: the joint 19-parameter fit over iterations, three seeds, 1200
  Gauss-Newton steps each on the SK-like geometry. `utils/calib_run.py` runs each seed and
  records the commit, package versions and device mapping; with enough GPUs the forward is
  `pmap`ped, otherwise run serially (the two agree to about 1e-3). `--steps` shortens the run
  for a smoke test. The trajectories are Polyak-smoothed when plotted (`--polyak-window`).
- `fig_calib_loss_geometry`: `log10 L` over scattering and absorption length for one laser, with
  gradient-descent and Gauss-Newton paths. The data stages are `utils/compute_2d_neyman.py`
  (sharded, `--shards`), `utils/combine_2d.py` and `utils/traj_2d_neyman.py`, each taking
  `--data-dir`; the trajectories read the recipe from the surface's `.npz`, so they lie on the
  plotted surface. The driver sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` for the shards and
  refuses to combine an incomplete or stale set.
