# Track-fitting studies on S3DF

How to run the single-track reconstruction campaigns (the three-start Fisher-GN of
[`lucid.fitting`](../lucid/fitting)) across the **nrays**, **energy**, and **geom** axes for
**muon** and **electron**, on S3DF SLURM. All code lives in
this directory (`analysis/paper/`); this is the S3DF runbook — see also
the [S3DF deployment guide](../../docs/guides/production/deploy-s3df.md).

The reconstruction itself is the validated recipe from
[`scripts/campaign_recon/`](../scripts/campaign_recon) (RESULTS.md): per event, load the
PhotonSim photons, place the gun randomly in the fiducial volume, make realistic per-PMT
(charge, first-arrival time) with **2.1 ns TTS** and the **`Abe_2013`** charge-resolution
model, seed from two complementary vertices plus their fusion
(charge-grid ‖ time-multilateration) each with a cone direction, then run
`fit_track_multistart` keeping the full per-iteration trajectory. `tot_n_scale = 1.0` for both
particles (no scalar charge norm).

> **The recipe lives in exactly one place: [`utils/studies.py`](utils/studies.py).** The
> figure scripts build every config from it at run time, so following this runbook
> reproduces the published data by construction. Do not hand-write study configs —
> that is how the campaigns drifted from the published set in the first place.

## What gets logged

One **HDF5 per config**, `<output>/<config-name>.h5`:

- **root attrs** — the full config JSON plus provenance (`n_sensors`, `n_rays`,
  `energy_nominal_MeV`, `particle`, `study`, `root_file`, `lucid_commit`, host, timestamps).
- **`events/ev%04d`** — per event, in the 7-tuple order `(x, y, z, phi, theta, t0, E)`:
  - `truth_phys` — **true** track parameters (exact, from the applied transform);
  - `seedA_phys`, `seedB_phys`, `seedF_phys` — the **initial guesses** (charge-grid,
    time-multilateration, and their fusion — the validated third start);
  - `traj_win_phys` `(niters+1, 7)` — **reco parameters at each optimization step** (winning seed).

  Plus the raw 9-vectors (`truth_vec9`, `seedA/B/F_vec9`, `fit_vec9`, `fitA/B/F_vec9`), each
  seed's full trajectory (`trajA`, `trajB`, `trajF`, `traj_win`) and gradient norms (`gnorm*`),
  errors (`fit_err`, `seedA/B/F_err` = `[vtx cm, dir deg, dE MeV, dt0 ns]`), `which` winning
  seed (0=A, 1=B, 2=fused), `best_iter*`, `losses`, and metadata attrs (`energy_true`, `n_hit`,
  `q_tot`, `seconds`).

Config knobs live in `utils/studies.py` (the recipe) and `utils/pipeline.py:DEFAULT_CONFIG`
(the fallback for anything a study config omits).

## Output layout

Each figure writes under its own data dir (`utils/paths.py`; the S3DF area is
`/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/paper/<figure>`):

```
<data-dir>/
├── configs/                  # the exact configs submitted (provenance)
├── <config-name>/
│   └── <config-name>.h5      # one HDF5 per config, e.g. nrays_mu_250k/
├── logs/                     # SLURM stdout/err
└── slurm/                    # generated sbatch scripts
```

## 1. Run a campaign

One command per study. It builds the configs from `utils/studies.py`, writes them next to
the output for provenance, and queues one GPU job per config. Geometry configs for the
geom axis are generated automatically — there is no prerequisite step.

```bash
cd LUCiD
python analysis/paper/fig_nrays.py         --generate-data --backend s3df
python analysis/paper/fig_energy_scan.py   --generate-data --backend s3df
python analysis/paper/fig_geometry_scan.py --generate-data --backend s3df
```

Each defaults to the published sweep and event count:

- **nrays** — `n_rays` ∈ {5k,10k,25k,50k,100k,150k,250k,500k} at 1000 MeV / SK_like,
  muon **and** electron, 500 events/point.
- **energy** — 400..1800 MeV (step 100) at a flat 250k rays, muon, 100 events/point. Two
  passes: `time_weight=1.0` throughout plus `0.75` at/above 1600 MeV (`studies.W_CROSSOVER`);
  the figure draws each energy from the appropriate pass. Reads `<E>MeV_500events.root` (§3).
- **geom** — SK_like PMT count 2k..18k (step 2k) at 1000 MeV, muon, 100 events/point.

Narrow the scope with `--tags`, `--energies`, `--sensors`, `--events`, `--root-base`.
Then plot:

```bash
python analysis/paper/fig_nrays.py --plot-results --backend s3df
```

## 2. Submitting configs by hand (debugging)

The campaign commands above call this for you; reach for it directly only when
re-submitting a single failed point. The container's baked jaxlib is CPU-only, so the CUDA
build is layered in via `APPTAINERENV_PYTHONUSERBASE` + `--nv` (QUICKSTART_S3DF §"Running
the JAX stack on a GPU"); the checkout's own `lucid` wins on `sys.path`.

```bash
D=/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/paper/nrays
# one config
python analysis/paper/utils/submit_job.py \
    --config $D/configs/nrays_mu_250k.json --output $D --submit
# every config in a directory (one job each)
python analysis/paper/utils/submit_job.py --config $D/configs --output $D --submit
```

Omit `--submit` for a dry run (writes the sbatch scripts to `<output>/slurm/` without queuing).
Useful flags: `--partition` (default `turing`; also `ampere`/`ada`/`hopper`), `--time`,
`--mem`, `--image`, `--env-base`.

Run one config locally (e.g. to debug), still inside the container:

```bash
APPTAINERENV_PYTHONUSERBASE=/sdf/data/neutrino/cjesus/python_envs/lucid \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 analysis/paper/utils/run_study.py \
        --config $D/configs/nrays_mu_250k.json \
        --output /tmp/trk_test --events 0,1
```

## 3. Generate the energy-sweep PhotonSim data (prerequisite for the energy study)

The energy study needs `<E>MeV_500events.root` for 400..1800 MeV (the generator covers
300..2100), both particles. These are made
by the PhotonSim monoenergetic / individual-photon recipe (the same one behind the distributed
files), on CPU nodes (`milano`):

```bash
# write macros + sbatch, inspect first
python analysis/paper/utils/generate_data.py
# write + queue everything, dropping stale files in the target dir
python analysis/paper/utils/generate_data.py --submit --clean
```

Output: `/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water/<mu-|e->/<E>MeV_500events.root`.
Muon macros disable the muon-decay processes; electrons are stable so that block is dropped.
nrays and geom studies reuse the 1000 MeV file from this same tree.

## Reading the output

```python
import h5py, numpy as np
with h5py.File('nrays_mu_250k.h5', 'r') as f:
    print(dict(f.attrs))                      # config + provenance
    ev = f['events/ev0000']
    true  = ev['truth_phys'][:]               # (x,y,z,phi,theta,t0,E)
    guess = ev['seedA_phys'][:]               # initial guess (charge-grid seed)
    steps = ev['traj_win_phys'][:]            # (niters+1, 7) reco per step
    print('vtx/dir/dE/dt0 err:', ev['fit_err'][:])
```

## Minimal local validation (no SLURM)

```bash
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 analysis/paper/utils/run_study.py \
        --config <small-config.json> --output /tmp/trk_smoke --events 0,1
```

Use a small config (e.g. `n_rays` 30000, `gn.niters` 15) pointed at an existing ROOT to confirm
the pipeline runs end-to-end and writes a readable HDF5 before launching a full campaign.
