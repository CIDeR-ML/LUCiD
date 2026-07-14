# Track-fitting studies on S3DF

How to run the single-track reconstruction campaigns (the three-start Fisher-GN of
[`lucid.fitting`](../lucid/fitting)) across the **nrays**, **energy**, and **geom** axes for
**muon** and **electron**, on S3DF SLURM. All code lives in
this directory (`analysis/paper/`); this is the S3DF runbook — see also
the [S3DF deployment guide](../../docs/guides/production/deploy-s3df.md).

The reconstruction itself is the validated recipe from
[`scripts/campaign_recon/`](../scripts/campaign_recon) (RESULTS.md): per event, load the
PhotonSim photons, place the gun randomly in the fiducial volume, make realistic per-PMT
(charge, first-arrival time) with 2.5 ns TTS, seed from two complementary vertices plus their fusion
(charge-grid ‖ time-multilateration) each with a cone direction, then run
`fit_track_multistart` keeping the full per-iteration trajectory. `tot_n_scale = 1.0` for both
particles (no scalar charge norm).

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

Config knobs live in `pipeline.DEFAULT_CONFIG`; a study config only overrides what varies.

## Output layout

```
/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/<particle>/<study>/
├── config_00.h5 ...          # one HDF5 per config
├── logs/                     # SLURM stdout/err
└── slurm/                    # generated sbatch scripts
```

## 1. Generate the study configs

```bash
cd LUCiD
python analysis/paper/make_geometries.py     # geom study: SK_like at 2k..20k PMTs (step 1k)
python analysis/paper/make_configs.py         # all studies × {muon,electron}, 100 events/config
# subset / override event count:
python analysis/paper/make_configs.py --studies nrays --particles muon --n-events 50
```

Configs land in `analysis/paper/configs/<study>/<particle>/config_NN.json`.

- **nrays** — sweeps predictor photon budget `n_rays` ∈ {5k,10k,25k,50k,100k,150k,250k} at
  1000 MeV / SK_like.
- **energy** — sweeps 300..2100 MeV (step 100); reads `<E>MeV_500events.root` (see §3).
- **geom** — sweeps SK_like PMT count 2k..20k (step 1k) at 1000 MeV.

## 2. Submit the reconstruction jobs (GPU, `turing`)

One GPU job per config. The container's baked jaxlib is CPU-only, so the CUDA build is layered
in via `APPTAINERENV_PYTHONUSERBASE` + `--nv` (QUICKSTART_S3DF §"Running the JAX stack on a
GPU"); the checkout's own `lucid` wins on `sys.path`.

```bash
OUT=/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking
# one config
python analysis/paper/submit_job.py \
    --config analysis/paper/configs/nrays/muon/config_00.json \
    --output $OUT/muon/nrays --submit
# every config in a directory (one job each)
python analysis/paper/submit_job.py \
    --config analysis/paper/configs/nrays/muon \
    --output $OUT/muon/nrays --submit
```

Omit `--submit` for a dry run (writes the sbatch scripts to `<output>/slurm/` without queuing).
Useful flags: `--partition` (default `turing`; also `ampere`/`ada`/`hopper`), `--time`,
`--mem`, `--image`, `--env-base`.

Run one config locally (e.g. to debug), still inside the container:

```bash
APPTAINERENV_PYTHONUSERBASE=/sdf/data/neutrino/cjesus/python_envs/lucid \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 analysis/paper/run_study.py \
        --config analysis/paper/configs/nrays/muon/config_06.json \
        --output /tmp/trk_test --events 0,1
```

## 3. Generate the energy-sweep PhotonSim data (prerequisite for the energy study)

The energy study needs `<E>MeV_500events.root` for 300..2100 MeV, both particles. These are made
by the PhotonSim monoenergetic / individual-photon recipe (the same one behind the distributed
files), on CPU nodes (`milano`):

```bash
# write macros + sbatch, inspect first
python analysis/paper/data_generation/generate_data.py
# write + queue everything, dropping stale files in the target dir
python analysis/paper/data_generation/generate_data.py --submit --clean
```

Output: `/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water/<mu-|e->/<E>MeV_500events.root`.
Muon macros disable the muon-decay processes; electrons are stable so that block is dropped.
nrays and geom studies reuse the 1000 MeV file from this same tree.

## Reading the output

```python
import h5py, numpy as np
with h5py.File('config_06.h5', 'r') as f:
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
    /opt/conda/bin/python3 analysis/paper/run_study.py \
        --config <small-config.json> --output /tmp/trk_smoke --events 0,1
```

Use a small config (e.g. `n_rays` 30000, `gn.niters` 15) pointed at an existing ROOT to confirm
the pipeline runs end-to-end and writes a readable HDF5 before launching a full campaign.
