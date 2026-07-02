# analysis/tracking

Single-track reconstruction studies (the three-start Fisher-GN of [`lucid.fitting`](../../lucid/fitting))
run across **nrays / energy / geom** axes for **muon / electron** on S3DF SLURM.

Full runbook: [`docs/TRACKING_STUDIES.md`](../../docs/TRACKING_STUDIES.md).

| file | role |
|------|------|
| `pipeline.py` | core: build detector + simulators + `ReconModel` from a config, reconstruct one event (generalises `scripts/campaign_recon/worker.py`). |
| `run_study.py` | loop a config's events, log everything to one HDF5. |
| `seed_study.py` | standalone **initial-guess (seed) study** — seedA/seedB/fused vs truth (no GN), HDF5 + summary + plots. |
| `submit_job.py` | one GPU sbatch (`turing`) per config, layered CUDA-jax env. |
| `make_configs.py` | generate the study config JSONs → `configs/<study>/<particle>/`. |
| `make_geometries.py` | SK_like geometry JSONs at 2k..20k PMTs → `geometries/`. |
| `data_generation/generate_data.py` | PhotonSim energy-sweep ROOTs (300..2100 MeV, mu-/e-, CPU/`milano`). |

Output → `/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/<particle>/<study>/`.
