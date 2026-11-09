# Paper figures

Each figure of the paper is one script, `analysis/paper/fig_<name>.py`. The published working
points live in `analysis/paper/utils/studies.py` (reconstruction) and
`analysis/paper/utils/calibration.py` (calibration), with the library's `MODEL_RECIPE`,
`SEED_RECIPE` and `FIT_RECIPE`.

## Redraw the published figures

The reconstructed scans and the cached inputs of every other figure are the
`lucid-paper-results-v1` release asset (299 MB unpacked). Install it, then plot, on a CPU:

```bash
python scripts/download_data.py --paper              # into analysis/paper/output/data/
python analysis/paper/fig_nrays.py --plot-results    # and likewise for each fig_*.py
```

## Regenerate

Most scripts take three modes:

```bash
python analysis/paper/fig_<name>.py                   # generate the data, then plot
python analysis/paper/fig_<name>.py --generate-data   # data only
python analysis/paper/fig_<name>.py --plot-results    # plot what exists
```

The display figures and the calibration figures run from the configs and the example data. The
three reconstruction scans (`fig_nrays`, `fig_energy_scan`, `fig_geometry_scan`) and the loss
landscapes need 500-event PhotonSim samples that are not distributed (`--root-base`), and run as
GPU jobs on S3DF (`--backend s3df`).

Every script lists its options with `--help`. The table of scripts, their inputs, the output
layout and the S3DF campaign commands are in
[`analysis/paper/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/analysis/paper/README.md).
