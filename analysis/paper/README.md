# Paper figures

Self-contained, reproducible scripts for the tracking-paper figures. Each figure is
one script (`fig_<name>.py`) with a uniform command-line contract; shared logic lives
in `utils/`.

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
--backend s3df               submit the full production (500 events/point) to the S3DF
                             GPU partitions via analysis/tracking/submit_job.py.
```

Common options: `--events N`, `--tags 5k,50k,250k`, `--root-base <dir>`, `--out <dir>`.

Figures that only re-plot existing reconstruction output (e.g. convergence, Polyak)
have no `--generate-data` step and are correspondingly simpler.

## Layout

```
analysis/paper/
  fig_<name>.py         one figure (or a small group) per script
  utils/
    paths.py            default output locations (repo-local; s3df area opt-in)
    studies.py          build study configs with an event-count knob (local vs full)
    run.py              run_local() in-process + submit_s3df() SLURM wrapper
    plotting.py         thin wrappers over analysis/tracking/ovv_plots
  output/               generated data + figures (default target; git-ignored)
```

`paper/` orchestrates and imports the `analysis/tracking/` engine (`TrackingPipeline`,
`submit_job`, `ovv_plots`) — it does not duplicate it.

## Inputs

Local reproduction reconstructs from PhotonSim ROOT files
(`<root-base>/<mu-|e->/<E>MeV_<N>events.root`). Point `--root-base` at a directory that
has them; generating them from scratch is `analysis/tracking/data_generation/generate_data.py`.

## Reconstruction defaults

Contained vertices, random true t0 (±15 ns), TTS = 2.0 ns with a matched likelihood
kernel, uncorrected — the paper's final working point (see `utils/studies.py`).
