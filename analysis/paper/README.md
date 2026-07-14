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
                             GPU partitions via analysis/paper/submit_job.py.
```

Common options: `--events N`, `--tags 5k,50k,250k`, `--root-base <dir>`, `--out <dir>`.

Figures that only re-plot existing reconstruction output (e.g. convergence, Polyak)
have no `--generate-data` step and are correspondingly simpler.

## Layout

```
analysis/paper/
  fig_<name>.py         one figure (or a small group) per script
  run_study.py          reconstruct one study config -> per-event HDF5 (the worker)
  submit_job.py         submit configs to S3DF SLURM (one GPU job per config)
  make_configs.py       generate the nrays/energy/geom study config JSONs
  make_geometries.py    generate SK_like geometry JSONs at varying PMT counts
  data_generation/      generate the energy-sweep PhotonSim ROOTs
  configs/              generated study config JSONs
  geometries/           generated SK_like geometry JSONs (git-ignored)
  utils/
    paths.py            default output locations (repo-local; s3df area opt-in)
    studies.py          build study configs with an event-count knob (local vs full)
    run.py              run_local() in-process + submit_s3df() SLURM wrapper
    plotting.py         thin wrappers over utils/ovv_plots
    pipeline.py         TrackingPipeline — the reconstruction engine the figures build on
    ovv_plots.py        the nrays/energy/geom/convergence plot modes
    p68_evolution.py    per-iteration p68 convergence helper (used by ovv_plots)
  output/               generated data + figures (default target; git-ignored)
```

`paper/` holds both the figure entry points (`fig_*.py`) and the tracking engine they build
on — the reconstruction pipeline (`utils/pipeline.py`), the plot modes (`utils/ovv_plots.py`),
and the campaign runners (`run_study.py` / `submit_job.py`).

## Inputs

Local reproduction reconstructs from PhotonSim ROOT files
(`<root-base>/<mu-|e->/<E>MeV_<N>events.root`). Point `--root-base` at a directory that
has them; generating them from scratch is `analysis/paper/data_generation/generate_data.py`.

## Reconstruction defaults

Contained vertices, random true t0 (±15 ns), TTS = 2.0 ns with a matched likelihood
kernel, uncorrected — the paper's final working point (see `utils/studies.py`).
