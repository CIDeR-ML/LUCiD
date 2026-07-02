# JUNO-like WbLS scintillation analysis

Three scripts for the JUNO-like WbLS sphere (`config/JUNO_wbls_{geom,physics}_config.json`):
a data-vs-prediction Cherenkov/scintillation event display, and a differentiability
showcase for the scintillation light yield `S` (ph/MeV).

| script | what it does | output |
|--------|--------------|--------|
| `juno_cher_scint_fraction.py` | Side-by-side **DATA-like vs PREDICTION-like** event display of the signed Cherenkov-vs-scintillation asymmetry `a = (Q_cher − Q_scint)/(Q_cher + Q_scint)` per sensor, on a diverging blue→white→red scale. Data = real PhotonSim ROOT photons (Cherenkov + scintillation expanded from the dE/dx segments) injected into the `is_data` simulator, one process at a time; prediction = forward sim of the same track, per process. Caches the per-process charges. | `figures/juno_cher_scint_fraction.{pdf,png}` + `data/juno_cher_scint_charges.npz` |
| `juno_S_loss_scan.py` | Scans the scintillation yield `S`, **re-running the forward simulator at every point** with `S` as a runtime `DetectorParams` field, and takes `dNLL/dS` by autodiff **through the simulator** (one jitted `jax.value_and_grad`, compiled once). Loss = per-sensor Poisson NLL of the total charge vs the cached data-like event. | `data/juno_S_loss_scan.npz` |
| `juno_S_loss_plot.py` | Plots the cached scan: `ΔNLL(S)` and `dNLL/dS(S)` with the gradient zero crossing (the best-fit `Ŝ`). Pure matplotlib, no simulation. | `figures/juno_S_loss_scan.{pdf,png}` |

## Run order

`juno_S_loss_scan.py` needs the charge cache written by `juno_cher_scint_fraction.py`,
so run them in this order:

```bash
python3 analysis/juno_wbls/juno_cher_scint_fraction.py   # display + charge cache
python3 analysis/juno_wbls/juno_S_loss_scan.py            # S scan (needs the cache)
python3 analysis/juno_wbls/juno_S_loss_plot.py            # figure from the cached scan
```

All caches/figures land under `analysis/juno_wbls/{data,figures}/` (created on demand).

## Environment (S3DF)

Run inside the LUCiD container; the display script needs plotly+kaleido and the scan
benefits from a GPU, both provided by the layered user-site env
(see `docs/QUICKSTART_S3DF.md` §"Running the JAX stack on a GPU"):

```bash
APPTAINERENV_PYTHONUSERBASE=/sdf/data/neutrino/cjesus/python_envs/lucid \
APPTAINERENV_PYTHONPATH="" \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/python3 analysis/juno_wbls/juno_cher_scint_fraction.py
```

The scripts import the repo they live in by default (`LUCID_DIR` env var overrides;
`/opt/LUCiD` is the last-resort fallback inside the container).

## Inputs & knobs

- **Data ROOT** (`juno_cher_scint_fraction.py`): a PhotonSim file carrying per-photon
  Cherenkov + dE/dx segments; default
  `/sdf/data/neutrino/cjesus/CIDER/ROOT_files/water/mu-/1000MeV_100events.root`
  (`ROOT`/`ENTRY` constants at the top of the script). Track: 1 GeV muon from the
  origin, θ=π/4, φ=π/6 — matched between the data injection and the forward sim.
- `juno_cher_scint_fraction.py`: `FORCE_SIM=1` re-simulates even if the charge cache
  exists (layout tweaks don't need it); `EYE`, `PANEL_GAP`, `CBAR_GUTTER` tune the montage.
- `juno_S_loss_scan.py`: `NPHOT` (default 1e6 forward photons), `NPOINTS` (default 11),
  `SREL_LO`/`SREL_HI` (scan window as fractions of the config truth `S0`, default
  0.5–1.5), `SEED`.

## What the S scan demonstrates

`S` enters the forward simulation as a runtime parameter, so the Poisson-NLL loss is
differentiable end-to-end: the plotted gradient is `jax.grad` of the loss **through the
whole photon simulation**, not a finite difference or an analytic rescaling. The
gradient's zero crossing recovers the truth yield (`Ŝ ≈ S0`), which is the point of the
figure: scintillation-yield calibration by gradient descent is possible in this framework.
