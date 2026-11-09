# Producing datasets

LUCiD has two very different jobs, and it is easy to end up in the wrong part
of the docs looking for the other one.

- **Explore, reconstruct, or calibrate** using the bundled, pre-trained SIREN
  emitter weights. This is the fast path — no GEANT4, no PhotonSim, nothing to
  build. Everything you need ships with the repo or comes down with
  `python scripts/download_data.py` (see [Install](../../getting-started/install.md)).
  Start at the
  [Getting Started quickstart](../../getting-started/quickstart.md).
- **Produce a labeled dataset** — new GEANT4-simulated events, saved as HDF5,
  for training a SIREN emitter or for other downstream analysis. This needs an
  external simulation engine (**PhotonSim**), built from source or pulled as
  part of the published container. This section is about that path.

If you only want to simulate an event, look at a display, run a
reconstruction, or run a calibration, you do not need anything on this page —
those all run on the SIREN weights `scripts/download_data.py` installs in `data/`.

## I want to…

| I want to… | Go to |
|---|---|
| Simulate an event and look at a display | [Getting Started quickstart](../../getting-started/quickstart.md) (`examples/hello_simulate.py`) |
| Produce a dataset from the container (Apptainer or Docker) | [Container production](container.md) |
| Produce a labeled dataset on my own machine | [Local production](local.md) |
| Produce at scale on a cluster (S3DF / NERSC / LXPLUS) | [Cluster deployment](clusters.md) — see [how the cluster layer works](cluster-abstraction.md) |
| Train a new SIREN emitter from GEANT4 output | [SIREN training inputs](siren-training-inputs.md) |
| Read or post-process a dataset I already produced | [Dataset schema: reading a batch](../../reference/dataset-schema.md#reading-a-batch) |

## What production writes

Whichever path you take, a production run ends with the same thing: one batch
of **four parallel HDF5 files** (`sensor/`, `hits/`, `step/`, `labl/`), sharing
per-event indexing, plus whatever digitizer and trigger model the dataset
config selected — the digitizer turns raw per-photon deposits into
per-sensor charge/time readout, and an optional trigger window decides which
stretches of time are kept. The full field-by-field layout is in the
[dataset schema reference](../../reference/dataset-schema.md); how the
digitizer and trigger models work is in
[digitizer & trigger](../../reference/digitizer-and-trigger.md). For the
`lucid-run-job` command-line flags, see the [CLI reference](../../reference/cli.md).

## Before you start

Every path in this section needs the **PhotonSim** binary (GEANT4-based photon
transport) — either built locally against your own GEANT4 + ROOT install, or
available pre-built inside the published container image. Pick
[local](local.md) if you already have GEANT4/ROOT/GENIE installed and want to
build PhotonSim yourself; pick [the container](container.md) if you are on macOS or don't
want to install those dependencies; pick the [cluster runbook](clusters.md)
(S3DF, NERSC, LXPLUS) once you are ready to generate a full training or production dataset rather
than a handful of test events.

## Config blocks

Production configs live under `lucid/production/dataprod/configs/<block>/NN_name.json`
(2-digit, numbering restarts per block). The JSONs are the source of the counts
below; pick a config from there (or write your own alongside them) for any of
the paths above. What each `GeV` config generates is tabulated in the
[container guide](container.md#3-the-configs).

| block | configs | splits (nominal events) | detector |
|---|---|---|---|
| `GeV` | `01_pbomb` (particle bomb); `02_mu`, `03_pi`, `04_e`, `06_pi0`, `07_gamma` (single particles); `02_mu_fast`, `02_mu_metrics`, `02_mu_smoke` (small `02_mu` variants) | `01`: train 1,000,000 + test 50,000. `02`–`07`: test only, 50,000 (the variants: 60, 500, 10) | none declared: `-D SK_WAND` or `-D HK_WAND` |
| `Solar` | `01_e_low_energy` | train 1,000,000 + test 50,000 | `HK` (declared) |
| `SN` | `01_supernova_50kpc` | flat, 5 bursts | `HK` (declared) |
| `Test` | `01_pile_up_bombs` | flat, 500 | `HK` (declared) |

The particle bomb is config 01 because it is the training set: it mixes the
species and multiplicities a real event mixes. The single-particle configs are
test-only, one evaluation set per species in the bomb's pool.

## Where a batch run writes

The fan-out (`lucid.production.dataprod.fanout`, driven by the cluster
runbooks) writes one cell per config and split:

```
<OUTPUT_BASE_PATH>/<detector>/[_test/]<block>/[<split>/]config_NN/
    submit_job_NNNNNN.{sbatch,sub}, job_NNNNNN-<jobid>.{out,err}
    sensor/ hits/ step/ labl/           one wc_<sub>_NNNN.h5 batch per job
```

- `_test/` appears only with `-t` (one job of 2 events per split), so a test
  wave never touches the production tree.
- `<split>` (`train`, `test`) appears for `GeV` and `Solar`; `SN` and `Test`
  are flat. Supernova nests `<model>/<ordering>/` below the cell, one per entry
  of the required `supernova.orderings`, and a monoenergetic config `<E>MeV/`.
- `NNNN = job_id − 1`; each cell is one dataset. Train and test use disjoint
  master seeds.
