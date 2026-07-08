# Producing datasets

LUCiD has two very different jobs, and it is easy to end up in the wrong part
of the docs looking for the other one.

- **Explore, reconstruct, or calibrate** using the bundled, pre-trained SIREN
  emitter weights. This is the fast path — no GEANT4, no PhotonSim, nothing to
  build. Everything you need ships with the repo or comes down with
  `./scripts/download_data.sh`. Start at the
  [Getting Started quickstart](../../getting-started/quickstart.md).
- **Produce a labeled dataset** — new GEANT4-simulated events, saved as HDF5,
  for training a SIREN emitter or for other downstream analysis. This needs an
  external simulation engine (**PhotonSim**), built from source or pulled as
  part of the published container. This section is about that path.

If you only want to simulate an event, look at a display, run a
reconstruction, or run a calibration, you do not need anything on this page —
those all consume the bundled SIREN weights already in `data/`.

## I want to…

| I want to… | Go to |
|---|---|
| Run the tutorial notebooks | [Getting Started quickstart](../../getting-started/quickstart.md) |
| Simulate an event and look at a display | [Getting Started quickstart](../../getting-started/quickstart.md) (`examples/hello_simulate.py`, `00_quickstart` notebook) |
| Produce a labeled dataset on my own machine | [Local production](local.md) |
| Produce a dataset with Docker (macOS or no Apptainer) | [Docker production](docker.md) |
| Produce at scale on a cluster (S3DF / NERSC / LXPLUS) | [S3DF](deploy-s3df.md), [NERSC](deploy-nersc.md), [LXPLUS](deploy-lxplus.md) — see [how the cluster layer works](cluster-abstraction.md) |
| Train a new SIREN emitter from GEANT4 output | [SIREN training inputs](siren-training-inputs.md) |
| Post-process a dataset I already produced | [Working with LUCiD data](../../reference/working-with-data.md) |

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
build PhotonSim yourself; pick [Docker](docker.md) if you are on macOS or don't
want to install those dependencies; pick a cluster runbook
([S3DF](deploy-s3df.md), [NERSC](deploy-nersc.md), [LXPLUS](deploy-lxplus.md))
once you are ready to generate a full training or production dataset rather
than a handful of test events.

Production configs live under `lucid/production/configs/<block>/`, grouped
into blocks (`GeV`, `Solar`, `SN`, `Test`) that each bundle a set of particle
and interaction choices with their own train/test event counts — pick a
config from there (or write your own alongside them) for any of the paths
above.
