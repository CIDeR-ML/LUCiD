# Producing datasets

This section is for producing new labelled datasets from GEANT4: PhotonSim simulates the
events, LUCiD simulates the detector and writes HDF5. Simulating, reconstructing and calibrating
with the shipped emitters need none of it ([quickstart](../../getting-started/quickstart.md)).

Every path needs the PhotonSim binary, built yourself or inside the published container image
`ghcr.io/cider-ml/lucid` (GEANT4, ROOT, GENIE, PhotonSim and LUCiD).

| I want to | Go to |
|---|---|
| run a job with the container (Apptainer or Docker), no build | [From the container](container.md) |
| build PhotonSim and run jobs on my own machine | [On your machine](local.md) |
| produce a full dataset on S3DF, NERSC or LXPLUS | [On a cluster](clusters.md) |
| train a new SIREN emitter from PhotonSim output | [Training a SIREN emitter](siren-training-inputs.md) |
| read what a job wrote | [Dataset schema](../../reference/dataset-schema.md) |

One job is one `lucid-run-job` call ([flags](../../reference/cli.md#lucid-run-job)). It writes
one batch: four HDF5 files (`sensor/`, `hits/`, `step/`, `labl/`) with the readout of the
[digitizer and trigger](../../reference/digitizer-and-trigger.md) the dataset uses.

## Config blocks

Dataset configs are `lucid/production/dataprod/configs/<block>/NN_name.json`; copy one to make
your own. Event counts are the configs' `nominal_train` and `nominal_test`.

| block | configs | nominal events | detector |
|---|---|---|---|
| `GeV` | `01_pbomb` (particle bomb); `02_mu`, `03_pi`, `04_e`, `06_pi0`, `07_gamma` (single particles); `02_mu_fast`, `02_mu_metrics`, `02_mu_smoke` (small `02_mu` variants) | `01`: train 1,000,000, test 50,000. `02` to `07`: test 50,000 (variants 60, 500, 10) | none: pass `SK_WAND` or `HK_WAND` |
| `Solar` | `01_e_low_energy` | train 1,000,000, test 50,000 | `HK` (declared) |
| `SN` | `01_supernova_50kpc` | 5 bursts | `HK` (declared) |
| `Test` | `01_pile_up_bombs` | 500 | `HK` (declared) |

The particle bomb is the training set: 1 to 5 particles per event, mixed as in real
interactions. The single-particle configs are test sets, one per species in the bomb. What each
`GeV` config generates is listed in [the container guide](container.md#3-the-configs).

## Where a batch run writes

The cluster fan-out writes one directory per config and split:

```
<OUTPUT_BASE_PATH>/<detector>/[_test/]<block>/[<split>/]config_NN/
    submit_job_NNNNNN.{sbatch,sub}, job_NNNNNN-<jobid>.{out,err}
    sensor/ hits/ step/ labl/           one wc_<kind>_NNNN.h5 per job
```

- `_test/` appears only for a test wave (`-t`: one job of 2 events per split).
- `<split>` (`train`, `test`) appears for `GeV` and `Solar`; `SN` and `Test` have none. A
  supernova config adds `<model>/<ordering>/` below the directory, a monoenergetic one `<E>MeV/`.
- `NNNN = job_id − 1`. Each directory is one dataset; train and test use different seeds.
