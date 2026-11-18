# Production

Turns PhotonSim (Geant4) output into LUCiD datasets: four parallel HDF5 files per batch
(`sensor/`, `hits/`, `step/`, `labl/`), described in the
[dataset schema](../../docs/reference/dataset-schema.md). The workflow and its configs are in
[producing datasets](../../docs/guides/production/index.md). Units are metres, nanoseconds and
MeV.

| directory | contents |
|---|---|
| `dataprod/` | the dataset line: `lucid-run-job`, the event loops, the generators (Geant4 macros, GENIE, sntools), the cluster fan-out, the dataset configs and the submit/monitor/verify scripts ([README](dataprod/README.md)) |
| `siren_inputs/` | PhotonSim scans for SIREN training: the s_max scan (`smax/`), the training-input scan, the t0 fit (`t0/`), and training sweeps (`train_siren/`) ([README](siren_inputs/README.md)) |
| `qa/` | validation of a written dataset (`validations.py`), the trigger study, job-log parsing |
| `derived_views/` | new labelings of an existing dataset without re-simulating ([README](derived_views/README.md)) |
| `platform/` | the cluster adapters (Slurm on S3DF and NERSC, HTCondor on LXPLUS), `run_in_container.sh`, and the `user_paths.sh` templates |

The applications share only `platform/` and the library; the dataset format is `lucid.dataset`.

## One job

A dataset config is `dataprod/configs/<block>/NN_name.json`. One job of one config, on a
PhotonSim file already in the output directory as `output_job_<job-id>.root` (the id
zero-padded to six digits, e.g. `output_job_000001.root`):

```bash
lucid-run-job --config lucid/production/dataprod/configs/GeV/02_mu_smoke.json \
    --detector SK_WAND --output-dir /path/to/dataset_root/ --job-id 1 --n-events 2 --skip-photonsim
```

This writes `{output}/{sensor,hits,step,labl}/wc_*_0000.h5` (batch index = job id − 1). The
flags are in the [CLI reference](../../docs/reference/cli.md).

The `GeV` configs declare no detector, so they need `--detector`. The digitizer and trigger
come from the dataset config or else from the detector's physics config; of the bundled
detectors only `SK_WAND` and `HK_WAND` carry both. On any other detector (`SK_like`, `HK`, ...)
the job stops before PhotonSim unless the config states its readout (`"digitizer": "basic"`,
`"trigger": null`). The `Solar`, `SN` and `Test` configs declare their own detector (`HK`) and
readout; a `--detector` given with them must match.

## Medium

The medium comes from the detector (`--detector`), not from the dataset config's `material`
field. The WAND detectors and `SK_like` are water, so their output is Cherenkov only (every
`emission_process` is 0). For scintillation use a WbLS detector such as `SK_like_wbls`:
`emission_process` is then 0 for Cherenkov and 1 for scintillation hits. `SK_like_wbls` has no
digitizer or trigger either, so a `GeV` config on it must state both.

## Digitizer

The digitizer models (`basic`, `ski`, `hk`), where the `"digitizer"` and `"trigger"` blocks
are read from, and their overrides are in
[digitizer and trigger](../../docs/reference/digitizer-and-trigger.md). Models other than
`basic` can record several digits per sensor. Dark noise is a labelled source in `hits.h5`
(`emission_process = 2`, `particle_idx = -1`).

Pile-up is digitized across vertices: the deposits of all vertices in an event are pooled in
time and digitized once, so light from different vertices that lands on one PMT within the
integration window becomes one digit, with its charge attributed to each contributing vertex in
`hits.h5`.

## Supernova bursts

`"primary_source": "supernova"` replaces GENIE with [sntools](https://github.com/SNEWS2/sntools),
which samples supernova-neutrino interactions in water (IBD, ν-e elastic scattering, νe and ν̄e
CC on ¹⁶O) from a flux and cross sections. `dataprod/generators/supernova.py` converts them to
the rooTracker format GENIE writes, and PhotonSim reads them through `/gun/genieInput`. These
datasets have `source_type = 2` and the true `neutrino_pdg` and `neutrino_energy` in `labl/`.
`labl/.../per_interaction/interaction_channel` holds sntools' NUANCE channel code (-1 in other
datasets); the rooTracker format has no slot for it, so it travels in a per-event sidecar file.

A dataset runs one subcase per entry of `supernova.models` × `supernova.orderings`.
`supernova.orderings` is required: a non-empty list whose entries are labels (`NMO`, `NO`,
`normal`, `IMO`, `IO`, `inverted`; `noosc` or `none` for `NoTransformation`) or sntools
`--transformation` names (`NoTransformation`, `AdiabaticMSW_NMO`, `AdiabaticMSW_IMO`,
`SNEWPY-CompleteExchange`, `SNEWPY-NonAdiabaticMSWH`, `SNEWPY-TwoFlavorDecoherence_NMO`,
`SNEWPY-TwoFlavorDecoherence_IMO`, `SNEWPY-ThreeFlavorDecoherence`). The singular
`supernova.ordering` is refused. Each job runs one ordering: the fan-out passes it as
`--sn-ordering`, and a direct `lucid-run-job` may omit the flag only when the list has one
entry. Every batch records it as `config/sn_ordering` and `sn_transformation`.

The shipped config, `dataprod/configs/SN/01_supernova_50kpc.json`, has this block (comments
added):

```json
"primary_source": "supernova",
"supernova": {
  "detector": "HyperK",        // sntools fiducial volume, for a realistic event count
  "distance_kpc": 50.0,
  "channels": "all",           // or a list: ["ibd", "es", "o16e", "o16eb"]
  "direction": [0.0, 0.0, 1.0],
  "cap_events": 25,            // omit or null for a full burst
  "models": [
    {"name": "nakazato_20Msun_shen_trev100", "format": "SNEWPY-Nakazato_2013",
     "flux_file": "nakazato-shen-z0.02-t_rev100ms-s20.0.fits"}
  ],
  "orderings": ["NMO"]
}
```

It writes `<OUT>/HK/SN/config_01/nakazato_20Msun_shen_trev100/NMO/{sensor,hits,step,labl}/`
(the `SN` block has no train/test split). One job is one burst with its own seed. A
`SNEWPY-<Model>` format names that model's SNEWPY data file, which must already be in the SNEWPY
cache; other sntools formats (`gamma`, `nakazato` ASCII) take a repository-relative `flux_file`,
and none is shipped. sntools and snewpy must be installed in `SN_ENV_BASE` (see
`platform/templates/user_paths.nersc.sh.template`).
