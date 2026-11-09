# LUCiD Production Pipeline

Turns a PhotonSim particle-based ROOT file into a four-file HDF5 dataset
(`sensor/`, `hits/`, `step/`, `labl/`). Full schema in
[`docs/reference/dataset-schema.md`](../../docs/reference/dataset-schema.md).

## Layout

```
production/
├── dataprod/        the dataset line: lucid-run-job, the event loops, generators, fan-out,
│                    configs/ and scripts/ (submit, monitor, verify, resubmit, reports)
├── siren_inputs/    SIREN training inputs: Stage 1 scan, smax/ (Stage 0), train_siren/ (Stage 3)
├── qa/              validations and trigger_study, over a written dataset
├── derived_views/   relabel / de-pile views of a dataset
└── platform/        cluster adapters (SLURM, NERSC, HTCondor), user_paths.sh and its templates
```

The applications share only `platform/` and the library; the dataset format they read and
write is `lucid.dataset`.

## Config blocks

Configs live under `dataprod/configs/<block>/NN_name.json` (2-digit, numbering
restarts per block). The blocks, their splits and nominal counts are listed in
[`docs/guides/production/index.md`](../../docs/guides/production/index.md#config-blocks);
what each `GeV` config generates is in the
[container guide](../../docs/guides/production/container.md#3-the-configs). The
particle bomb (`GeV/01_pbomb.json`) draws from {e-, e+, mu-, mu+, pi+, pi-, pi0,
gamma}; its `description` says why both charges are in the pool.

A config declares `nominal_train` / `nominal_test`; the detector comes from
`-D` (or the config's own `detector`, for the self-contained blocks). The
fan-out writes `OUTPUT_BASE/<detector>/[_test/]<block>/[<split>/]config_NN/`
and sizes jobs from `nominal / (target_seconds_per_job / seconds_per_event)`;
see [`dataprod/README.md`](dataprod/README.md).

## Example

One job of a dataset config, on a PhotonSim file a previous stage left in the output directory
(as `output_job_<job-id>.root`, the id zero-padded to six digits, e.g. `output_job_000001.root`):

```bash
lucid-run-job --config lucid/production/dataprod/configs/GeV/02_mu_smoke.json \
    --detector SK_WAND \
    --output-dir /path/to/dataset_root/ --job-id 1 --n-events 2 --skip-photonsim
```

Writes `{output}/{sensor,hits,step,labl}/wc_*_0000.h5`. A GeV config needs
`--detector`, a `*_WAND` one: any other (e.g. `SK_like`) carries no digitizer or
trigger, and the job stops before PhotonSim unless the config states its
readout (`"digitizer": "basic"`, `"trigger": null`). A config that declares its
own `detector` needs no flag; one given must match. The flags are in
[`docs/reference/cli.md`](../../docs/reference/cli.md).

**Units:** meters, nanoseconds, MeV throughout.

## Medium / scintillation (WbLS)

The medium is set by the **detector config** (`--detector`), not by the
dataprod config's `material` field. The WAND detectors and `SK_like` are
water → Cherenkov-only output (every `emission_process` row is 0). To
produce scintillation, run against a WbLS detector (`SK_like_wbls`); the log
then reports `geometry: cylinder / wbls` and `emission_process` carries both
Cherenkov (0) and scintillation (1) hits. `SK_like_wbls` carries no digitizer
or trigger block either, so a GeV config on it must state both.

## Digitizer / hit-making

The digitizer presets (`basic`, `ski`, `hk`), where the `"digitizer"` and
`"trigger"` blocks are read from, and the overrides are described in
[`docs/reference/digitizer-and-trigger.md`](../../docs/reference/digitizer-and-trigger.md).
Non-`basic` models can record **multiple digits per sensor**, and dark noise is
a labelled source in the `hits.h5` decomposition (`emission_process = 2`,
`particle_idx = -1`); see
[`docs/reference/dataset-schema.md`](../../docs/reference/dataset-schema.md).

**Pile-up** is digitized **cross-vertex**: all vertices' per-photon deposits are
pooled in absolute time and windowed once, so light from different vertices that
overlaps on a PMT within the integration window merges into one digit (with the
decomposition attributing its charge across the contributing vertices). `basic`
pile-up reproduces the legacy per-sensor first-arrival + summed charge.

## Supernova bursts (sntools)

`primary_source: "supernova"` injects supernova-burst neutrino interactions
in water instead of running GENIE. [sntools](https://github.com/SNEWS2/sntools)
Monte-Carlo–samples the interactions (IBD, ν-e ES, νe/ν̄e CC on ¹⁶O) from a
supernova flux × cross section; `dataprod/generators/supernova.py` converts them to the same
rooTracker file GENIE produces, so PhotonSim consumes them through the
existing `/gun/genieInput` path (no PhotonSim change). Datasets get
`source_type = 2` and true `neutrino_pdg` / `neutrino_energy` in `labl/`.

Each event's **interaction channel** is preserved: `labl/…/per_interaction`
carries `interaction_channel` (sntools' integer NUANCE code; -1 outside
supernova datasets). sntools' channel code has no
slot in the rooTracker, so `dataprod/generators/supernova.py` writes it to a per-event sidecar
aligned 1:1 with the rooTracker entries, and the runner stamps it into the
labl truth after the writer completes.

A dataset fans out into one `<model>/<ordering>/` subcase per
`supernova.models × supernova.orderings` — mass ordering maps to sntools'
`AdiabaticMSW_NMO` / `AdiabaticMSW_IMO` transformation.

`supernova.orderings` is **required**: a non-empty list, each entry a label
(`NMO`/`NO`/`normal`, `IMO`/`IO`/`inverted`, `noosc`/`none` → `NoTransformation`)
or one of sntools' `--transformation` names (`NoTransformation`,
`AdiabaticMSW_NMO`, `AdiabaticMSW_IMO`, `SNEWPY-CompleteExchange`,
`SNEWPY-NonAdiabaticMSWH`, `SNEWPY-TwoFlavorDecoherence_NMO`,
`SNEWPY-TwoFlavorDecoherence_IMO`, `SNEWPY-ThreeFlavorDecoherence`). There is no
default (sntools' own is `NoTransformation`), and the singular `supernova.ordering`
is refused. Each job runs one entry: the fan-out passes it as `--sn-ordering`; a
direct `lucid-run-job` may omit the flag only when the list has one entry. Every
batch records it as `config/sn_ordering` (the label) and `sn_transformation`.


```
<detector>/SN/config_NN/                     # SN block is flat (no train/test)
  <model>/<ordering>/{sensor,hits,step,labl}/
```

The shipped config, `configs/SN/01_supernova_50kpc.json` (comments added):

```json
"primary_source": "supernova",
"supernova": {
  "detector": "HyperK",        // sntools fiducial → realistic event count
  "distance_kpc": 50.0,
  "channels": "all",           // or ["ibd","es","o16e","o16eb"]
  "direction": [0.0, 0.0, 1.0],
  "cap_events": 25,            // omit/null for a full realistic burst
  "models": [
    {"name": "nakazato_20Msun_shen_trev100", "format": "SNEWPY-Nakazato_2013",
     "flux_file": "nakazato-shen-z0.02-t_rev100ms-s20.0.fits"}
  ],
  "orderings": ["NMO"]
}
```

so it writes `<OUT>/HK/SN/config_01/nakazato_20Msun_shen_trev100/NMO/`. One job = one burst realization (a distinct
seed); `/run/beamOn` is the sntools event count. A `SNEWPY-<Model>` format names
that model's SNEWPY data file; other sntools formats (`gamma`, `nakazato`
ASCII) take a repo-relative `flux_file` path, and none ships. Requires
sntools+snewpy on `SN_ENV_BASE` (see
`platform/templates/user_paths.nersc.sh.template`).
