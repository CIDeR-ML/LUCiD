# LUCiD Production Pipeline

Turns a PhotonSim particle-based ROOT file into a v3 four-file HDF5 dataset
(`sensor/`, `inst/`, `seg/`, `labl/`). Full schema in
[`docs/LUCID_DATASET.md`](../../docs/LUCID_DATASET.md).

## Example

```bash
python -m lucid.production.generate_events_with_particles \
    --root-file /path/to/photonsim_output.root \
    --output     /path/to/dataset_root/ \
    --dataset-name my_run_2026_04_20 \
    --apply-smearing
```

Writes `{output}/{sensor,hits,step,labl}/wc_*_0000.h5`. Use `--help` for
the full flag list (`--batch-size`, `--n-events`, `--master-seed`,
`--physics-config`, ...).

**Units:** meters, nanoseconds, MeV throughout.

## Medium / scintillation (WbLS)

The medium is set by the **detector config** (`--detector`), not by the
dataprod config's `material` field. The default `--detector SK_like` is
water → Cherenkov-only output (every `emission_process` row is 0). To
produce scintillation, run with `--detector SK_like_wbls`; the log then
reports `geometry: cylinder / wbls` and `emission_process` carries both
Cherenkov (0) and scintillation (1) hits.

## Supernova bursts (sntools)

`primary_source: "supernova"` injects supernova-burst neutrino interactions
in water instead of running GENIE. [sntools](https://github.com/SNEWS2/sntools)
Monte-Carlo–samples the interactions (IBD, ν-e ES, νe/ν̄e CC on ¹⁶O) from a
supernova flux × cross section; `run_supernova.py` converts them to the same
rooTracker file GENIE produces, so PhotonSim consumes them through the
existing `/gun/genieInput` path (no PhotonSim change). Datasets get
`source_type = 2` and true `neutrino_pdg` / `neutrino_energy` in `labl/`.

Each event's **interaction channel** is preserved: `labl/…/per_interaction`
carries `interaction_channel` (sntools' integer NUANCE code) and `channel`
(string — `"ibd"`, `"es"`, `"o16e"`, `"o16eb"`). sntools' channel code has no
slot in the rooTracker, so `run_supernova.py` writes it to a per-event sidecar
aligned 1:1 with the rooTracker entries, and the runner stamps it into the
labl truth after the v3 writer completes.

A dataset fans out into one `<model>/<ordering>/` subcase per
`supernova.models × supernova.orderings` — mass ordering maps to sntools'
`AdiabaticMSW_NMO` / `AdiabaticMSW_IMO` transformation:

```
config_000090/
  analytic_accretion/{NMO,IMO}/{sensor,hits,step,labl}/
  analytic_hot/{NMO,IMO}/{sensor,hits,step,labl}/
```

Config block (see `configs/dataprod_90_supernova.json`):

```json
"primary_source": "supernova",
"supernova": {
  "detector": "SuperK",        // sntools fiducial → realistic event count
  "distance_kpc": 10.0,
  "channels": "all",           // or ["ibd","es","o16e","o16eb"]
  "cap_events": 50,            // omit/null for a full realistic burst
  "models": [
    {"name": "analytic_accretion", "format": "gamma",
     "flux_file": "lucid/production/configs/sn_fluxes/analytic_burst_gamma.txt"}
  ],
  "orderings": ["NMO", "IMO"]
}
```

One job = one burst realization (a distinct seed); `/run/beamOn` is the
sntools event count. For physics, set each model's `format` to
`SNEWPY-<Model>` (e.g. `SNEWPY-Nakazato_2013`) and `flux_file` to that
model's SNEWPY data file. Requires sntools+snewpy on `SN_ENV_BASE`
(see `jobs/user_paths.nersc.sh.template`); the shipped analytic `gamma`
fluxes + the built-in MSW transformations validate end-to-end with no
download.

## End-to-end (PhotonSim + LUCiD + HTMLs)

`generate_validation_htmls.sh` runs PhotonSim, feeds its ROOT into the
script above, and emits per-event HTML visualizations. Currently wired
for S3DF (Singularity image + `/sdf/...` paths); adapt for local use.

```bash
./lucid/production/generate_validation_htmls.sh -c 07 -n 5
```

A longer doc with the full pipeline walkthrough and a portable wrapper
will land later.
