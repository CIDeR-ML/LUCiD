# LUCiD Production Pipeline

Turns a PhotonSim particle-based ROOT file into a four-file HDF5 dataset
(`sensor/`, `inst/`, `seg/`, `labl/`). Full schema in
[`docs/reference/dataset-schema.md`](../../docs/reference/dataset-schema.md).

## Config blocks

Configs live under `configs/<block>/NN_name.json` (2-digit, numbering restarts
per block):

| block | contents | splits |
|---|---|---|
| `GeV` | single particles (01–05), particle-bomb (06), multiparticle (07–12), GENIE numu+nue (13) | 01–06 train+test (1M/50k); 07–13 **test-only** (50k) |
| `Solar` | low-energy e⁻ (01) | train+test (1M/50k) |
| `SN` | supernova bursts (01) | flat |
| `Test` | dev/scratch (pile-up bombs 01) | flat |

Each config declares its `detector` and `nominal_train` / `nominal_test`; the
fanout writes `OUTPUT_BASE/<detector>/<block>/[<split>]/config_NN/` and sizes jobs
from `nominal / (target_seconds_per_job / seconds_per_event)`. Train/test datasets
use disjoint master seeds. See
[`jobs/dataprod/`](jobs/dataprod/) and [`docs/guides/production/deploy-nersc.md`](../../docs/guides/production/deploy-nersc.md).

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

## Digitizer / hit-making

Sensor hit-making is a per-sensor sliding-window integrator
(`lucid/simulation/digitizer.py`). The model is set by an optional `"digitizer"`
block in the **detector physics config** (e.g. `config/SK_like_physics_config.json`);
absent ⇒ `basic`.

```json
"digitizer": { "model": "ski" }
```

Models (every electronics/PMT number sourced from WCSim — see
`lucid/simulation/digitizer.py` for `WCSim/...` citations):
- **`basic`** — ∞ window → one digit per sensor, the legacy first-arrival + summed
  charge behaviour (default; idealized passthrough).
- **`ski`** — SK 20″ PMT (`PMT20inch`): 200 ns window, no deadtime, 0.25 pe
  threshold, per-photoelectron SPE charge, charge-dependent Gaussian time jitter,
  4.2 kHz dark.
- **`hk`** — HK 20″ Box&Line PMT (`BoxandLine20inchHQE`): same window/threshold,
  its own (sharper) SPE + exponentially-modified-Gaussian time jitter, 4.2 kHz dark.

Any parameter (`integration_window_ns`, `deadtime_ns`, `threshold_pe`,
`dark_rate_khz`, …) is overridable in the block.

Non-`basic` models can record **multiple digits per sensor** when light arrives
in separated time clusters (delayed coincidence, pile-up, dark noise), so
`sensor.h5` becomes a digit list and `hits.h5` / `step/sensor_hits` carry a
`digit_idx` FK. **Dark noise** is a labelled source in the
`hits.h5` decomposition (`emission_process = 2`, `particle_idx = -1`). See
`docs/reference/dataset-schema.md`.

**Pile-up** is digitized **cross-vertex**: all vertices' per-photon deposits are
pooled in absolute time and windowed once, so light from different vertices that
overlaps on a PMT within the integration window merges into one digit (with the
decomposition attributing its charge across the contributing vertices). `basic`
pile-up reproduces the legacy per-sensor first-arrival + summed charge.

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
labl truth after the writer completes.

A dataset fans out into one `<model>/<ordering>/` subcase per
`supernova.models × supernova.orderings` — mass ordering maps to sntools'
`AdiabaticMSW_NMO` / `AdiabaticMSW_IMO` transformation:

```
<detector>/SN/config_NN/                     # SN block is flat (no train/test)
  analytic_accretion/{NMO,IMO}/{sensor,hits,step,labl}/
  analytic_hot/{NMO,IMO}/{sensor,hits,step,labl}/
```

Config block (see `configs/SN/01_supernova_50kpc.json`):

```json
"primary_source": "supernova",
"supernova": {
  "detector": "HyperK",        // sntools fiducial → realistic event count
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
