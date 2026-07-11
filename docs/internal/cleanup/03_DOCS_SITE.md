# 03 — Hosted documentation site plan

**Status:** proposal. Content-only (no site build in this pass).
**Principle:** general guidance — *how the framework works and how to use it*. **No research
numbers, CRB/resolution tables, or `mie_hunter` findings.** External-first, detailed enough
for internal collaborators.

## Tooling

**MkDocs Material + `mkdocs-jupyter` + `mkdocstrings[python]`, deploy to GitHub Pages.**
Chosen over Sphinx: the repo has zero RST investment, the `docs/` corpus is already Markdown,
and the goal is hand-authored conceptual docs (Material's strength) with a *minimal curated*
API reference (`mkdocstrings`), not a full autodoc dump. Precedent: Equinox/Diffrax.
Add a `[docs]` extra in `pyproject.toml` (`mkdocs-material`, `mkdocs-jupyter`,
`mkdocstrings[python]`).

## Information architecture (8 sections)

1. **Home** — the pitch (echo the preprint abstract), the differentiable-forward mental
   model, a "what do you want to do?" decision guide (simulate / reconstruct / calibrate /
   produce data), citation.
2. **Getting Started** — install (`pip install -e .` + extras) → `download_data.sh` →
   `hello_simulate` → Local/Docker quickstart → quickstart notebook. *SIREN-only path, no
   PhotonSim/GEANT4.*
3. **Concepts** — architecture (the one engine), the photon pipeline, geometry & the
   registry, detectors & geometry families, parameters & configuration, the SIREN surrogate,
   wavelength-dependent physics.
4. **Workflow Guides** — one per canonical notebook, prose anchored on the `examples/hello_*`
   scripts with the notebook embedded: simulate, reconstruct, calibrate (incl. per-PMT QE),
   gradients/landscapes, data-vs-prediction, train SIREN.
5. **Data & Production** — v3 dataset schema (`step` modality), the production chain
   (`lucid-run-job`, PhotonSim/GENIE), cluster deployment (S3DF / NERSC / LXPLUS fronted by
   cluster abstraction).
6. **Reference** — CLI (5 console scripts), config JSON field reference, PMT `.npz` schema,
   minimal curated API (simulation, geometry, fitting, detector_params).
7. **Contributing** — dev setup & tests, the "inside/outside rule", add a
   geometry/source/medium (the registry story).
8. **Performance & scaling** *(short)* — the efficiency story, pointing at
   `ci_tests/speed_test.py` as the reproducible harness. (No absolute numbers baked in.)

> **Detectors & geometry families** deserves a dedicated Concepts page — it is LUCiD's
> strongest selling point (cylinder/sphere/box/string/nested; SK/HK/WCTE from measured
> `.npz`; IceCube-style strings) and is currently homeless.

## Existing-doc disposition

| Doc | Disposition | Target section |
|-----|-------------|----------------|
| `README.md` | rewrite (see `04`) | repo root |
| `docs/ARCHITECTURE.md` | **reuse, light edit** (drop internal "reconciliation status" tail) | Concepts / Architecture |
| `docs/DETECTOR_PARAMS_VS_ARGS.md` | **reuse** (accurate; footnote `SourceParams` as future) | Concepts / Params & config |
| `docs/LUCID_DATASET.md` | **reuse, verify columns**; ensure `step` modality | Data & Production |
| `docs/CLUSTER_ABSTRACTION.md` | **reuse** | Data & Production / cluster |
| `docs/QUICKSTART_LOCAL.md`, `_DOCKER.md` | **reuse** (add `download_data.sh` step) | Getting Started |
| `docs/QUICKSTART_S3DF/_NERSC/_LXPLUS.md` | **reuse** | Data & Production / cluster |
| `docs/SIREN_TRAINING_INPUTS.md` | **reuse** | Workflow / Train SIREN |
| `docs/WAVELENGTH_DESIGN.md` | **reuse the "how it is now" half**; drop proposal parts | Concepts / Wavelength |
| `docs/WBLS_INFORMATION.md` | **reuse** (short domain background) | Concepts or Reference |
| `lucid/geometry/PMT_NPZ_SCHEMA.md` | **reuse** (symlink/include) | Reference |
| `docs/CALIBRATION.md` | **rewrite/split** — workflow guide (anchor on `hello_calibrate`); move any findings out | Workflow / Calibrate |
| `docs/RECONSTRUCTION.md` | **rewrite/split** — workflow guide (anchor on `hello_reconstruct`); findings out | Workflow / Reconstruct |
| `docs/PRACTICE_ARCHITECTURE.md` | **DO NOT SHIP** — aspirational (`Source.emit`/`ResponseModel`/`ParamRegistry` don't exist). Move to `docs/internal/` or delete | — |
| `docs/CALIBRATION_FRAMEWORK.md` | **DO NOT SHIP** — references non-existent `lucid/calibration/`. `docs/internal/` or delete | — |
| `docs/{LUCID_MIGRATION,LUCID_MIGRATION_FILES,MAIN_BRANCH_PLAN,RECONCILIATION_PLAN,PLAN_UNIFY,PHASE3_PLAN,MERGE_TTS_CHANGES,MERGED_DETECTORPARAMS_PROPOSAL,UNIFY_CALIB_RECON,RECON_CONSOLIDATION}.md` | **exclude from site** → `docs/internal/` or delete | — |
| `docs/EVENT_FORMAT_V2.md` | **exclude** (self-marked superseded by v3) | — |
| `docs/TWO_BOUNDARY_{PLAN,INTEGRATION}.md` | **out of scope** (lands as a later PR) | — |

## Net-new pages to author (from code, not from stale docs)

Priority order. These have **no correct existing source** — write from the modules noted.

**Newcomer:**
1. `index.md` (Home) — pitch + mental model + decision guide + citation.
2. `getting-started/install.md` — pip + extras + `download_data.sh` (the SIREN weights are
   not in git; call this out).
3. `getting-started/hello-simulate.md` — wraps `examples/hello_simulate.py`.
4. `concepts/photon-pipeline.md` — sources → propagation → `photon_step` → `sensor_response`;
   the "all JIT/vmap/scan, gradients flow end-to-end" thesis. Write from
   `lucid/simulation/{photon_step,sensor_response}.py`, `lucid/sources/*`,
   `lucid/wavelength/spectrum.py` (`Monochromatic/PowerLaw/QEWeighted`).
5. `concepts/geometry.md` + `concepts/detectors.md` — registry + geometry families.
6. `concepts/params-and-config.md` — `DetectorParams`/`ParticleParams` pytrees + two-JSON
   composable config (reuse `DETECTOR_PARAMS_VS_ARGS.md` core).

**Contributor:**
7. `contributing/index.md` — dev setup, test gating (`--slow`, `JAX_PLATFORM_NAME=cpu`),
   no-eager-imports convention, the inside/outside rule (harvest from `CLAUDE.md` +
   `good_notebooks/STRUCTURE.md`).
8. `contributing/extending.md` — add a geometry (registry decorator + propagator), a source,
   a medium/material JSON.

**Reference:**
9. `reference/cli.md` — the 5 console entry points with real invocations.
10. `reference/config.md` — geom + physics JSON field reference (see appendix below).
11. `reference/api-*.md` — `mkdocstrings` blocks for the curated surface (appendix below).

## Coverage additions (from the coverage stress-test)

The proposed set otherwise demonstrates only SK-cylinder-water; these close the gap to the
paper's "diverse geometries and materials" + "unified calibration" claims without bloat:

- **`concepts/detectors.md` must cover the full geometry range** — cylinder / sphere / **string
  (IceCube-style telescopes)** / box / nested_sphere — and the material axis (water / **WbLS** /
  **ice**), linking to the new `detector_and_material_gallery` notebook and `viewer/string/`.
- **New `reference/working-with-v3-data.md`** — read/loop a v3 HDF5 batch (seeded from
  `read_production_output.ipynb`); this is where the dropped `work_with_a_dataset` notebook lands.
- **Timing calibration** — surface `lucid.fitting.calibrate_timing` in the calibration workflow
  guide + the new gallery/`hello_telescope` should exercise `string` + `cascade` so the
  neutrino-telescope community has an on-ramp.
- `nested_sphere` / two-boundary, box-beyond-the-gallery, per-λ calibration → concept mention or
  explicitly deferred (two-boundary is a later PR).

## `mkdocs.yml` (copy-ready starting point)

```yaml
site_name: LUCiD
site_description: Differentiable photon simulation for optical particle detectors
site_url: https://cider-ml.github.io/LUCiD/
repo_url: https://github.com/CIDeR-ML/LUCiD
repo_name: CIDeR-ML/LUCiD

theme:
  name: material
  features: [navigation.sections, navigation.top, content.code.copy,
             content.code.annotate, search.suggest, search.highlight]
  palette:
    - scheme: default
      toggle: {icon: material/weather-night, name: Dark mode}
    - scheme: slate
      toggle: {icon: material/weather-sunny, name: Light mode}

plugins:
  - search
  - mkdocs-jupyter: {include: ["*.ipynb"], execute: false}   # ship *.executed.ipynb
  - mkdocstrings:
      handlers: {python: {options: {docstring_style: google, show_source: false}}}

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.arithmatex: {generic: true}
  - toc: {permalink: true}
extra_javascript: [https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js]

nav:
  - Home: index.md
  - Getting Started:
      - Install: getting-started/install.md
      - Hello, simulate: getting-started/hello-simulate.md
      - Local quickstart: QUICKSTART_LOCAL.md
      - Docker quickstart: QUICKSTART_DOCKER.md
      - Quickstart notebook: ../tutorials/00_quickstart.executed.ipynb
  - Concepts:
      - Architecture: ARCHITECTURE.md
      - Photon pipeline: concepts/photon-pipeline.md
      - Geometry & registry: concepts/geometry.md
      - Detectors & geometry families: concepts/detectors.md
      - Parameters & config: DETECTOR_PARAMS_VS_ARGS.md
      - SIREN surrogate: SIREN_TRAINING_INPUTS.md
      - Wavelength physics: WAVELENGTH_DESIGN.md
  - Workflow Guides:
      - Simulate an event: ../tutorials/00_quickstart.executed.ipynb
      - Reconstruct a track: ../tutorials/track_optimization.executed.ipynb
      - Calibrate optics: ../tutorials/calibration_optimization.executed.ipynb
      - Track gradients: ../tutorials/track_gradients.executed.ipynb
      - Data vs prediction: ../tutorials/data_vs_prediction.executed.ipynb
      - Event displays: ../tutorials/event_displays.executed.ipynb
      - Train SIREN: ../tutorials/train_siren.executed.ipynb
  - Data & Production:
      - v3 dataset schema: LUCID_DATASET.md
      - Production chain: guides/production-chain.md
      - Cluster deployment: CLUSTER_ABSTRACTION.md
      - S3DF: QUICKSTART_S3DF.md
      - NERSC: QUICKSTART_NERSC.md
      - LXPLUS: QUICKSTART_LXPLUS.md
  - Reference:
      - CLI: reference/cli.md
      - Config JSON fields: reference/config.md
      - PMT .npz schema: reference/pmt-npz-schema.md
      - API — simulation: reference/api-simulation.md
      - API — geometry: reference/api-geometry.md
      - API — fitting: reference/api-fitting.md
  - Contributing:
      - Dev setup & tests: contributing/index.md
      - Extend LUCiD: contributing/extending.md
  - Performance & scaling: performance.md
```

Note: `../tutorials/*.ipynb` assumes `mkdocs-jupyter` is allowed to reach outside `docs/`;
alternatively symlink/copy `tutorials/*.executed.ipynb` into `docs/`. Exclude every
`docs/*_PLAN.md` / migration / superseded doc from `nav` (move to `docs/internal/`).

---

## Appendix A — source-verified public API surface (for Reference pages)

Write the API/Reference pages from these real signatures (verified against source), **not**
from `PRACTICE_ARCHITECTURE.md`.

**Hub — `lucid/simulation/simulator.py`:**
`setup_event_simulator(json_filename, n_photons=1_000_000, temperature=0.2, K=7,
is_data=False, is_calibration=False, max_candidates_per_ray=4, detector_type='Cylinder',
use_expected_value=True, particle='muon', apply_smearing=True, physics_config=None,
default_detector_params=False, wavelength_mode=True, hit_mode=None, ...,
wavelength_sampling='cherenkov', reflection_model='scalar', spectrum=None, **grid_params)`
→ returns a JIT callable whose signature depends on mode:
- calibration: `(source, detector_params, key) -> (charges, times)`
- track: `(particle_params, detector_params, key) -> (charges, times)`
- data: `(particle_params, detector_params, key, photon_data) -> (charges, times)`
- `hit_mode ∈ {'aggregated' (calib), 'per_photon' (track), 'realistic' (data)}`.

**`DetectorParams`** (`lucid/detector_params.py`, NamedTuple pytree) — sub-tuples:
`scattering(scatter_length, mie_scatter_length, g, rayleigh_dev, mie_dev)`,
`absorption(absorption_length, abs_dev)`,
`reflection(wall_reflection_rate, sensor_reflection_rate, + angular-model fields)`,
`response(qe, spe_width, tts, qe_dev)`,
`per_pmt(qe_corrections[N], gain[N], t0[N], walk[N])`,
`scintillation(...)`, `outer_optics(None | two-medium)`. Ctor `DetectorParams.from_flat(**flat)`.

**`ParticleParams`:** `energy (MeV), position (3, m), theta (rad), phi (rad), t0 (ns)`.

**Calibration (`lucid.fitting`):** `build_calibration_problem(sim, sources, dp_true,
trainable_fields, *, per_pmt_field='qe_corrections', ...)`, `fit(...)`, `crb(...)`.

**Reconstruction (`lucid.fitting`):** `ReconModel(pred, num_detectors, sigma=2.5, ...)`,
`fit_track(...)`, `fit_track_multistart(...)`, helpers `track_from_vec9`/`vec9_from_track`/
`SCALE9`/`seed_vertex_time`; seeding via `lucid.optimization.grid_search`.

**Geometry:** `@register_detector(name)`, `generate_detector(file_path)`; registered types
`cylinder, sphere, box, string, nested_sphere`; `Cylinder.from_pmt_file(npz)`.

## Appendix B — config schema (for `reference/config.md`)

Two JSONs per detector (loaders: `lucid/detector_params.py::load_physics_config` /
`load_detector_params`):

- **`*_geom_config.json`**: `{ "material", "detector_type", "geometry_definitions": {...} }`.
  `detector_type` selects the registry class.
- **`*_physics_config.json`**: flat composable leaf keys (`scatter_length`,
  `absorption_length`, `wall_reflection_rate`, `qe`, `qe_corrections`, ...). Each value may
  be: `null`/missing → projected from a referenced λ-curve at `scalar_ref_wavelength`
  (default 400 nm); a number → scalar; a list → inline array; `"path.json"` → loaded;
  `"__array__:file.npy"` → companion array. Extra keys `medium_model`, `qe_curve` reference
  model/curve JSONs relative to the config dir.

## Accuracy notes

- **v3 modality = `step`** on `main` (`edep` is the per-segment energy-deposit *column*).
  `LUCID_DATASET.md` on `main` already reflects this after commit `24e069d`.
- `CLAUDE.md` stale-recon and `download_data.sh` "gdown" errors are fixed in `04`.
