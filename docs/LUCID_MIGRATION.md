> **STATUS: COMPLETED.** This migration was executed during the production-interm → refactor-v2 merge. Retained for historical reference.

# LUCiD Production HDF5 Migration

This document specifies everything required to migrate LUCiD's HDF5 production
output from its current two-file flat-CSR format to the new four-file
per-event-group format defined in
[LUCID_DATASET.md](LUCID_DATASET.md).

It bundles design rationale, the upstream PhotonSim changes that have been
made, the LUCiD writer and reader changes that still need to be made, the
files affected, the files explicitly NOT affected, and the validation plan.

The migration is a **clean break** — there is no backwards compatibility
requirement.

---

## 1. Executive summary

The migration changes only the **production output writers and the
production-output readers** in LUCiD. The interactive single-event
save/load workflow used by notebooks, optimization, and visualization
debugging is **not affected** and stays as-is.

**Scope**:
- 1 file in `lucid/sources/` (`event_io.py`) — writer rewrites + reader
  additions, plus removal of obsolete production helpers.
- 4 files in `lucid/production/` — readers, voxelization, and an
  invocation script.
- 0 files in `s3df_jobs/` — confirmed independent (uses PhotonSim ROOT
  input, not LUCiD HDF5 output).
- 0 files in `lucid/optimization/`, `lucid/visualization.py`,
  `lucid/siren/`, `lucid/wavelength/`, `lucid/sources/calibration_sources.py`,
  `lucid/sources/particle_model.py`, `lucid/sources/siren_rays.py` —
  none consume production batch output.
- 0 files in `notebooks/`, `good_notebooks/`, `tools/` — they use
  the interactive single-event API which is not part of the production
  schema being migrated.

**Already done (PhotonSim side)**: branch
`add-per-segment-beta-ncherenkov` adds `Segment_BetaStart` and
`Segment_NCherenkov` per Geant4 step. Built and verified.

**Still to do (LUCiD side)**: implement v3 writers and readers, refactor
the two production-entry generators to call them, drop the obsolete
v1/v2 helpers, fix the pre-existing `qe_corrections` bug that blocks
end-to-end runs. (Voxelization has already been dropped — `voxelize.py`
and its notebook are deleted.)

---

## 2. Current state

### 2.1 Current LUCiD output

Two HDF5 files per batch:

`sensor_events_NNNN.h5` — flat CSR, datasets at file root:
- Sensor hits: `event_hit_PE`, `event_hit_T`, `event_hit_offsets`,
  `event_hit_sensor_idx`
- Per-particle hits: `particle_hit_PE`, `particle_hit_T`,
  `particle_hit_offsets`, `particle_hit_sensor_idx`,
  `particle_event_offset`
- Per-event: `event_number`, `n_particles`, `t0`, `overall_containment`
- Per-particle metadata: `particle_category`, `particle_containment`,
  `genealogy_data`, `genealogy_offsets`
- Geometry: `sensor_positions`
- (optional) Voxel data: `voxel_n_nonzero`, `voxel_offsets`,
  `voxel_flat_indices`, `voxel_counts`
- File-level attrs: `format_version`, `n_events`, `n_sensors`, `source`,
  `detector_config`, `material`, `detector_type`, `master_seed`,
  `smearing_applied`, `root_file`

`segment_events_NNNN.h5` — flat CSR, datasets at file root:
- Per-segment: `start_x/y/z`, `end_x/y/z`, `dir_x/y/z`, `edep`, `time`
- Per-track: `track_id`, `parent_id`, `pdg`, `initial_energy`,
  `n_cherenkov`
- CSR offsets: `track_event_offset`, `segment_offset`
- Per-event: `event_number`
- Per-particle: `ext_genealogy_data`, `ext_genealogy_offsets`
- File-level attrs: `format_version`, `n_events`

### 2.2 Current writers and entry points

`lucid/sources/event_io.py`:
- `save_sensor_batch_v2(...)` — current production writer for sensor file
- `save_segment_batch_v2(...)` — current production writer for segment file
- `merge_event_files(...)` — merges per-event temp files into batches
- Older v1 helpers used by interactive workflows: `save_single_event`,
  `load_single_event`, `save_single_event_with_extended_info`,
  `save_single_event_with_particle_info`, `read_event_file`,
  `read_multi_folder_events`

`lucid/production/`:
- `generate_events.py` — entry CLI for plain photon-based generation
- `generate_events_with_particles.py` — entry CLI for particle-based
  generation; this is the canonical path
- `data_prod_utils.py` — production reader `read_multi_event_file()`
- `particle_data_utils.py` — production reader for particle data
- `visualize_particle_events.py` — production-output visualization
- `voxelize.py` — voxelization (to be removed entirely)

### 2.3 Pre-existing pipeline bug (blocks testing)

`lucid/simulation/sensor_response.py:51` indexes `qe_corrections[flat_indices]`,
but the default config `config/SK_like_physics_config.json` sets
`qe_corrections: 1.0` (a scalar). End-to-end production runs currently
fail with `IndexError: Too many indices: array is 0-dimensional`.

This is unrelated to the schema migration but needs fixing before any
new writer can be tested end-to-end. Fix options:
1. Detect rank in `make_hits_data` and broadcast scalar
   `qe_corrections` instead of indexing
2. Update physics config to array form (`[1.0] * n_sensors`)
3. Treat scalar `qe_corrections` as `None`

Option 1 is the safest and most permissive.

---

## 3. Target schema (summary)

Four parallel files per batch:

```
dataset_root/
├── sensor/    wc_sensor_NNNN.h5    raw PMT readout (post-smearing, t0-shifted)
├── inst/      wc_inst_NNNN.h5      per-particle PMT decomposition (pre-smearing)
├── seg/       wc_seg_NNNN.h5       3D segments + per-segment kinematics
└── labl/      wc_labl_NNNN.h5      labels and truth metadata at three granularities
```

Inside each file, `config/` group + a sequence of `event_NNN/` groups.
Indices within each event group are local `[0, n)`. Per-row FK columns
(`particle_idx`, `track_idx`) replace global CSR offsets. The full
schema, datasets, attrs, and field-by-field migration mapping live in
[LUCID_DATASET.md](LUCID_DATASET.md).

Key conventions:
- Group naming: `event_NNN` is the sequence position in the file
  (zero-padded). The canonical PhotonSim entry index lives as
  `source_event_idx` attr on each group and as a file-level
  `config/source_event_idx (n_events,) uint32` array.
- All times in `sensor.h5`, `inst.h5`, `seg.h5` are stored
  **t0-shifted** (detector frame). The truth `t0` lives only in
  `labl.h5/event_NNN/per_event/t0` as a per-event scalar.
- Voxel data is dropped entirely — superseded by per-segment Geant4
  geometry.
- Sensor file carries no labels — labels live exclusively in `labl.h5`.

---

## 4. Design choices and rationale

These were the load-bearing decisions made during the design discussion.
Each is documented here so colleagues can understand or contest the
reasoning.

### 4.1 Per-event H5 groups instead of flat-CSR

Empirically benchmarked (4 storage configurations × 4 dataset scales):
per-event groups beat flat-CSR on per-event read time by 2–20× because
the loader opens one small group instead of scanning multi-MB index
arrays. The cost is ~2–18% file-size overhead from HDF5 group
metadata, which is acceptable.

### 4.2 FK columns instead of CSR offsets within an event

ID columns (`particle_idx`, `track_idx`) per row instead of `seg_off`
arrays. With gzip compression, the size difference is negligible
(~0.5% in benchmarks). FK columns are shuffle-friendly and match
standard tabular conventions; CSR offsets break under any reordering.

### 4.3 Multi-granularity labl with sub-tables

`labl.h5/event_NNN/` has three subgroups: `per_event/`, `per_particle/`,
`per_track/`. They reflect a real LUCiD distinction:

- **Particles** (~8/event) — Geant4-categorized objects (Primary,
  DecayElectron, SecondaryPion, Gamma). Used to decompose the PMT
  signal in `inst/`.
- **Tracks** (~600/event) — full Geant4 tracks. The 3D truth in
  `seg/` is organized at this granularity.

Each track maps to exactly one particle via a `particle_idx` FK
column derived from the genealogy chain at write time. Verified
injective on production data (>99.9% of tracks map cleanly; the
~1 orphaned track per event is a Geant4 housekeeping artifact and
gets `particle_idx = -1`).

### 4.4 t0 is truth, not denormalization

`t0` is the true emission time of the event. It belongs in `labl/`
as a label, not in `sensor/` as a denormalization parameter. All
hit times across files are stored already-shifted (detector frame),
so the sensor file looks like real detector data.

### 4.5 β per segment as the kinematic base unit

For Cherenkov resimulation outside Geant4:
- Frank–Tamm needs only β (`cos θ_c = 1/(βn(λ))`, photon yield ∝
  `sin²θ_c · L`).
- For charged particles in water, `z² = 1`, so PDG isn't needed in
  the Cherenkov calculation at all.
- Storing β rather than KE means downstream code needs no PDG
  lookup (mass) to do the resimulation — fully self-contained per
  segment.
- β is bounded `[0, 1]`, native Geant4 quantity (`G4StepPoint::GetBeta()`),
  and pairs naturally with PDG (in labl per-track) for the rare cases
  needing total energy.

`n_cherenkov` per segment is also stored as the simulator's exact
G4 count — useful as a forward-sim ML target. Both were verified at
the time via `sum(Segment_NCherenkov) == sum(MTrack_NCherenkov)` per
event; `MTrack_*` was retired in PhotonSim Stage 5a, so the same
invariant now reads as `sum(Segment_NCherenkov)` summed per
`Segment_TrackID` group equaling the previously-stored per-track
totals.

### 4.6 Voxels dropped

Voxels were a 100³ binning of per-particle photon positions, useful
when segments weren't available. With segment data, voxels are
strictly coarser, redundant info — anyone wanting a 3D density grid
can compute it from segments at load time.

### 4.7 No backwards compatibility

The migration is a clean break. Old v2 writers and v1 readers are
deleted. Existing v1 datasets are not migratable in place — they
would need re-generation with the new pipeline.

### 4.8 Field renames (clarity)

- `event_number` → `source_event_idx` (the existing name implied a
  DAQ event number; the actual semantic is "PhotonSim ROOT entry
  index" — a source traceback ID).
- `master_seed` → `lucid_master_seed` (explicit about scope; PhotonSim
  has its own seed not currently propagated, so a placeholder
  `photonsim_seed = -1` is reserved).
- `root_file` → `source_file` (generic across detector types).

`label_names` config attr in labl is kept (small list of class-scheme
columns; currently `["category"]` for LUCiD).

### 4.9 sensor_positions duplicated to inst

`inst/` carries its own copy of `sensor_positions` so it's loadable
standalone for SSL or supervised training tasks that don't need
sensor's raw hits. ~130 KB duplication; kept.

### 4.10 Smearing function name stored as string

`sensor/config/smearing_charge_function` and
`sensor/config/smearing_time_function` — stored as identifiers
(`"SK_like"`). No params dataset until smearing functions become
parameterizable.

---

## 5. PhotonSim changes (done)

Branch: `add-per-segment-beta-ncherenkov` in
`/home/oalterka/desktop_linux/diffWC/PhotonSim/`. Commit `1ef5ace`.

### 5.1 Files modified

`include/DataManager.hh`:
- Added `betaStart`, `nCherenkov` fields to `TrackSegment` struct.
- Extended `AddTrackSegment(...)` signature with `betaStart` and
  `nCherenkov` parameters.
- Added `fSegment_BetaStart`, `fSegment_NCherenkov` member vectors.

`src/DataManager.cc`:
- New TBranch declarations: `Segment_BetaStart` (vector<double>),
  `Segment_NCherenkov` (vector<int>).
- Merge logic: `betaStart` of merged segment preserved from first
  sub-step; `nCherenkov` accumulates across merged sub-steps.
- Added push_back in segment-output loop.
- Added clear() for the two new vectors at end of event.
- Updated `AddTrackSegment` body to write the two new fields.

`src/SteppingAction.cc`:
- Compute β per pre-step via `step->GetPreStepPoint()->GetBeta()`.
- Count Cherenkov photons emitted in this step by iterating
  `step->GetSecondaryInCurrentStep()` and matching creator process
  name `"Cerenkov"`.
- Pass both into `AddTrackSegment(...)`.

### 5.2 Verification

Built with `make -j4` in `build/`, runtime tested with
`macros/test_wavelength_5_events.mac` (5 muon events at 1050 MeV).

Sanity checks pass:
- Both branches written (`Segment_BetaStart`, `Segment_NCherenkov`
  appear in the ROOT file).
- `sum(Segment_NCherenkov) == sum(MTrack_NCherenkov)` per event for
  all 5 events — confirmed per-step counts sum to per-track totals.
  (`MTrack_*` was retired in Stage 5a; the equivalent post-Stage-5a
  check sums `Segment_NCherenkov` grouped by `Segment_TrackID`.)
- β values fall in `[0, 1]` as physically expected.
- Segment counts (~700–900 per event) consistent with merge-logic
  output unchanged.

### 5.3 Future PhotonSim work (not done)

Add a per-event `random_seed` branch that exposes the Geant4 RNG
seed used for that event, so downstream code can populate
`photonsim_seed` in LUCiD's provenance. Currently a placeholder
(`-1`) in the LUCiD writer. Independent of the schema migration.

---

## 6. LUCiD writer changes

All inside `lucid/sources/event_io.py`. The current `save_sensor_batch_v2`
and `save_segment_batch_v2` produce the flat-CSR format and are
**replaced** (not extended).

### 6.1 New writer functions

```
save_sensor_event_v3(f, event_dict)         — writes one event_NNN/ group
save_hits_event_v3(f, event_dict)           — writes one event_NNN/ group
save_edep_event_v3(f, event_dict)            — writes one event_NNN/ group
save_labl_event_v3(f, event_dict)           — writes one event_NNN/ group
```

Each opens an already-open file handle and writes a single event
group with the schema documented in `LUCID_DATASET.md`.

### 6.2 New config writers

```
write_sensor_config_v3(f, ...)
write_hits_config_v3(f, ...)
write_edep_config_v3(f, ...)
write_labl_config_v3(f, ...)
```

Each writes the file-level `config/` group with provenance attrs,
geometry, and the `source_event_idx (n_events,) uint32` array.

### 6.3 Per-event derived computations at write time

- **`particle_idx` per track** — walk each track's `parent_id`
  chain until reaching a categorized particle's `track_id` (the
  last entry of its `genealogy_data`). Implementation: build a
  per-event lookup `categorized_track_id → particle_idx` once,
  then trace parents per track. Tracks that fail to reach any
  categorized particle get `particle_idx = -1` (~0.03% in
  production data).
- **`t0` shift on `seg.time`** — current `Segment_Time` from
  PhotonSim is G4-absolute. New writer adds `t0` at write time so
  all per-segment times are in detector frame.
- **`t0` shift on inst `T`** — verify whether the LUCiD photon
  transport already adds `t0` to PMT arrival times before
  smearing. If not, apply at write time so inst and sensor share
  the same time frame.

### 6.4 Provenance block additions

Currently captured in `production_meta`: `detector_config`,
`material`, `detector_type`, `smearing_applied`, `master_seed`,
`root_file`. To add:
- `git_commit` — auto-detect via `subprocess.check_output(['git',
  'rev-parse', 'HEAD'], cwd=lucid_repo_root)` at run time.
- `run_id` — generate (UUID4 or timestamp+seed).
- `dataset_name` — pass via CLI.
- `file_index` — already implicit; record explicitly.
- Rename `master_seed` → `lucid_master_seed`.
- Rename `root_file` → `source_file`.
- Add `photonsim_seed = -1` placeholder until PhotonSim exposes
  it (see §5.3).
- Smearing function names: `smearing_charge_function`,
  `smearing_time_function` (e.g., `"SK_like"`, `"default"`).

### 6.5 Functions to delete

After v3 writers are in and all callers in production are
migrated:
- `save_sensor_batch_v2` (lines 2822–3021)
- `save_segment_batch_v2` (lines 3024–3145)
- `read_sensor_event_v2` (lines 3148+) — V1-compatible reader
- `merge_event_files` (lines 2143–2216) — batch files don't need
  merging in the new flow
- `save_single_event_with_extended_info` (lines 1869–1938) —
  unused by production after migration; verify no other callers
- `save_single_event_with_particle_info` (lines 1940–2142) —
  same, verify no other callers

**Functions to keep** (used by interactive single-event workflows
in optimization, visualization, notebooks — NOT production
schema):
- `save_single_event` (lines 1556–1633)
- `load_single_event` (lines 1635–1705)

These handle a single test event saved/loaded inside a notebook
or optimization run. They are independent of the production batch
schema and stay as-is.

`read_event_file` (2292+) and `read_multi_folder_events` (2218+)
read v1 merged production output. After migration they should be
deleted (no backwards compat needed).

### 6.6 Pipeline refactor in event_io.py

`generate_events_from_photonsim_particles` (lines 983–1508) is the
canonical particle-based generation entry. Currently it batches
events in memory then calls `save_sensor_batch_v2` and
`save_segment_batch_v2`. The new flow:

1. Compute provenance block once at start of run.
2. Open four file handles: `wc_sensor_NNNN.h5`, `wc_inst_NNNN.h5`,
   `wc_seg_NNNN.h5`, `wc_labl_NNNN.h5`.
3. Call `write_*_config_v3` once per file.
4. For each event in the batch:
   - Compute the per-event derived fields (`particle_idx` per
     track, t0-shifted times).
   - Call `save_sensor_event_v3`, `save_hits_event_v3`,
     `save_edep_event_v3`, `save_labl_event_v3` writing
     `event_NNN/` groups.
5. Close all four files.

`generate_events_from_photonsim` (line 732) — the older
photon-based path that doesn't use the particle system. Decision
point: either also migrate it to v3 writers, or deprecate it
(particle-based is the canonical path now). My recommendation:
deprecate.

---

## 7. LUCiD reader changes

### 7.1 New v3 readers (in `event_io.py`)

```
read_sensor_event_v3(filename, event_idx) → dict
read_hits_event_v3(filename, event_idx) → dict
read_edep_event_v3(filename, event_idx) → dict
read_labl_event_v3(filename, event_idx) → dict
```

Open file, navigate to `event_NNN/` group (where N = event_idx),
return all datasets as a dict keyed by name. Cross-reference
`source_event_idx` attr against `config/source_event_idx[event_idx]`
for integrity.

Plus:

```
list_events_v3(filename) → np.ndarray   # returns config/source_event_idx
```

For loaders that need to know what events live in the file before
opening any group.

### 7.2 Production-side reader rewrites

`lucid/production/data_prod_utils.py`:
- `read_multi_event_file()` — currently reads merged v1 HDF5 from
  flat root datasets. Needs full rewrite:
  - Take 4 file paths (or one dataset_root path)
  - For each event, open the four `event_NNN/` groups across files
  - Reconstruct whatever the existing API contract expects (or
    update the contract to expose the four-file view explicitly)
- Roughly ~180 lines, full rewrite.

`lucid/production/particle_data_utils.py`:
- `read_particle_event_file()` — currently reads particle data
  from per-event groups (this format already used per-event
  groups internally, just at a different layout). Update group
  paths and dataset names to match v3 schema.
- `_read_single_particle_event()` helper — same.
- ~100 lines of path updates.

`lucid/production/visualize_particle_events.py`:
- Multiple direct `h5py.File(...)` access sites (lines 137, 646,
  669) reading datasets like `genealogy_data`, `track_id`,
  segment data, voxel data.
- Update HDF5 navigation to v3 four-file structure.
- Remove voxel branch entirely (voxels are gone).
- ~150 lines of edits.

### 7.3 CLI invocation updates

`lucid/production/generate_events.py`:
- Drop or no-op `--merged-filename` (no merging in v3).
- Add `--dataset-name` (required, used in provenance).
- Add `--run-id` (optional, auto-generated if absent).
- Output to `--output/{sensor,inst,seg,labl}/wc_*_NNNN.h5`.
- ~20 lines.

`lucid/production/generate_events_with_particles.py`:
- Same CLI changes.
- ~20 lines.

`lucid/production/generate_validation_htmls.sh`:
- Possibly minor adjustment if it reads merged production output.
  Check the validation script's input expectation.

### 7.4 Voxelization deletion

`lucid/production/voxelize.py` — delete entire file (~464 lines).

After deletion, verify nothing imports from it. Likely only the
old writers reference it.

---

## 8. Files not affected (and why)

This section exists to prevent over-claiming impact.

### 8.1 Notebooks and good_notebooks (~15 files)

Investigated: `single_ring_track_optimization.ipynb`,
`detector_3D_visualization.ipynb`, `event_hit_animation.ipynb`,
and the rest by pattern.

These notebooks use `save_single_event` / `load_single_event` for
a **single-event roundtrip workflow**: they generate a test event
inside the notebook, save it to a temporary file
(e.g., `events/MidBox_event_data.h5`), and load it back later for
plotting or optimization debugging.

This is **not** production batch output. The notebooks never call
`read_event_file`, `read_multi_event_file`, or any of the
production-output readers.

→ **Notebooks unaffected.** `save_single_event` /
`load_single_event` are kept in `event_io.py` exactly as they
are.

### 8.2 lucid/optimization/

`run.py` and `pipeline.py` import `save_single_event` /
`load_single_event` for the same single-event roundtrip pattern
(saving optimization snapshots). Independent of production
schema.

→ **Optimization unaffected.**

### 8.3 lucid/visualization.py

The shared visualization helper module is data-format agnostic.
It consumes already-loaded arrays via `sparse_to_full()` and
similar helpers, not HDF5 files directly.

→ **Visualization helpers unaffected.**

### 8.4 lucid/siren/

SIREN training reads its own lookup-table HDF5 files (dEdx, photon
tables) generated by a separate pipeline. Not production event
output.

→ **SIREN unaffected.**

### 8.5 lucid/wavelength/, lucid/sources/calibration_sources.py, particle_model.py, siren_rays.py

None consume production HDF5 output.

→ **Unaffected.**

### 8.6 s3df_jobs/

Inventory:
- `submit_*.py` (3 files) — job submission wrappers
- `run_eval_with_parametrization.py` — reads PhotonSim ROOT, not
  LUCiD HDF5
- `run_track_optimization.py` — reads PhotonSim ROOT, not LUCiD
  HDF5
- `*_config/create_*.py` — config generators
- `*.json` — optimization configs

None read LUCiD production HDF5 output.

→ **s3df_jobs unaffected.**

### 8.7 tools/

100+ files, mostly diagnostic notebooks/scripts and stale
exploration code (e.g., `gradient_diagnostic_v*.py`). Spot-checks
suggest these read PhotonSim lookup tables or test data, not
production output. If any are found to depend on production
output during the migration, update them then.

→ **Tools likely unaffected; audit case-by-case if anything
breaks.**

---

## 9. Files affected — concrete change list

| File | Change | Approx LoC |
|------|--------|-----------|
| `lucid/sources/event_io.py` | Add v3 writers (4) and v3 readers (4); refactor `generate_events_from_photonsim_particles` to call them; add provenance block computation; delete `save_sensor_batch_v2`, `save_segment_batch_v2`, `read_sensor_event_v2`, `merge_event_files`, `read_event_file`, `read_multi_folder_events`, `save_single_event_with_extended_info`, `save_single_event_with_particle_info` | ~+800 / -800 |
| `lucid/production/data_prod_utils.py` | Rewrite `read_multi_event_file()` for 4-file/per-event-group layout | ~180 (full rewrite) |
| `lucid/production/particle_data_utils.py` | Update HDF5 paths and dataset names to v3 schema | ~100 |
| `lucid/production/visualize_particle_events.py` | Update h5py call sites; remove voxel branch | ~150 |
| `lucid/production/generate_events.py` | Drop `--merged-filename`; add `--dataset-name`, `--run-id`; output to 4 subdirs | ~20 |
| `lucid/production/generate_events_with_particles.py` | Same CLI updates | ~20 |
| `lucid/production/generate_validation_htmls.sh` | Update if consuming production output | ~5 (if any) |
| `lucid/production/voxelize.py` | **Delete entire file** | -464 |
| `lucid/utils.py` | Update re-export list; remove deleted symbols | ~5 |

---

## 10. Pre-existing bug to fix before testing

`lucid/simulation/sensor_response.py:51`:
```python
per_photon_qe = qe * qe_corrections[flat_indices] if qe_corrections is not None else qe
```

Default config sets `qe_corrections: 1.0` (scalar) → `IndexError`
on first end-to-end run. Fix: detect rank and broadcast scalar
case before indexing. Required to validate any new writer.

---

## 11. Validation plan

Once §6, §7, and §10 are done:

1. **Round-trip integrity** — generate a small batch (5–10 events)
   from the verified PhotonSim test ROOT
   (`PhotonSim/build/test_wavelength_5_events.root`); confirm the
   four output files exist with the expected schema; read each
   event back via the new v3 readers; verify all data round-trips.

2. **Cross-modality alignment** — for each `event_NNN` group,
   verify that `source_event_idx` attribute matches across all
   four files and matches `config/source_event_idx[N]`.

3. **`particle_idx` derivation correctness** — for each track in
   `labl/per_track`, walk its `parent_id` chain and confirm it
   reaches the `track_id` of the categorized particle indicated
   by `particle_idx`. Count orphaned tracks; expect <0.1%.

4. **Time-frame consistency** —
   - `seg.time[i] - labl.per_event.t0[event] ≈ G4_truth_time` (re-
     check against PhotonSim ROOT `Segment_Time`).
   - All hit times in `sensor.T` and `inst.T` lie in
     `[t0_min, t0_max + max_propagation_time]`.

5. **Cherenkov sanity** — `sum(seg.n_cherenkov per track) ==
   labl.per_track.n_cherenkov[track]` (already verified at
   PhotonSim level; re-verify after LUCiD writes them).

6. **β consistency** — `seg.beta_start ∈ (0, 1]` for all segments
   with edep > 0. β=0 occurs only for terminating segments
   (acceptable).

7. **Schema completeness** — for each `event_NNN/` group across
   the four files, verify every documented dataset is present and
   has the documented dtype/shape.

8. **No orphan datasets** — verify no datasets exist outside the
   schema (e.g., leftover voxel arrays from a stale code path).

9. **pimm-side reader** — once the LUCiD pimm reader is updated to
   v3, run pimm's existing LUCiD test suite against the new files
   and confirm it loads cleanly.

10. **Compare against current production** — for the same
    PhotonSim ROOT input, generate output with both v2 and v3
    writers (during the brief transition); confirm the per-event
    physics content (PE, T, segments, edep, etc.) matches.

---

## 12. Critical-path order

1. Fix `qe_corrections` bug in `sensor_response.py` (blocks all
   testing).
2. Implement v3 writers + config writers in `event_io.py`. Add
   `particle_idx` derivation, t0 shift logic, provenance block.
3. Refactor `generate_events_from_photonsim_particles` to call
   v3 writers. Update CLI in `generate_events_with_particles.py`.
4. Run validation §11 1–8.
5. Implement v3 readers in `event_io.py`.
6. Rewrite `data_prod_utils.py::read_multi_event_file()`.
7. Update `particle_data_utils.py`, `visualize_particle_events.py`.
8. Run validation §11 9–10.
9. Delete `voxelize.py`, deprecated v2 writers, v1 production
   readers, `merge_event_files`.
10. Update pimm-side LUCiD reader (separate repo) to consume v3.
11. Once stable, push PhotonSim branch to mainline; release.

---

## 13. Pointers

- New schema (full spec): [LUCID_DATASET.md](LUCID_DATASET.md)
- General design principles: [DATASET_DESIGN.md](DATASET_DESIGN.md)
- LUCiD repo, on `refactor-v2` branch:
  `/home/oalterka/desktop_linux/diffWC/LUCiD/`
- PhotonSim repo, on `add-per-segment-beta-ncherenkov` branch:
  `/home/oalterka/desktop_linux/diffWC/PhotonSim/` (commit
  `1ef5ace`)
- Test PhotonSim ROOT (verified with new branches):
  `/home/oalterka/desktop_linux/diffWC/PhotonSim/build/test_wavelength_5_events.root`
- Sample current LUCiD output (for reader migration testing):
  `/home/oalterka/desktop_linux/diffWC/LUCiD/output/dataprod10_v2/`
