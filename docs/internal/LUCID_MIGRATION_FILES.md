> **STATUS: COMPLETED.** File movements described here were executed. Retained for historical reference.

# LUCiD Migration — Complete File Scope (Harsh Mode)

Companion to [LUCID_MIGRATION.md](LUCID_MIGRATION.md). Assumes **harsh
mode**: no backwards compatibility, `notebooks/` disposable, and a clean
separation between the production batch format (v3, four files) and the
interactive single-event API (`save_single_event` / `load_single_event`,
unchanged).

---

## 1. Guiding principles

- The **production format** (v3) is a full rewrite. The current v2 flat
  CSR + the orphan `Q/T/P/V/PDG` format that `save_single_event_with_*`
  wrote are both deleted outright. No migration path for existing v2
  output.
- The **interactive single-event API** (`save_single_event`,
  `load_single_event`) is a different tool for a different job: small
  notebook roundtrips and optimization snapshots. It stays untouched.
- `good_notebooks/` is verified safe — every notebook there uses only
  the kept interactive API, PhotonSim ROOT readers, and
  `sparse_to_full`. Nothing breaks.
- `notebooks/` is **disposable**. Any breakage there is acceptable.
- `lucid/production/notebooks/` is the small set of production-facing
  notebooks; they get explicit disposition below.

---

## 2. Corrections to the main plan (still relevant)

1. **LUCiD does not yet read the new PhotonSim branches.**
   `lucid/sources/event_io.py:561` (branch list) and lines 712–725
   (segment dict) must pull `Segment_BetaStart` and `Segment_NCherenkov`.
   LUCiD does not consume these fields internally — pure pass-through
   to `seg/event_NNN/`.

2. **`particle_idx` derivation does not exist today.** Need a new
   helper `derive_particle_idx_per_track(event_dict)` that walks each
   track's `parent_id` chain to a categorized particle. Orphans get
   `-1`.

3. **t0 shift is not applied anywhere.** Current writers store
   G4-absolute times. `save_edep_event_v3` and `save_hits_event_v3` must
   subtract `t0` at write time. `save_sensor_event_v3` smears first,
   then subtracts t0.

4. **`qe_corrections` bug claim in main plan §2.3 is stale.** Guard at
   `lucid/simulation/simulator.py:174–183` covers the production path.
   Not a migration prerequisite.

---

## 3. `lucid/sources/event_io.py` — full rewrite of production surface

### 3.1 Delete (production-side only)

All of these are production batch I/O; none are used by
`good_notebooks/`, `lucid/optimization/`, `lucid/visualization.py`, or
tests. Verified.

| Function | Approx lines | Why |
|---|---|---|
| `save_sensor_batch_v2` | 2822–3021 | v2 flat-CSR writer |
| `save_segment_batch_v2` | 3024–3145 | v2 flat-CSR writer |
| `read_sensor_event_v2` | 3148+ | v2 reader |
| `merge_event_files` | 2143–2216 | per-event merge step; v3 doesn't merge |
| `save_single_event_with_extended_info` | 1869–1938 | production-side despite the name; wrote the `Q/T/P/V/PDG` format |
| `save_single_event_with_particle_info` | 1940–2142 | same |
| `read_event_file` | 2292+ | reads the `Q/T/P/V/PDG` format |
| `read_multi_folder_events` | 2218+ | reads multi-folder production output |
| `generate_multi_folder_events` | ~336 | writes multi-folder production output |
| `generate_events_from_photonsim` | ~732 | older photon-based production path; only caller is `generate_events.py` (also being deleted) |

Net deletion: ~1400 LoC.

### 3.2 Keep (untouched — these are not the production surface)

| Function | Reason |
|---|---|
| `save_single_event`, `load_single_event` | Interactive single-event API. Small file, simple schema, used by good_notebooks and optimization snapshots. Different tool, different job. |
| `read_photon_data_from_photonsim`, `read_particle_data_from_photonsim`, `read_photon_data_from_root` | Reads PhotonSim ROOT, not LUCiD HDF5. |
| `get_max_photons_per_particle`, `get_random_root_entry_index` | PhotonSim ROOT utilities. |
| `get_pdg_code`, `get_particle_mass`, `PARTICLE_MASSES` | Physics constants. |
| `extract_particle_properties`, `analyze_loaded_particle`, `analyze_event_directory`, `momentum_to_angles_and_energy`, `analyze_event_kinematics`, `print_event_kinematics` | Analysis helpers, no production schema dependency. |
| `full_to_sparse`, `sparse_to_full` | Compression utilities. |

### 3.3 Add

**v3 writers (4) + config writers (4) + readers (4) + 1 enumerator + 1 helper:**

- `save_sensor_event_v3`, `save_hits_event_v3`, `save_edep_event_v3`,
  `save_labl_event_v3`
- `write_sensor_config_v3`, `write_hits_config_v3`,
  `write_edep_config_v3`, `write_labl_config_v3`
- `read_sensor_event_v3`, `read_hits_event_v3`, `read_edep_event_v3`,
  `read_labl_event_v3`
- `list_events_v3(filename)` — returns `config/source_event_idx`
- `derive_particle_idx_per_track(event_dict)` — parent-chain walk:

  ```python
  def derive_particle_idx_per_track(event_dict):
      tracks = event_dict['meaningful_tracks']   # {track_id -> info}
      particles = event_dict['particles']
      id_to_idx = {p['genealogy'][-1]: i
                   for i, p in enumerate(particles) if p['genealogy']}
      out = np.full(len(tracks), -1, dtype=np.int32)
      for row, tinfo in enumerate(tracks.values()):
          cur = tinfo['track_id']
          while cur > 0:
              if cur in id_to_idx:
                  out[row] = id_to_idx[cur]; break
              parent = tracks.get(cur)
              if parent is None: break
              cur = parent['parent_id']
      return out
  ```

### 3.4 Modify

**Extend PhotonSim branch reads** (prerequisite for the v3 seg writer;
pass-through only, LUCiD does not consume these fields):

- Line 561: add `'Segment_BetaStart'`, `'Segment_NCherenkov'` to the
  branch list.
- Lines 712–725: add `beta_start` and `n_cherenkov` keys to the
  segment dict.
- Fail fast with explicit `ValueError` if branches absent:

  ```python
  missing = {'Segment_BetaStart', 'Segment_NCherenkov'} - set(tree.keys())
  if missing:
      raise ValueError(
          f"PhotonSim ROOT file is missing branches {sorted(missing)}. "
          f"Upgrade to PhotonSim branch 'add-per-segment-beta-ncherenkov' "
          f"(commit 1ef5ace or later)."
      )
  ```

- Shape assertion: both arrays must match `len(Segment_Edep)`.
- In `save_edep_event_v3`, cast to `LUCID_DATASET.md` dtypes
  (`float32` for `beta_start`, `int32` for `n_cherenkov`).

**Refactor `generate_events_from_photonsim_particles`** to open four
files per batch, write `config/` groups once, and loop events calling
`save_*_event_v3`. Drop the merge step. Apply t0 shift before calling
`save_edep_event_v3` and `save_hits_event_v3`.

**Remove voxel import** at line 1037 (from `lucid.production.voxelize`).
Voxels no longer on the save path. The module itself stays (see §4.1).

### 3.5 Provenance writer details

Attrs on every `config/` group per `LUCID_DATASET.md`:
`format_version`, `n_events`, `git_commit`, `run_id`, `dataset_name`,
`file_index`, `source_file`, `lucid_master_seed`, `photonsim_seed`.

Two implementation notes:

- `git_commit`: wrap `subprocess.check_output(['git', 'rev-parse', 'HEAD'])`
  in try/except; fall back to `os.environ.get('GIT_COMMIT', 'unknown')`
  for Singularity/S3DF.
- `source_file`: apply `os.path.abspath(...)` at the CLI entry before
  storing.
- `run_id`: auto-generated UUID4 in the CLI. Convention — one logical
  dataset per `dataset_root/` directory; concurrent writes to the same
  directory are not supported. Document this in `--help`.
- `photonsim_seed = -1` placeholder until PhotonSim exposes its RNG
  seed as a branch (independent future work).

### 3.6 Writer invariants (assert inline)

Across the four files, for each `event_NNN/`:

- Group names match (sequence position is the single source of truth).
- `source_event_idx` attr matches across all four files.
- `n_particles` attr matches between inst and labl.
- `n_tracks` attr matches between seg and labl.

---

## 4. `lucid/production/`

| File | Action | Notes |
|---|---|---|
| `generate_events_with_particles.py` | Modify | Drop `--merged-filename` and `--include-voxels`. Add `--dataset-name` (required), `--run-id` (optional UUID4). Update module docstring and argparse help. Output to `{sensor,inst,seg,labl}/` subdirs. ~25 LoC net. |
| `generate_events.py` | **Delete** | Non-canonical photon-based CLI. Its backend (`generate_events_from_photonsim`) is also deleted. ~150 LoC gone. |
| `visualize_particle_events.py` | Modify | Update group navigation to v3 per-event groups. Drop voxel import (line 15), voxel loading block (lines 638–658), voxel render block (lines 755–820), and voxel slider step (lines 900–909). Reindex remaining slider steps. ~150 LoC net. |
| `particle_data_utils.py` | Modify | Update dataset paths and names to v3 schema (`read_particle_event_file` remains, reads `inst/event_NNN/` and `labl/event_NNN/`). ~100 LoC. |
| `data_prod_utils.py` | **Delete** | Only consumers were `notebooks/` (disposable) and two production notebooks being migrated below. ~180 LoC gone. |
| `generate_validation_htmls.sh` | Modify | Line 156: swap `--merged-filename` for `--dataset-name`/`--run-id`. Adjust downstream paths. ~5 LoC. |
| `voxelize.py` | **Deleted** | Voxelization dropped entirely — superseded by per-segment `seg/` data. No remaining callers. |

No `lucid/production/__init__.py` exists — no package-level re-exports
to prune.

### 4.1 `lucid/production/notebooks/`

| Notebook | Action | Notes |
|---|---|---|
| `2D_event_visualization.ipynb` | Modify | Update reader call to v3. ~20 LoC. |
| `3D_event_visualization.ipynb` | Modify | Update reader call to v3. ~20 LoC. |
| `read_production_output.ipynb` | **Delete** | Demo of the old schema; replaced by v3 readers. |
| `voxel_visualization.ipynb` | **Deleted** | Removed alongside `voxelize.py` (its only dependency). |

---

## 5. `lucid/` shims

| File | Action | LoC |
|---|---|---|
| `lucid/generate.py` | Prune re-exports | Remove lines 41, 50, 51, 52, 53, 54 (six symbols: `generate_events_from_photonsim`, `save_single_event_with_extended_info`, `save_single_event_with_particle_info`, `merge_event_files`, `read_multi_folder_events`, `read_event_file`). |
| `lucid/utils.py` | Prune re-exports | Same six symbols at ~line 501-518, including the comment block. |

Under harsh mode, `good_notebooks/` imports via these shims still
resolve cleanly — they only use the kept symbols.

---

## 6. Unaffected (verified)

| Area | Why |
|---|---|
| `good_notebooks/` (all 22 notebooks) | Use only kept symbols (`save_single_event`, `load_single_event`, `read_photon_data_from_photonsim`, `sparse_to_full`). Verified with full import audit. |
| `notebooks/` (entire directory) | Disposable under harsh mode. Make no guarantees. |
| `lucid/optimization/`, `lucid/visualization.py`, `lucid/siren/`, `lucid/wavelength/`, `lucid/sources/calibration_sources.py`, `lucid/sources/particle_model.py`, `lucid/sources/siren_rays.py`, `lucid/geometry/`, `lucid/propagation/`, `lucid/simulation/` (rest), `lucid/gradient_analysis/`, `lucid/losses.py`, `lucid/detector_params.py`, `lucid/overlap.py` | No production HDF5 consumption; no deleted symbols referenced. |
| `tests/` | No production HDF5 consumption. Only `test_e2e_wavelength.py` imports from `event_io`, and only the kept `read_photon_data_from_photonsim`. |
| `s3df_jobs/`, `scripts/`, `baseline_scripts/`, `relax_test/`, `spatial_overlap_integrals/`, `tools/`, `config/`, `data/` | No production HDF5 references. |

---

## 7. Docs

| File | Action |
|---|---|
| `docs/LUCID_MIGRATION.md` | Update in place. Remove §8.1 "0 files in notebooks/" claim; drop §2.3 `qe_corrections` prerequisite wording; extend §5 with the LUCiD-side PhotonSim branch-read addition; fold in the harsh-mode delete list expansion. |
| `docs/EVENT_FORMAT_V2.md` | Annotate with a top banner: "Superseded by LUCID_DATASET.md. Retained for historical reference." |
| `docs/LUCID_DATASET.md` | Unchanged (authoritative v3 spec). |
| `docs/LUCID_MIGRATION_FILES.md` | This file. |

---

## 8. New tests

| File | Purpose | LoC |
|---|---|---|
| `tests/test_v3_writer_roundtrip.py` | Generate 3-event batch with v3 writers; verify every documented dataset/dtype/shape; read back with v3 readers; cross-file `source_event_idx` alignment. | ~200 |
| `tests/test_v3_particle_idx_derivation.py` | Walk fixture `parent_id` chains; assert writer `particle_idx` matches; orphan rate <1%. | ~100 |
| `tests/test_v3_time_frame.py` | Assert `seg.time - labl.per_event.t0 ≈ G4 Segment_Time`; assert `sensor.T`/`inst.T` within `[t0, t0+max_prop]`. | ~80 |
| `tests/test_v3_cherenkov_consistency.py` | `sum(seg.n_cherenkov per track) == labl.per_track.n_cherenkov`; `seg.beta_start ∈ (0, 1]`. | ~50 |
| `tests/test_qe_corrections_setup_broadcast.py` | Pin the existing simulator-setup broadcast behavior for scalar configs. | ~40 |

No existing test needs to change.

---

## 9. Execution order

1. Extend PhotonSim branch reads in `event_io.py:561, 712–725` to pull
   `Segment_BetaStart` and `Segment_NCherenkov`. Sanity-check against
   `PhotonSim/build/test_wavelength_5_events.root`.
2. Implement v3 writers + config writers + `derive_particle_idx_per_track`
   in `event_io.py`, including t0-shift and writer invariants.
3. Remove `lucid.production.voxelize` import from `event_io.py:1037`.
4. Refactor `generate_events_from_photonsim_particles` to call v3
   writers against four file handles.
5. Update CLI in `generate_events_with_particles.py` (drop flags, add
   flags, rewrite docstring).
6. Delete `lucid/production/generate_events.py`.
7. Add the 5 new test files from §8.
8. Implement v3 readers + `list_events_v3` in `event_io.py`.
9. Update `particle_data_utils.py` dataset paths for v3.
10. Update `visualize_particle_events.py` (v3 paths + voxel deletions +
    slider reindex).
11. Update `generate_validation_htmls.sh` CLI flags.
12. Migrate `2D_event_visualization.ipynb` and `3D_event_visualization.ipynb`
    to direct v3 reader calls. Delete `read_production_output.ipynb`.
13. Delete `data_prod_utils.py` and all §3.1 functions from `event_io.py`.
14. Prune `lucid/generate.py` and `lucid/utils.py` re-exports (six
    symbols each).
15. Annotate `docs/EVENT_FORMAT_V2.md` as superseded; update
    `docs/LUCID_MIGRATION.md` in place.
16. Run validation plan (§11 of main migration doc) end-to-end.
17. Update pimm-side LUCiD reader (separate repo) to consume v3.

---

## 10. Summary change table

| Area | Modified | Deleted | Added | Net LoC |
|---|---|---|---|---|
| `lucid/sources/event_io.py` | 1 | 0 | 0 | ~+200 (–1400 deleted, +1600 added) |
| `lucid/production/` | 5 | 2 (`generate_events.py`, `data_prod_utils.py`) | 0 | ~-380 |
| `lucid/production/notebooks/` | 2 | 1 (`read_production_output.ipynb`) | 0 | ~+40 |
| `lucid/` shims | 2 | 0 | 0 | -12 |
| `tests/` | 0 | 0 | 5 | ~+470 |
| Docs | 2 | 0 | 1 (this file) | — |
| **Total** | **~12** | **3** | **6** | **≈+320** |

Harsh mode is a net smaller footprint than the gentle-mode plan was —
the aggressive deletion of the production-side `save/read_*_single_*`
and `*_multi_folder_*` functions (~600 LoC total) more than offsets the
added v3 surface. `notebooks/` absorbs all the "decide" complexity by
being disposable.
