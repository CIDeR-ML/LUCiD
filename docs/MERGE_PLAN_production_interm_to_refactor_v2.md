> **STATUS: COMPLETED.** This merge was executed (commit cfc9581). Retained for historical reference.

# Merge Plan: production-interm → refactor-v2

**Date:** 2026-05-15 (updated with audit findings)
**Branches:** `production-interm` (4b44d8a) → `refactor-v2` (a9f4744)
**Merge base:** 03ef0c9
**Scope:** 49 commits on production-interm, 34 commits on refactor-v2 since base

---

## 1. Executive Summary

Production-interm brings a major event_io rewrite (per-photon host aggregation,
segment grouping, particle categorization) and production pipeline tooling.
Refactor-v2 brings correct per-photon TTS physics, wavelength/QE infrastructure,
and shotgun simulation modes. The two branches diverged from a prior successful
merge (e027c43) and now conflict in 4 files:

| File | Conflict severity | Strategy |
|------|-------------------|----------|
| `lucid/simulation/sensor_response.py` | **High** | Manual merge — marry TTS with per-photon returns |
| `lucid/simulation/simulator.py` | **Medium** | Base on refactor-v2, apply production-interm renames |
| `lucid/sources/event_io.py` | **High** | Take production-interm, adapt for dual-time, remove host smearing |
| `.gitignore` | **Low** | Auto-merge or trivial manual |

**Design decisions:**
- No backwards compatibility for HDF5 schema changes
- No fallbacks — `smear_times` host-side calls for TTS are removed entirely
- Per-photon TTS in the kernel is the single source of truth for time smearing

---

## 2. Critical Physics: Per-Photon TTS

### What refactor-v2 implemented (MUST preserve)

Per-photon transit-time spread (TTS): each detected photoelectron gets
independent Gaussian time jitter **before** `segment_min`, correctly modeling
the N-PE-dependent narrowing of first-arrival distributions (order statistic).

- **Parameter:** `SimConfig.tts_sigma_ns` (default 2.5 ns, matching SK R3600 PMTs)
- **Dual returns:** `(measured_charge, measured_time_true, measured_time_reco)`
  - `T_true` = `segment_min` on unsmeared times (physics truth)
  - `T_reco` = `segment_min` on per-photon smeared times (what PMTs measure)
- **Applied in:** `make_hits_data`, `make_hits_per_segment`, all waveform factories
- **Mechanism:** `_qe_roll` returns 4 values (adds `detected_mask`, `timing_mask`),
  then `smear_times(flat_times, tts_sigma_ns, key)` is called per-photon

### What production-interm has (INCORRECT, must be replaced)

Per-sensor post-hoc smearing: `smear_times(detector_mins, key)` — applies one
random jitter per sensor after aggregation. This does not model the N-PE order
statistic effect. No `tts_sigma_ns` parameter. No dual T_true/T_reco.

---

## 3. `smear_times` Comprehensive Cleanup

This is the most important correctness concern in the merge. There are two
physically distinct smearing operations that were conflated under one function:

| Operation | Physics | Where it belongs | σ |
|-----------|---------|------------------|---|
| **Per-photon TTS** | Single-PE transit-time jitter | Inside `sensor_response.py` kernel, BEFORE `segment_min` | 2.5 ns (SK R3600) |
| **Per-sensor charge smearing** | Gain fluctuation | `smear_charges_SK_like` — stays as host-side call | f(PE) per SK model |

**`smear_times` for time smearing should ONLY live in the kernel.** All host-side
`smear_times()` calls for T_reco are removed — the kernel produces T_reco directly.

### 3.1 `smear_times` default value

| Branch | Default | Correct for |
|--------|---------|-------------|
| production-interm | `time_resolution=0.4` | Per-sensor post-aggregation (legacy, wrong model) |
| refactor-v2 | `time_resolution=2.5` | Per-photon TTS (correct physics) |

**After merge:** Take refactor-v2's default (2.5 ns) and docstring. The function
is now only called per-photon inside the kernel, where 2.5 ns is correct.

### 3.2 Every `smear_times` call site — exhaustive disposition

#### `lucid/simulation/sensor_response.py` (kernel — KEEP)

| Line (R2) | Call | Purpose | Action |
|-----------|------|---------|--------|
| 92 | `smear_times(flat_times, time_resolution=tts_sigma_ns, key=...)` in `make_hits_data` | Per-photon TTS before segment_min | **KEEP** — correct physics |
| 156 | Same pattern in `make_hits_per_segment` | Per-photon TTS before segment_min | **KEEP** (will be in renamed `make_hits_per_photon`) |

#### `lucid/simulation/sensor_response.py` (production-interm — REMOVE)

| Line (PI) | Call | Purpose | Action |
|-----------|------|---------|--------|
| 77 | `smear_times(detector_mins, key=...)` in `make_hits_data` | Post-aggregation per-sensor smear | **REMOVE** — replaced by per-photon TTS from R2 |
| 128 | `smear_times(detector_mins, key=...)` in `make_hits_per_photon` | Post-aggregation per-sensor smear | **REMOVE** — replaced by per-photon TTS from R2 |

#### `lucid/sources/event_io.py` (host-side — REMOVE)

| Line (PI) | Call | Purpose | Action |
|-----------|------|---------|--------|
| 1607 | `T_reco = smear_times(T_true, key=smear_t_key)` | Host-side post-hoc time smearing for single-vertex events | **REMOVE** — T_reco now comes from kernel's per-photon TTS |
| 2390 | `smear_times(jnp.asarray(T_true), key=smear_t_key)` | Host-side post-hoc time smearing for pile-up events | **REMOVE** — T_reco now comes from kernel |

**What replaces these:** The kernel (`make_hits_data` / `make_hits_per_photon`)
returns `measured_time_reco` directly. The event_io pipeline propagates this as
`T_reco_per_particle` through to the HDF5 writers (see Section 5).

**Note:** Refactor-v2's event_io.py has ALREADY removed these host-side
`smear_times` calls and gets T_reco from the kernel. Production-interm's
event_io.py still has them. The adaptation in Section 5 follows refactor-v2's
pattern.

#### `lucid/sources/event_io.py` — `smear_charges_SK_like` (KEEP)

| Line (PI) | Call | Purpose | Action |
|-----------|------|---------|--------|
| 1606 | `PE_reco = smear_charges_SK_like(PE_true, key=...)` | Charge gain smearing | **KEEP** — this is a separate physical effect (PMT gain fluctuation), not TTS |
| 2387 | `smear_charges_SK_like(jnp.asarray(PE_true), key=...)` | Charge gain smearing for pile-up | **KEEP** |

#### `lucid/simulation/sensor_response.py` — `smear_charges_SK_like` (KEEP)

| Line (R2) | Call | Purpose | Action |
|-----------|------|---------|--------|
| 109 | `smear_charges_SK_like(total_charge, key=...)` in `make_hits_data` | Per-sensor charge smearing | **KEEP** |
| 174 | Same in `make_hits_per_segment` | Per-sensor charge smearing | **KEEP** |
| 353 | In waveform builder | Per-photon gain variation | **KEEP** |

#### `lucid/utils.py` — function definition

| Line | Item | Action |
|------|------|--------|
| 666 (R2) | `def smear_times(times, time_resolution=2.5, key=None)` | **KEEP** — take R2's version with 2.5 default and updated docstring |
| 697 (R2) | `def smear_charges_SK_like(counts, key=None)` | **KEEP** — identical on both branches |

#### `tests/test_utils.py` — unit tests

All tests pass explicit `time_resolution` (0.1, 0.4, 2.0), so the default
change does not affect them. **No changes needed.**

#### `lucid/simulation/simulator.py` — import

| Line (R2) | Item | Action |
|-----------|------|--------|
| 14 | `from lucid.utils import smear_times, smear_charges_SK_like` | **KEEP** — still needed for waveform factories |

### 3.3 Summary: after merge, `smear_times` is called ONLY in:

1. `sensor_response.py:make_hits_data` — per-photon TTS (σ = `tts_sigma_ns`)
2. `sensor_response.py:make_hits_per_photon` — per-photon TTS (σ = `tts_sigma_ns`)
3. Waveform factory builders — per-photon TTS (σ from config)
4. `tests/test_utils.py` — unit tests with explicit σ

**Zero host-side `smear_times` calls remain in event_io.py.**

---

## 4. File-by-File Merge Plan

### 4.1 `lucid/simulation/sensor_response.py` — Manual Merge (HIGH priority)

**Base on:** refactor-v2 (has correct per-photon TTS + waveform factories)

**Changes to apply from production-interm:**

1. **Rename** `make_hits_per_segment` → `make_hits_per_photon`
2. **Change return signature** of `make_hits_per_photon`: instead of returning
   pre-aggregated `(n_segments, num_detectors)` tensors, return flat per-photon
   arrays for host-side groupby
3. **Drop `n_segments` param** from `make_hits_per_photon`

**Merged `make_hits_per_photon` return signature:**

```python
(measured_charge,          # (num_detectors,) — per-sensor total PE
 measured_time_true,       # (num_detectors,) — per-sensor min unsmeared time
 measured_time_reco,       # (num_detectors,) — per-sensor min TTS-smeared time
 qe_weights,              # (n_rays_bucket,) — per-photon QE weight (0 if failed)
 qe_filtered_times,       # (n_rays_bucket,) — per-photon unsmeared time
 qe_filtered_times_reco,  # (n_rays_bucket,) — per-photon TTS-smeared time
 flat_indices,            # (n_rays_bucket,) — per-photon sensor index
 flat_segment_idx)        # (n_rays_bucket,) — per-photon segment index
```

**Function-by-function:**

| Function | Action |
|----------|--------|
| `_qe_roll` | Keep R2's 4-return: `(qe_weights, qe_filtered_times, detected_mask, timing_mask)` |
| `make_hits_simulation` | No changes (identical) |
| `make_hits_data` | Keep R2 wholesale: per-photon TTS, dual T_true/T_reco, `tts_sigma_ns`, 3-value return |
| `make_hits_per_segment` → `make_hits_per_photon` | Rename. Port R2's TTS pattern. Return flat per-photon arrays (PI design) + `qe_filtered_times_reco`. Drop `n_segments`. |
| `make_hits_likelihood` | No conflict (identical) |
| `_resolve_first_detection` | Keep from R2 (new) |
| `build_make_hits_waveform` | Keep from R2 (new) |
| `build_make_hits_waveform_expected` | Keep from R2 (new) |
| `build_make_hits_per_photon_shotgun` | Keep from R2 (new) |


### 4.2 `lucid/simulation/simulator.py` — Base on refactor-v2 (MEDIUM priority)

**Base on:** refactor-v2 (TTS threading, QE/wavelength infra, waveform hit modes)

**Apply from production-interm:**

1. **Import rename:** `make_hits_per_segment` → `make_hits_per_photon`
2. **Drop `n_segments`** from 14 param-threading sites:
   - `_common_propagation` signature + `static_argnames`
   - All `_make_hits_*` wrapper signatures (4 wrappers)
   - `_simulation_with_data_impl` signature + `static_argnames`
   - Call to `_common_propagation` inside `_simulation_with_data_impl`
3. **Update `_make_hits_per_segment_fn`** body: call `make_hits_per_photon(...)`
   (dict key `'per_segment'` stays unchanged)

**Keep from R2 only:**

| Feature | Notes |
|---------|-------|
| `tts_sigma_ns=sim_config.tts_sigma_ns` threading | Lines 268, 276 |
| `waveform_config` + `wavelength_sampling` params | `setup_event_simulator` sig |
| Waveform + shotgun hit modes | Hit mode dict |
| QE curve bounds + importance sampling | `_get_optical_arrays` |
| Safe-positions gradient guard | `_common_propagation` |
| `_pgt = sim_config.K` default | Track-mode path |

**Note on `n_segments`:** Only remove from simulator/sensor_response parameter
chain. The `n_segments` used in `event_io.py` (segment counting),
`geometry/utils.py` (mesh segments), and `production/visualize_particle_events.py`
(3D viz) is **completely unrelated** — do NOT touch.


### 4.3 `lucid/sources/event_io.py` — Take production-interm (HIGH priority)

**Base on:** production-interm (29-commit rewrite, 72 functions)

**Cherry-pick from refactor-v2:**
- Legacy ROOT fallback in `read_photon_data_from_photonsim` (~line 778-832 R2)

**Dual-time adaptation** (detailed in Section 5)

**Host-side `smear_times` removal** (detailed in Section 3.2):
- Remove line 1607: `T_reco = smear_times(T_true, key=smear_t_key)`
- Remove line 2390: `smear_times(jnp.asarray(T_true), key=smear_t_key)`
- Keep `smear_charges_SK_like` calls (separate physical effect)


### 4.4 `lucid/simulation/config.py` — Additive (LOW priority)

Ensure `SimConfig.tts_sigma_ns: float = 2.5` exists. It's on R2 but not PI.
SimConfig uses keyword construction everywhere — adding a field with a default
is safe. No JSON configs serialize this field.


### 4.5 `lucid/utils.py` — Take refactor-v2's `smear_times` (LOW priority)

Take R2's version: default `time_resolution=2.5`, updated docstring explaining
per-photon vs per-sensor usage. `smear_charges_SK_like` is identical on both.


### 4.6 Tests — resolve version conflicts

| Test file | Action | Reason |
|-----------|--------|--------|
| `test_sensor_response.py` | Take R2 | Already expects 3-value `make_hits_data` return |
| `test_sensor_response_physics.py` | Take R2 | Already expects 3-value returns |
| `test_v3_writer_roundtrip.py` | Take PI | Has `group_id` tests matching PI's event_io |
| `test_inst_from_segments_byte_identity.py` | Take PI (new file) | Clean addition |
| `test_python_categorizer_byte_identity.py` | Take PI (new file) | Clean addition |
| `test_e2e_wavelength.py` | Add `elif n_outputs == 3:` branch | Data-mode now returns 3 values |
| `test_shotgun_*.py` (3 files) | Keep R2 (new files) | Clean additions |


### 4.7 `.gitignore` — Auto-merge (LOW priority)

Minor independent additions on both sides. Should merge cleanly.


### 4.8 Clean additions (NO conflicts)

**New source modules (PI):**
- `lucid/sources/particle_categorization.py` — only depends on numpy
- `lucid/sources/segment_grouping.py` — only depends on numpy

**New tests (PI):**
- `tests/test_inst_from_segments_byte_identity.py`
- `tests/test_python_categorizer_byte_identity.py`

**New production tooling (PI):**
- `lucid/production/configs/dataprod_18_pile_up_bombs.json`
- `lucid/production/s3df_jobs/jobs/dataprod_metrics_report.py`
- `lucid/production/s3df_jobs/siren_inputs/` (entire directory)
- `lucid/production/s3df_jobs/smax/` (entire directory)
- `good_notebooks/calibration_visualization.ipynb`

**New test + source files (R2):**
- `tests/test_shotgun_io.py`, `test_shotgun_source.py`, `test_shotgun_waveform.py`
- `tests/test_grid_off_wall.py`, `test_pmt_file_loader.py`, `test_qe_importance_sampling.py`

**Modified production files (PI, no conflict):**
- `lucid/production/generate_macro.py`, `run_job.py`, etc.
- `docs/LUCID_DATASET.md`, `LUCID_MIGRATION.md`, `QUICKSTART_S3DF.md`

### 4.9 `ci_tests/production_pipeline_perf.py` — Take production-interm

PI's version references `categorize_event` and `assign_group_ids` which match
PI's event_io.py. R2 removed those references. Since we take PI's event_io,
take PI's CI script.

### 4.10 `max_sensors_per_cell` naming

**CORRECTION:** The rename `max_sensors_per_cell` → `max_candidates_per_ray`
does NOT exist on either branch. Both use `max_sensors_per_cell` everywhere
(zero occurrences of `max_candidates_per_ray`). **No rename pass needed.**

---

## 5. Dual-Time Propagation Through event_io

This section details every variable and function in production-interm's
`event_io.py` that must be adapted to carry T_reco alongside T_true.

Refactor-v2's event_io already implements this pattern — use it as reference
(lines 191-378, 1529-1679 on R2).

### 5.1 `_trace_event_bucketed` (line 217)

**Current (PI):** Unpacks simulator 6-tuple:
```python
PE_chunk, T_chunk, qe_w_chunk, qe_t_chunk, sen_i_chunk, seg_i_chunk = result
```

**After merge:** Simulator returns 8-tuple (per_segment mode with TTS):
```python
PE_chunk, T_chunk, T_reco_chunk, qe_w_chunk, qe_t_chunk, qe_t_reco_chunk, sen_i_chunk, seg_i_chunk = result
```

**Changes:**
- Carry parallel `qe_t_reco_chunks` accumulator list
- Per-sensor T_reco accumulation: `t_reco_per_sensor = min(T_reco_chunk)`
- Return additional `photon_qe_time_reco` and `t_reco_per_sensor` arrays

### 5.2 `_aggregate_from_photon_records` (line 3554)

**Current:** Produces `T_per_particle` and `seg_hits['T']` via min-reduce on
`photon_qe_time`.

**After merge:** Accept `photon_qe_time_reco`. Produce:
- `T_reco_per_particle` via identical min-reduce on reco times
- `seg_hits['T_reco']` via identical min-reduce on reco times

### 5.3 `generate_events_from_photonsim_particles` (line ~1588)

**Current (PI lines 1604-1611):**
```python
PE_true = jnp.asarray(pe_per_sensor_np)
T_true  = jnp.asarray(t_per_sensor_np)
if apply_smearing:
    PE_reco = smear_charges_SK_like(PE_true, key=smear_pe_key)
    T_reco = smear_times(T_true, key=smear_t_key)   # ← REMOVE
```

**After merge:**
```python
PE_true = jnp.asarray(pe_per_sensor_np)
T_true  = jnp.asarray(t_per_sensor_np)
T_reco  = jnp.asarray(t_reco_per_sensor_np)   # ← from kernel
if apply_smearing:
    PE_reco = smear_charges_SK_like(PE_true, key=smear_pe_key)
else:
    PE_reco = PE_true
# T_reco already has per-photon TTS from kernel — no host smearing needed
```

### 5.4 `_derive_views_from_segments` (line 992)

**Current:** Passes `photon_records['qe_time']` through to
`photon_records_filtered`.

**After merge:** Also pass through `qe_time_reco`:
```python
photon_records_filtered = {
    'qe_weight':     photon_records['qe_weight'],
    'qe_time':       photon_records['qe_time'],
    'qe_time_reco':  photon_records['qe_time_reco'],   # ← ADD
    'sensor_idx':    photon_records['sensor_idx'],
    ...
}
```

### 5.5 Pile-up merger (line ~2380)

**Current (PI lines 2384-2390):**
```python
if apply_smearing:
    PE_reco = smear_charges_SK_like(PE_true, key=smear_pe_key)
    T_reco = smear_times(T_true, key=smear_t_key)   # ← REMOVE
```

**After merge:**
```python
# T_reco from kernel-aggregated T_reco_per_particle
masked_reco = np.where(T_reco_per_particle > 0, T_reco_per_particle, np.inf)
T_reco = np.min(masked_reco, axis=0)
T_reco = np.where(np.isfinite(T_reco), T_reco, 0.0).astype(np.float32)

if apply_smearing:
    PE_reco = smear_charges_SK_like(PE_true, key=smear_pe_key)
else:
    PE_reco = PE_true.copy()
# No smear_times call — T_reco already from kernel
```

### 5.6 HDF5 Writers

No backwards compatibility — add new datasets directly.

| Writer | Current schema | After merge |
|--------|---------------|-------------|
| `save_sensor_event_v3` (3872) | `PE`, `T` (from PE_reco/T_reco) | **No change** — already writes reco values |
| `save_hits_event_v3` (3903) | `PE`, `T` (from T_per_particle) | **Add `T_reco`** dataset from `T_reco_per_particle` |
| `save_edep_event_v3` (3946) | `sensor_hits/{PE, T}` | **Add `sensor_hits/T_reco`** from `seg_hits['T_reco']` |

### 5.7 t0 shift block

The t0 shift block (~line 1632) must also shift T_reco arrays:
```python
T_per_particle = np.where(T_per_particle > 0, T_per_particle + t0, T_per_particle)
T_reco_per_particle = np.where(T_reco_per_particle > 0, T_reco_per_particle + t0, T_reco_per_particle)
```

---

## 6. Risk Assessment

### 6.1 HIGH: Dual-time propagation completeness

Every function in the event_io chain must carry T_reco. Missing one produces
silent data corruption (T_reco stays zero or unshifted).

**Mitigation:** Section 5 enumerates every site. After implementation, grep:
```bash
grep -n "T_per_particle\b" lucid/sources/event_io.py
```
Every occurrence should have a matching `T_reco_per_particle` nearby.

### 6.2 HIGH: Return signature mismatches

**Broken call sites (from exhaustive audit):**

| # | File | Current (PI) | After merge | Fix |
|---|------|-------------|-------------|-----|
| 1 | `test_sensor_response.py:69` | `q, t = make_hits_data(...)` | Gets 3 | Take R2 version |
| 2 | `test_sensor_response.py:79` | `q, _ = make_hits_data(...)` | Gets 3 | Take R2 version |
| 3 | `test_sensor_response_physics.py:117` | `q, _ = make_hits_data(...)` | Gets 3 | Take R2 version |
| 4 | `test_sensor_response_physics.py:126` | `q, _ = make_hits_data(...)` | Gets 3 | Take R2 version |
| 5 | `event_io.py:352` | Unpacks 6-tuple (per_segment) | Gets 8-tuple | Adapt per Section 5.1 |

**Safe call sites (confirmed OK):**
- `make_hits_simulation` — 2-tuple on both branches, 12 call sites all OK
- `make_hits_likelihood` — 4-tuple on both branches, 9 call sites all OK
- `_common_propagation` — transparent pass-through, adapts automatically
- All R2-only tests already expect new signatures
- `baseline_scripts/L1_capture.py` (R2) — already expects 3-value return

### 6.3 MEDIUM: `_qe_roll` 4-value return

PI's `_qe_roll` returns 2, R2's returns 4. Only called by `make_hits_data` and
`make_hits_per_photon` — both rewritten in merge. **Contained.**

### 6.4 MEDIUM: `ci_tests/production_pipeline_perf.py` divergence

R2 removed categorization references; PI still has them. Take PI's version
(matches PI's event_io). Verify imports resolve after merge.

### 6.5 MEDIUM: `test_e2e_wavelength.py` data-mode branch

Uses `n_outputs = len(result)` with branches for 2 and 4. Data-mode now returns
3. Currently only tests track mode (4), so won't fail — but add `elif n_outputs == 3:`
for correctness.

### 6.6 LOW: JIT recompilation

Dropping `n_segments` from `static_argnames` changes cache keys. First run
recompiles. Expected, no action.

### 6.7 LOW: `test_v3_writer_roundtrip.py` divergence

Take PI's version (has `group_id` tests matching PI's event_io).

---

## 7. Merge Execution Order

### Phase 1: Git merge and conflict resolution

1. Checkout refactor-v2, ensure clean working tree
2. `git merge production-interm` — will report conflicts
3. Resolve `.gitignore` — take union of both sides
4. Resolve `sensor_response.py` — manual merge per Section 4.1
5. Resolve `simulator.py` — base on R2, apply renames per Section 4.2
6. Resolve `event_io.py` — take PI, cherry-pick legacy ROOT fallback
7. Resolve `test_v3_writer_roundtrip.py` — take PI
8. Verify `config.py` has `tts_sigma_ns` field

### Phase 2: smear_times cleanup

9. Remove host-side `smear_times()` calls from event_io.py (lines 1607, 2390)
10. Keep `smear_charges_SK_like` calls
11. Verify `smear_times` default is 2.5 ns in `utils.py`
12. Verify `smear_times` only called in sensor_response.py kernel + waveform + tests

### Phase 3: Dual-time adaptation

13. Adapt `_trace_event_bucketed` — unpack 8-tuple, carry `qe_t_reco`
14. Adapt `_aggregate_from_photon_records` — produce `T_reco_per_particle`, `seg_hits['T_reco']`
15. Adapt `generate_events_from_photonsim_particles` — use kernel T_reco
16. Adapt `_derive_views_from_segments` — pass through `qe_time_reco`
17. Adapt pile-up merger — merge `T_reco_per_particle`, drop host smearing
18. Adapt t0 shift block — shift T_reco arrays
19. Adapt HDF5 writers — add `T_reco` to inst.h5 and seg.h5

### Phase 4: Test resolution

20. Take R2 versions: `test_sensor_response.py`, `test_sensor_response_physics.py`
21. Add `elif n_outputs == 3:` to `test_e2e_wavelength.py`
22. Keep PI versions: `test_v3_writer_roundtrip.py`
23. Take PI CI script: `ci_tests/production_pipeline_perf.py`

### Phase 5: Verification

24. Run tests:
    ```bash
    pytest tests/test_sensor_response.py
    pytest tests/test_sensor_response_physics.py
    pytest tests/test_shared_propagator.py
    pytest tests/test_shared_propagator_differentiability.py
    pytest tests/test_inst_from_segments_byte_identity.py
    pytest tests/test_python_categorizer_byte_identity.py
    pytest tests/test_v3_writer_roundtrip.py
    pytest tests/test_e2e_wavelength.py
    pytest tests/test_utils.py
    ```
25. Grep verification:
    ```bash
    # Should find ZERO matches:
    grep -rn "make_hits_per_segment" lucid/simulation/
    grep -rn "n_segments" lucid/simulation/simulator.py lucid/simulation/sensor_response.py

    # Should find ZERO host-side smear_times in event_io:
    grep -n "smear_times" lucid/sources/event_io.py
    # (should only appear in comments, not calls)

    # Verify smear_times only called in kernel + tests:
    grep -rn "smear_times(" lucid/ tests/ --include="*.py"
    # Expected: sensor_response.py (2), test_utils.py (4), possibly waveform builders
    ```
26. Commit merge

---

## 8. Post-Merge Validation Checklist

- [ ] `make_hits_data` returns 3 values `(charge, t_true, t_reco)`
- [ ] `make_hits_per_photon` returns 8 values `(charge, t_true, t_reco, + 5 flat arrays)`
- [ ] `SimConfig.tts_sigma_ns` exists with default 2.5
- [ ] `_qe_roll` returns 4 values
- [ ] `smear_times` default is 2.5 ns in `utils.py`
- [ ] ZERO `smear_times()` calls in `event_io.py` (only comments)
- [ ] `smear_charges_SK_like` calls preserved in `event_io.py`
- [ ] `_trace_event_bucketed` unpacks 8-tuple and propagates T_reco
- [ ] `_aggregate_from_photon_records` produces `T_reco_per_particle`
- [ ] t0 shift applies to both `T_per_particle` and `T_reco_per_particle`
- [ ] HDF5 inst.h5 includes `T_reco` dataset
- [ ] HDF5 seg.h5 includes `sensor_hits/T_reco` dataset
- [ ] No remaining `make_hits_per_segment` references in `lucid/simulation/`
- [ ] No remaining `n_segments` param in `lucid/simulation/`
- [ ] `n_segments` still used correctly in `event_io.py` for segment counting
- [ ] All tests pass
- [ ] Waveform hit modes functional
- [ ] Legacy ROOT file fallback works
- [ ] `particle_categorization.py` and `segment_grouping.py` importable
- [ ] `ci_tests/production_pipeline_perf.py` runs without import errors

---

## 9. Addendum: Final Audit Corrections (2026-05-15)

Findings from three independent review agents verifying the plan against code.

### 9.1 CRITICAL: `lucid/optimization/pipeline.py:598` — 2-tuple unpack will crash

```python
true_data = jax.lax.stop_gradient(data_simulator(true_track, key, photon_data))
hit_counts, hit_times_raw = true_data   # ← CRASH: gets 3 values
```

`data_simulator` uses `hit_mode='realistic'` (`make_hits_data`) which now returns
3 values. This exists on BOTH branches and was never updated for TTS.

**Fix:** Unpack as `hit_counts, hit_times_raw, _hit_times_reco = true_data`.
Or use `hit_counts, _, hit_times_raw = true_data` if reco times should be used
for the observed data (physics decision — see 9.8).

**Also affects:** `lucid/optimization/run.py` which imports from `pipeline.py`.

### 9.2 CRITICAL: `lucid/simulation/__init__.py` — exports not updated

PI exports: `make_hits_simulation, make_hits_data, make_hits_likelihood`
R2 exports: above + `make_hits_per_segment, build_make_hits_waveform, build_make_hits_per_photon_shotgun`

After merge, must export:
- `make_hits_per_photon` (renamed from `make_hits_per_segment`)
- `build_make_hits_waveform`
- `build_make_hits_per_photon_shotgun`

**Add to Phase 1** as a new step after step 5.

### 9.3 CRITICAL: `tests/test_containers.py` and `tests/test_integration.py` — assertion conflicts

R2 changed `SimConfig` track-mode default for `effective_n_grad_iters` from 0
to `self.K`. PI tests assert `== 0`, R2 tests assert `== cfg.K`.

These files won't auto-conflict (different assertion values on different lines)
so the PI assertions will silently fail after merge.

**Fix:** Take R2 versions of `test_containers.py` and the
`test_effective_n_grad_iters_values` test in `test_integration.py`.

**Add to Phase 4.**

### 9.4 HIGH: 17+ notebooks unpack data simulator as 2-tuple

Many `good_notebooks/*.ipynb` do:
```python
hit_counts, hit_times_raw = true_data
```

After merge, data-mode returns 3 values. Affected notebooks include:
- `parameter_scans_1D.ipynb`, `parameter_scans_1D_v2.ipynb`,
  `parameter_scans_1D_likelihood.ipynb`
- `tracking_opt_with_gif.ipynb`, `tracking_opt_development.ipynb`,
  `tracking_opt_development_likelihood.ipynb`
- `data_vs_pred_hit_predictions.ipynb`
- `grad_loss_and_opt_in_2D.ipynb`, `grad_loss_and_opt_in_2D_likelihood.ipynb`
- `visualize_3D_track_optimization.ipynb`
- `cylinder_2D_displays.ipynb`, `event_hit_animation.ipynb`

**Fix:** Update all to unpack 3 values. Can be done as a batch
search-and-replace after merge.

### 9.5 HIGH: `generate_events_from_root` legacy function in event_io.py

Lines 646, 697 in PI's event_io.py unpack simulator as 2-tuple:
```python
muon_charges, muon_times = event_simulator(...)
```

This is a **separate function from** `generate_events_from_photonsim_particles`
(which Section 5 covers). The plan's Section 5 does NOT mention it.

**Fix:** Update unpacking to handle 3-value return, or determine if this
function is dead code and remove it.

### 9.6 MEDIUM: `baseline_scripts/L1_capture.py` not in execution order

PI line 174: `q_data, t_data = make_hits_data(...)` (expects 2).
R2 line 174: `q_data, t_data_true, t_data = make_hits_data(...)` (expects 3).

R2's version is correct. The plan mentions this as "safe" in Section 6.2 but
**never schedules it in the execution order**. Since both branches modified
this file, it WILL conflict and must be resolved (take R2).

**Add to Phase 1** step 5 area.

### 9.7 Corrections to counts and naming

| Plan claim | Actual | Correction |
|-----------|--------|------------|
| "4 wrappers" in simulator.py need `n_segments` dropped | 7 wrappers (includes waveform, waveform_expected, shotgun) | Fix count to 7 |
| "14 param-threading sites" for `n_segments` | ~20+ lines across ~13 logical sites | Approximate; use "all occurrences" instead of specific count |
| Phase 5 grep `make_hits_per_segment` expects ZERO | Will find `_make_hits_per_segment_fn` wrapper name | Either rename wrapper to `_make_hits_per_segment_mode` or adjust grep to `grep "make_hits_per_segment[^_]"` |
| Section 3.2 omits waveform builder `smear_times` calls | Waveform builders do inline TTS noise (`jax.random.normal * tts_sigma_ns`), not via `smear_times()` | Section 3.3 item 3 should say "inline TTS noise" not "smear_times call" |

### 9.8 Physics decision needed: T_true vs T_reco for optimization

`lucid/optimization/pipeline.py:generate_event_data()` generates "observed" data
that the optimizer fits against. After merge, the simulator returns both T_true
and T_reco. The optimization pipeline must decide:

- **Use T_reco** (with TTS smearing) — more realistic, matches what a real PMT
  measures. This is the physically correct choice for reconstruction studies.
- **Use T_true** (no smearing) — simpler, matches pre-TTS behavior.

**Recommendation:** Use T_reco for observed data, T_true for gradient computation.
This matches the real-world scenario where the optimizer reconstructs tracks
from PMT-measured (smeared) times.

### 9.9 Minor: dead `smear_times` import in R2 event_io.py

R2 line 1189: `from lucid.utils import smear_charges_SK_like, smear_times` —
`smear_times` is imported but never called. Remove the dead import when
cherry-picking into PI's event_io.py. Only import `smear_charges_SK_like`.

### 9.10 Minor: t0 shift block already handles T_reco on PI

Section 5.7 implies the T_reco shift is new, but PI already shifts T_reco
(from host-side smearing) at lines 1632-1636:
```python
np.add(T_reco, t0_f32, out=T_reco, where=T_reco > 0)
```

The actual change is sourcing T_reco from the kernel instead of host smearing,
not adding a new shift line. The shift block itself needs minimal modification.

---

## 10. Updated Execution Order (incorporating addendum)

### Phase 1: Git merge and conflict resolution

1. Checkout refactor-v2, ensure clean working tree
2. `git merge production-interm` — will report conflicts
3. Resolve `.gitignore` — take union of both sides
4. Resolve `sensor_response.py` — manual merge per Section 4.1
5. Resolve `simulator.py` — base on R2, apply renames per Section 4.2
   - Drop `n_segments` from ALL 7 wrapper signatures + all threading sites
   - Update `_make_hits_per_segment_fn` body to call `make_hits_per_photon`
6. **Update `lucid/simulation/__init__.py`** — export `make_hits_per_photon`,
   `build_make_hits_waveform`, `build_make_hits_per_photon_shotgun`
7. Resolve `event_io.py` — take PI, cherry-pick legacy ROOT fallback
8. Resolve `test_v3_writer_roundtrip.py` — take PI
9. Resolve `baseline_scripts/L1_capture.py` — take R2
10. Verify `config.py` has `tts_sigma_ns` field

### Phase 2: smear_times cleanup

11. Remove host-side `smear_times()` calls from event_io.py (lines 1607, 2390)
12. Remove dead `smear_times` import from event_io.py (keep only `smear_charges_SK_like`)
13. Keep `smear_charges_SK_like` calls
14. Verify `smear_times` default is 2.5 ns in `utils.py`

### Phase 3: Dual-time adaptation

15. Adapt `_trace_event_bucketed` — unpack 8-tuple, carry `qe_t_reco`
16. Adapt `_aggregate_from_photon_records` — produce `T_reco_per_particle`, `seg_hits['T_reco']`
17. Adapt `generate_events_from_photonsim_particles` — use kernel T_reco
    (t0 shift block already handles T_reco — just change the source)
18. Adapt `_derive_views_from_segments` — pass through `qe_time_reco`
19. Adapt pile-up merger — merge `T_reco_per_particle`, drop host smearing
20. Adapt HDF5 writers — add `T_reco` to inst.h5 and seg.h5
21. Adapt `generate_events_from_root` legacy function (lines 646, 697) —
    update 2-tuple unpacks or confirm dead code and remove

### Phase 4: Fix broken callers

22. Fix `lucid/optimization/pipeline.py:598` — unpack 3 values
    (use T_reco for observed data — see 9.8)
23. Take R2 versions: `test_sensor_response.py`, `test_sensor_response_physics.py`
24. Take R2 versions: `test_containers.py`, relevant parts of `test_integration.py`
25. Add `elif n_outputs == 3:` to `test_e2e_wavelength.py`
26. Take PI CI script: `ci_tests/production_pipeline_perf.py`
27. Update 17+ notebooks — batch replace `hit_counts, hit_times_raw = true_data`
    with 3-value unpack

### Phase 5: Verification

28. Run full test suite:
    ```bash
    pytest tests/test_sensor_response.py
    pytest tests/test_sensor_response_physics.py
    pytest tests/test_shared_propagator.py
    pytest tests/test_shared_propagator_differentiability.py
    pytest tests/test_inst_from_segments_byte_identity.py
    pytest tests/test_python_categorizer_byte_identity.py
    pytest tests/test_v3_writer_roundtrip.py
    pytest tests/test_e2e_wavelength.py
    pytest tests/test_utils.py
    pytest tests/test_containers.py
    pytest tests/test_integration.py
    pytest tests/test_shotgun_io.py
    pytest tests/test_shotgun_source.py
    pytest tests/test_shotgun_waveform.py
    pytest tests/test_grid_off_wall.py
    pytest tests/test_pmt_file_loader.py
    pytest tests/test_qe_importance_sampling.py
    ```
29. Grep verification:
    ```bash
    # No old function name (excluding internal wrapper name):
    grep -rn "make_hits_per_segment[^_]" lucid/simulation/
    # No n_segments in simulation layer:
    grep -rn "n_segments" lucid/simulation/simulator.py lucid/simulation/sensor_response.py
    # No host-side smear_times calls in event_io:
    grep -n "smear_times(" lucid/sources/event_io.py
    # Verify smear_times only in kernel + tests:
    grep -rn "smear_times(" lucid/ tests/ --include="*.py"
    # Check no 2-tuple unpacks of data simulator remain:
    grep -rn "hit_counts, hit_times" lucid/ good_notebooks/ --include="*.py"
    ```
30. Commit merge

---

## 11. Updated Post-Merge Validation Checklist

- [ ] `make_hits_data` returns 3 values `(charge, t_true, t_reco)`
- [ ] `make_hits_per_photon` returns 8 values `(charge, t_true, t_reco, + 5 flat arrays)`
- [ ] `SimConfig.tts_sigma_ns` exists with default 2.5
- [ ] `_qe_roll` returns 4 values
- [ ] `lucid/simulation/__init__.py` exports `make_hits_per_photon` + waveform builders
- [ ] `smear_times` default is 2.5 ns in `utils.py`
- [ ] ZERO `smear_times()` calls in `event_io.py`
- [ ] `smear_charges_SK_like` calls preserved in `event_io.py`
- [ ] `_trace_event_bucketed` unpacks 8-tuple and propagates T_reco
- [ ] `_aggregate_from_photon_records` produces `T_reco_per_particle`
- [ ] t0 shift applies to T_reco (already exists, just re-sourced from kernel)
- [ ] HDF5 inst.h5 includes `T_reco` dataset
- [ ] HDF5 seg.h5 includes `sensor_hits/T_reco` dataset
- [ ] `lucid/optimization/pipeline.py` unpacks 3 values from data simulator
- [ ] `generate_events_from_root` updated or removed
- [ ] No remaining `make_hits_per_segment[^_]` in `lucid/simulation/`
- [ ] No remaining `n_segments` param in `lucid/simulation/`
- [ ] `n_segments` still used correctly in `event_io.py` for segment counting
- [ ] `test_containers.py` assertions match R2's `effective_n_grad_iters` behavior
- [ ] All 17 test files pass
- [ ] All notebooks updated for 3-value unpack
- [ ] Waveform hit modes functional
- [ ] Legacy ROOT file fallback works
- [ ] `particle_categorization.py` and `segment_grouping.py` importable
- [ ] `ci_tests/production_pipeline_perf.py` runs without import errors
