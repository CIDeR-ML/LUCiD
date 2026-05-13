# TTS Smearing & Production-Interm Merge

## Per-Photon TTS Model

Transit Time Spread (TTS) is now applied **per-photon before aggregation**, not per-sensor after.

**Before (production-interm):**
```
measured_time = min(t_1, t_2, ..., t_N) + epsilon
```
Single Gaussian draw per sensor after taking the first-arrival min. Same sigma regardless of how many photons hit the sensor.

**After (merged):**
```
measured_time_reco = min(t_1 + eps_1, t_2 + eps_2, ..., t_N + eps_N)
```
Each detected photon gets an independent TTS draw. The per-sensor first-arrival time is the order statistic of N independent Gaussians, producing correct N-dependent narrowing (high-PE sensors get tighter timing).

**Default:** sigma = 2.5 ns (SK 20-inch R3600, Fukuda et al. 2003). Configurable via `SimConfig.tts_sigma_ns`.

## Dual T_true / T_reco

The simulation kernel now computes **two** `segment_min` operations per call:

| Output | What it is | Stored in |
|--------|-----------|-----------|
| `time_true` | `min(t_i)` across detected photons (no TTS) | `inst/T`, `seg/sensor_hits/T` |
| `time_reco` | `min(t_i + eps_i)` with per-photon TTS | `sensor/T`, `inst/T_reco`, `seg/sensor_hits/T_reco` |

When `apply_smearing=False`, both are identical.

Post-hoc `smear_times(T_true)` calls in event_io.py have been **removed**. T_reco comes directly from the kernel.

## Return Value Changes

| Hit Mode | Old Return | New Return |
|----------|-----------|------------|
| `realistic` | `(charge, time)` | `(charge, time_true, time_reco)` |
| `per_segment` | `(charge, time, pe_seg, t_seg)` | `(charge, time_true, time_reco, pe_seg, t_seg_true, t_seg_reco)` |
| `aggregated` | `(charge, time)` | unchanged |
| `per_photon` | `(log_w, times, indices, charge)` | unchanged |
| `waveform` / `waveform_expected` / `shotgun_per_photon` | 3 values | unchanged |

## Key Assumptions

- TTS is a PMT property (independent per photoelectron), not a per-sensor property.
- The discriminator fires on the earliest detected PE: `min(t_i + eps_i)`.
- Charge smearing (`smear_charges_SK_like`) is still applied post-aggregation at the event level (PE is a count, not an order statistic).
- `inst/T` and `seg/sensor_hits/T` store **true** (unsmeared) times for truth-level analysis. The TTS-smeared view is in `T_reco`.
- A different particle can "win" first-arrival in reco vs truth (TTS can reorder arrivals). This is physically correct.

## What Else Changed in the Merge

**From production-interm:**
- V3 four-file HDF5 output schema (`sensor/`, `inst/`, `seg/`, `labl/`)
- Bucketed simulation with PAD_SIZE chunking (handles large showers)
- Pile-up event merging pipeline
- GENIE production chain (`run_job.py`, `run_genie.py`)
- Hierarchical RNG via `jax.random.fold_in` (collision-free seeds)
- WebGL event viewer
- In-kernel segment-sensor decomposition (`per_segment` hit mode)

**From refactor-v2 (kept):**
- Per-photon TTS with dual returns (this document)
- `cherenkov_qe` importance-sampled wavelength mode
- Waveform / shotgun hit modes
- `_project_missing_scalars` in detector_params (safe defaults from wavelength curves)
- QE curve bounds clamping for wavelength sampling range

**Bug fixes applied during merge:**
- `initial_survival` mask: photons originating outside the detector no longer leak hits at step 0
- `qe_corrections` scalar broadcast: prevents indexing errors when config provides a scalar placeholder
- Circular import in `utils.py`: backward-compat re-exports deferred via `__getattr__`
- Legacy ROOT reader fallback: `read_photon_data_from_photonsim` handles old files without `OpticalPhotonsRaw`
- Unit conversion consolidated: all data loaders convert mm to m at load time (no `/100.0` in simulator)
