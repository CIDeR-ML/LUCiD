# Event Output Format V2

> **Superseded by [LUCID_DATASET.md](LUCID_DATASET.md) for v3 output.**
> This file is retained for historical reference only. V2 writers were
> removed from `lucid/sources/event_io.py` during the v3 migration; any
> surviving v2 batches should be treated as read-only archival data.

## Overview

Production events are stored in two separate HDF5 files per batch:
- **Sensor file** (`sensor_events_NNNN.h5`) — detector response and per-particle labels. Read every training step.
- **Segment file** (`segment_events_NNNN.h5`) — Geant4 track truth and trajectory segments. Read on demand for validation.

Both files use **flat arrays** across all events in the batch (no per-event HDF5 groups), **sparse CSR** for sensor data, **gzip compression**, and **minimal types**. This eliminates the ~15 KB/event HDF5 metadata overhead from the V1 group-per-event layout.

Files are linked by `event_number` and share the same batch index `NNNN`.

---

## Sensor File

### Production Metadata (File Attributes)

| Attribute | Type | Description |
|---|---|---|
| `format_version` | int | `2` |
| `n_events` | int | Number of events in this file |
| `n_sensors` | int | Total sensor count for this detector geometry |
| `source` | string | e.g. `'PhotonSim_Particles_VMAP'` |
| `detector_config` | string | Path to detector geometry JSON used for generation |
| `material` | string | Medium material (e.g. `'water'`) |
| `detector_type` | string | Geometry type (e.g. `'cylinder'`) |
| `smearing_applied` | bool | Whether charge/time smearing was applied |
| `master_seed` | int | RNG seed used for generation (reproducibility) |
| `root_file` | string | PhotonSim ROOT file path used as input |

### Sensor Positions

Stored once per file. Maps sensor index to physical coordinates so the file is self-contained.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `sensor_positions` | float32 | `(n_sensors, 3)` | XYZ coordinates of each sensor in meters |

### Event Header

One entry per event. Fixed-size arrays.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `event_number` | uint32 | `(n_events,)` | Global event index |
| `n_particles` | uint16 | `(n_events,)` | Number of categorized particles in this event |
| `t0` | float32 | `(n_events,)` | Event time offset in ns, sampled U(-15, 15) |
| `overall_containment` | float32 | `(n_events,)` | Fraction of photons inside detector volume |

### Particle Metadata

One entry per particle across all events. Use `particle_event_offset` to find which particles belong to which event.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `particle_event_offset` | uint32 | `(n_events + 1,)` | CSR offsets: event `i` owns particles `[offset[i], offset[i+1])` |
| `particle_category` | uint8 | `(total_particles,)` | Category: 0=Primary, 1=DecayElectron, 2=SecondaryPion, 3=Gamma, 255=Unknown |
| `particle_containment` | float32 | `(total_particles,)` | Per-particle light containment fraction |
| `genealogy_offsets` | uint32 | `(total_particles + 1,)` | CSR offsets into `genealogy_data` |
| `genealogy_data` | int32 | `(total_genealogy_entries,)` | Flat array of track IDs forming each particle's ancestry chain |

### Event-Level Sensor Data (Sparse)

Observed detector response after summing all particles and applying smearing. Only hit sensors are stored.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `event_hit_offsets` | uint32 | `(n_events + 1,)` | CSR offsets: event `i` owns hits `[offset[i], offset[i+1])` |
| `event_hit_sensor_idx` | uint16 | `(total_event_hits,)` | Sensor ID for each hit (max 65535) |
| `event_hit_PE` | float32 | `(total_event_hits,)` | Smeared photoelectron count at this sensor |
| `event_hit_T` | float32 | `(total_event_hits,)` | Smeared first-hit time in ns at this sensor |

**Hit mask:** A sensor is stored if `PE > 0` OR if it has a finite meaningful time (`0 < T < 1e5`). This captures the rare case where smearing clips PE to 0 while T remains valid.

### Per-Particle Sensor Data (Sparse)

True (unsmeared) per-particle decomposition. Same sparsity pattern for PE and T — indices are shared.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `particle_hit_offsets` | uint32 | `(total_particles + 1,)` | CSR offsets: particle `j` owns hits `[offset[j], offset[j+1])` |
| `particle_hit_sensor_idx` | uint16 | `(total_particle_hits,)` | Sensor ID for each hit |
| `particle_hit_PE` | float32 | `(total_particle_hits,)` | True (unsmeared) PE at this sensor from this particle |
| `particle_hit_T` | float32 | `(total_particle_hits,)` | True first-hit time in ns at this sensor from this particle |

### Voxel Data (Optional)

Sparse 3D voxelization of photon emission positions, stored when `--include-voxels` is enabled. One set per particle.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `voxel_particle_offsets` | uint32 | `(total_particles + 1,)` | CSR offsets: particle `j` owns voxels `[offset[j], offset[j+1])` |
| `voxel_flat_indices` | int64 | `(total_nonzero_voxels,)` | Linearized 3D voxel index (ravel of ix, iy, iz) |
| `voxel_counts` | int32 | `(total_nonzero_voxels,)` | Photon count in each occupied voxel |

Voxel grid parameters stored as attributes: `voxel_grid_size_m`, `voxel_size_m`, `voxel_n_per_dim`.

---

## Segment File

### Event-to-Track Mapping

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `event_number` | uint32 | `(n_events,)` | Matches sensor file for joining |
| `track_event_offset` | uint32 | `(n_events + 1,)` | CSR offsets: event `i` owns tracks `[offset[i], offset[i+1])` |

### Track Table

One entry per meaningful track (tracks that produced Cherenkov photons).

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `track_id` | int32 | `(total_tracks,)` | Geant4 track ID (up to ~1.2M) |
| `parent_id` | int32 | `(total_tracks,)` | Parent track ID |
| `pdg` | int16 | `(total_tracks,)` | PDG particle code (-13 to 211 for typical particles) |
| `initial_energy` | float32 | `(total_tracks,)` | Kinetic energy at track creation in MeV |
| `n_cherenkov` | int32 | `(total_tracks,)` | Number of Cherenkov photons from this track |
| `segment_offset` | uint32 | `(total_tracks + 1,)` | CSR offsets into segment arrays |

### Segment Table

One entry per trajectory segment.

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `start_x` | float32 | `(total_segments,)` | Segment start X position in cm |
| `start_y` | float32 | `(total_segments,)` | Segment start Y position in cm |
| `start_z` | float32 | `(total_segments,)` | Segment start Z position in cm |
| `end_x` | float32 | `(total_segments,)` | Segment end X position in cm |
| `end_y` | float32 | `(total_segments,)` | Segment end Y position in cm |
| `end_z` | float32 | `(total_segments,)` | Segment end Z position in cm |
| `dir_x` | float16 | `(total_segments,)` | Momentum direction X at pre-step (unit vector) |
| `dir_y` | float16 | `(total_segments,)` | Momentum direction Y at pre-step (unit vector) |
| `dir_z` | float16 | `(total_segments,)` | Momentum direction Z at pre-step (unit vector) |
| `edep` | float32 | `(total_segments,)` | Energy deposited in MeV |
| `time` | float32 | `(total_segments,)` | Time at pre-step in ns |

**Note on direction:** Direction is the particle's momentum direction at the pre-step point, NOT the geometric direction from start to end. These differ when the particle scatters within the segment. float16 is sufficient for unit vectors (max error ~0.0002).

**Note on end positions:** End positions are kept even though segments within a track are continuous (`end[i] == start[i+1]`), because segments from different tracks are interleaved and the last segment of each track needs its endpoint.

### Extended Genealogy

Per-particle full track ancestry chain (all meaningful track IDs from primary down to this particle).

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `ext_genealogy_offsets` | uint32 | `(total_particles + 1,)` | CSR offsets |
| `ext_genealogy_data` | int32 | `(total_ext_entries,)` | Flat array of track IDs |

### File Attributes

| Attribute | Type | Description |
|---|---|---|
| `format_version` | int | `2` |
| `n_events` | int | Number of events in this file |

---

## Derived Quantities (not stored, computable)

These can be derived from the stored data and are therefore not saved:

| Quantity | How to compute |
|---|---|
| NSegments per track | `segment_offset[i+1] - segment_offset[i]` |
| Segment length | `sqrt((end - start)^2)` |
| dE/dx | `edep / segment_length` |
| PE_true (event-level, unsmeared) | `sum(particle_hit_PE)` grouped by event |
| T_true (event-level, unsmeared) | `min(particle_hit_T)` grouped by event |
| Dense PE_per_particle | Scatter `particle_hit_PE` into `(n_particles, n_sensors)` using indices |

---

## Type Rationale

| Choice | Reason |
|---|---|
| float32 for PE/T values | PE has fractional values from smearing (min ~0.95). T at late times (>2000 ns) needs >0.4 ns precision (SK TTS); float16 loses ~1 ns there. |
| uint16 for sensor indices | Max sensor count across detectors is ~40K (HK). Fits uint16 (max 65535). |
| uint16 for n_particles | Conservative headroom over uint8. |
| uint8 for category | Only 5 values: 0-3 and 255 (unknown). |
| float16 for segment direction | Unit vectors in [-1, 1]. float16 error is ~0.0002, adequate. |
| float32 for segment positions | Positions range ±1900 cm. float16 step at 1000 cm is 1.0 cm = 10 mm, but average segment length is 1.5 mm. Would corrupt geometry. |
| float32 for segment edep | Has values < 0.001 MeV. float16 would lose small deposits. |
| float32 for segment time | Range up to 150,000 ns. float16 max is 65,504 — would overflow. |
| int32 for track_id/parent_id | Geant4 track IDs reach ~1.2M. |
| int16 for PDG | Standard codes for relevant particles: -13 to 211. |
| int32 for n_cherenkov | Counts up to ~410K. |

---

## Compression

All datasets use `compression='gzip', compression_opts=4`. The sparse format with gzip is particularly effective on per-particle data: PE_per_particle compresses ~27x as dense gzipped (mostly zeros), but the sparse format avoids storing zeros entirely.

---

## Reading Events

To reconstruct event `i` from the sensor file:

```python
# Event header
t0_i = t0[i]
n_part_i = n_particles[i]

# Particles for this event
p_start = particle_event_offset[i]
p_end = particle_event_offset[i + 1]
categories_i = particle_category[p_start:p_end]

# Event-level hits
h_start = event_hit_offsets[i]
h_end = event_hit_offsets[i + 1]
sensor_ids = event_hit_sensor_idx[h_start:h_end]
pe_values = event_hit_PE[h_start:h_end]

# Reconstruct dense PE array
PE_dense = np.zeros(n_sensors)
PE_dense[sensor_ids] = pe_values

# Per-particle hits
for j in range(p_start, p_end):
    ph_start = particle_hit_offsets[j]
    ph_end = particle_hit_offsets[j + 1]
    pp_sensors = particle_hit_sensor_idx[ph_start:ph_end]
    pp_pe = particle_hit_PE[ph_start:ph_end]
```
