# LUCiD Dataset Layout

This document describes the concrete HDF5 schema for LUCiD water-Cherenkov
simulation output, produced by the
[LUCiD](https://github.com/CIDeR-ML/LUCiD) production pipeline using PhotonSim ROOT files
as input.

The design splits detector output into four parallel HDF5 files -- raw sensor
readout, per-particle signal decomposition, 3D track segments, and labels --
so that each ML task can load only the modalities it needs. This document
specifies the LUCiD-specific instantiation of that design.

## Overview

LUCiD output for one batch consists of four parallel HDF5 files, all sharing
the same per-event indexing:

```
dataset_root/
├── sensor/    wc_sensor_NNNN.h5    — raw PMT readout (post-smearing)
├── hits/      wc_hits_NNNN.h5      — per-particle decomposition of PMT signal
├── step/       wc_step_NNNN.h5       — 3D track segments (Geant4 truth)
└── labl/      wc_labl_NNNN.h5      — labels, truth metadata, dimension tables
```

Each file is independently loadable. Each file contains a `config/` group
holding file-level metadata and a sequence of `event_NNN/` groups (NNN is the
sequence position in the file, zero-padded).

## Standard conventions

These apply across all four files:

- **Group naming**: `event_000/`, `event_001/`, ... — sequence positions, not
  canonical event IDs. The canonical PhotonSim entry index is the
  `source_event_idx` attribute on each group and listed in the file-level
  `config/source_event_idx` array.
- **Index locality**: every index column inside `event_NNN/` ranges over
  `[0, count_for_this_event)`. No global IDs are used as indices anywhere.
- **Foreign keys**: cross-table joins within an event use FK columns
  (`particle_idx`, `track_idx`) rather than CSR offset arithmetic.
- **Time convention**: all times in `sensor/`, `hits/`, and `step/` are stored
  in the **detector frame** — the per-event `t0` offset has been added at
  write time. The truth `t0` itself lives in `labl/event_NNN/per_event/t0`.
- **Cross-file alignment**: parallel filenames (`*_NNNN.h5`) and matching
  `event_NNN/` group names align modalities. Each `event_NNN/` carries an
  `source_event_idx` attribute that loaders cross-check.
- **Emission-process encoding**: per-row `emission_process: int8` columns on
  `hits/` and `step/sensor_hits/` tag which physical process produced each
  charge contribution. Enum: `0 = Cherenkov`, `1 = scintillation`. A given
  `(particle_idx, sensor_idx)` (or `(segment_idx, sensor_idx)`) pair can appear
  in multiple rows differing only in `emission_process` when both processes
  contribute. Cherenkov-only datasets carry the column with all-zeros so
  consumers can group/filter by it without a presence check. Pre-Phase-0
  datasets that lack the column read as all-zeros via the reader's backward-
  compat default.

## Provenance block (every file's `config/`)

The following attrs appear on `config/` of every file:

- `format_version` (int)
- `n_events` (int)
- `git_commit` (str) — LUCiD repository commit
- `run_id` (str) — unique batch identifier
- `dataset_name` (str)
- `file_index` (int) — NNNN of `wc_*_NNNN.h5`
- `source_file` (str) — input PhotonSim ROOT path
- `lucid_master_seed` (int) — JAX PRNG root seed for this batch
- `photonsim_seed` (int, optional) — Geant4 seed used when generating the
  source ROOT file. `-1` if not propagated.

Plus a per-file `source_event_idx (n_events,) uint32` array in `config/`
for integrity checks and fast event enumeration.

## File schemas

### `sensor.h5` — raw PMT readout

Post-smearing, t0-shifted. This is what the detector electronics would see.

```
sensor.h5
├── config/
│   ├── attrs: provenance + n_sensors, detector_type, material,
│   │           smearing_applied, smearing_charge_function, smearing_time_function
│   ├── source_event_idx      (n_events,) uint32
│   └── sensor_positions      (n_sensors, 3) float32
└── event_NNN/
    │ attrs: source_event_idx, n_hits
    ├── sensor_idx            (n_hits,) uint16
    ├── PE                    (n_hits,) float32   — post-smearing photoelectrons
    └── T                     (n_hits,) float32   — first-hit time, detector frame
```

Notes:
- `sensor_idx` is the PMT index (0..n_sensors-1).
- `PE` and `T` are sparse over PMTs that registered any hit in the event.
- Sensor file is the only file that needs smearing parameters in its config.

### `hits.h5` — per-particle PMT decomposition

Pre-smearing per-particle contributions to the PMT signal. This is
the truth-level sensor decomposition by source particle.

```
hits.h5
├── config/
│   ├── attrs: provenance + n_sensors, detector_type, material
│   ├── source_event_idx      (n_events,) uint32
│   └── sensor_positions      (n_sensors, 3) float32   — duplicated for standalone use
└── event_NNN/
    │ attrs: source_event_idx, n_particles, n_particle_hits
    ├── particle_idx          (n_particle_hits,) int32   — local FK to labl/per_particle row
    ├── sensor_idx            (n_particle_hits,) uint16
    ├── PE                    (n_particle_hits,) float32   — pre-smearing per-particle
    ├── T                     (n_particle_hits,) float32   — pre-smearing per-particle, detector frame
    └── emission_process      (n_particle_hits,) int8     — 0=Cherenkov, 1=scintillation
```

Notes:
- `particle_idx` is local to the event (`0..n_particles-1`).
- A `(particle_idx, sensor_idx)` pair can appear in **multiple rows** differing
  only in `emission_process` when both processes contribute to that sensor from
  that particle. Cherenkov-only datasets emit one row per pair, all with
  `emission_process = 0`. Aggregating PE/T over the `emission_process` axis
  recovers a process-agnostic per-(particle, sensor) view.
- Loaders that train on `hits/` alone use `sensor_positions` from this file's
  `config/` to map `sensor_idx` to physical positions.

### `step.h5` — 3D Geant4 track segments

Per-segment truth from Geant4: trajectory geometry, energy deposition,
direction, kinetic state.

```
step.h5
├── config/
│   ├── attrs: provenance + detector_type, material, detector_shape,
│   │           detector_bbox (6,), detector_radius, detector_half_height, detector_axis (3,)
│   └── source_event_idx      (n_events,) uint32
└── event_NNN/
    │ attrs: source_event_idx, n_tracks, n_segments, has_segment_sensor_map (always true for data-mode datasets)
    ├── track_idx             (n_segments,) int32   — local FK to labl/per_track row
    ├── start_x, start_y, start_z   (n_segments,) float32  — meters
    ├── end_x, end_y, end_z         (n_segments,) float32  — meters
    ├── dir_x, dir_y, dir_z         (n_segments,) float16
    ├── time                       (n_segments,) float32   — ns, detector frame (t0-shifted)
    ├── edep                       (n_segments,) float32   — MeV
    ├── beta_start                 (n_segments,) float32   — particle β at segment start
    ├── n_cherenkov                (n_segments,) int32     — Cherenkov photons emitted in segment
    ├── group_id                   (n_segments,) int32     — coarser segment grouping (see notes)
    ├── contained                  (n_segments,) bool      — both endpoints inside detector_bounds
    └── sensor_hits/                — flat per-(segment, sensor) PE/T (n_segment_hits rows)
        │ attrs: n_segment_hits
        ├── segment_idx           (n_segment_hits,) int32  — FK to this event's segments (0..n_segments-1)
        ├── sensor_idx            (n_segment_hits,) uint16 — FK to sensor_positions
        ├── PE                    (n_segment_hits,) float32 — segment's contribution to that sensor's PE
        ├── T                     (n_segment_hits,) float32 — segment's first-arrival time on that sensor (ns, detector frame)
        └── emission_process      (n_segment_hits,) int8    — 0=Cherenkov, 1=scintillation
```

Notes:
- `track_idx` is local; ranges `0..n_tracks-1`. Joins to `labl/event_NNN/per_track/`.
- `beta_start` is the particle's β at the start of the segment — sufficient to
  reconstruct Cherenkov physics (opening angle, photon yield, spectrum) outside
  Geant4. Storing per-segment beta avoids re-deriving it from the kinetic
  energy and mass, and keeps each segment self-contained for forward-simulation
  tasks.
- `n_cherenkov` is Geant4's exact count for this segment — direct truth label
  for forward-simulation ML.
- Each segment row is one raw G4 sub-step (PhotonSim ships unmerged
  segments; the prior in-PhotonSim merger now runs in Python at
  ingestion time — see `lucid/sources/segment_grouping.py`).
  `group_id` holds the merger's output: aggregating segments by
  `group_id` (start = first row's start, end = last row's end, edep
  and n_cherenkov summed, dir / beta_start / time = first row's value)
  reproduces the coarser pre-merge schema. Group ids are contiguous
  within an event (`0..n_groups-1`) and never re-used across tracks.
  Consumers should pick whichever granularity their task needs.
- Segments are physically ordered by track (so segments belonging to track k
  appear contiguously), but the ordering is **not** required for correctness
  — `track_idx` is the canonical link. Loaders may shuffle segments freely.
- `sensor_hits/` is the per-(segment, sensor) ground truth that `hits.h5`
  is aggregated from: aggregating `sensor_hits.PE` over `segment_idx →
  track_idx → particle_idx` reproduces `hits.h5/event_NNN/PE` exactly
  (column-sums match `sensor.h5/event_NNN/PE` by construction — shared
  `qe_weights` in `sensor_response.make_hits_per_segment`). Mandatory in
  data-mode datasets — re-bucketing into a different particle definition
  is a Python re-aggregation over the same on-disk file.

### `labl.h5` — labels and truth dimension tables

Multi-granularity truth: per-event scalars, per-particle metadata, per-track
metadata.

```
labl.h5
├── config/
│   ├── attrs: provenance + label_names
│   └── source_event_idx      (n_events,) uint32
└── event_NNN/
    │ attrs: source_event_idx, n_particles, n_tracks
    ├── per_event/
    │   ├── t0                                   ()   float32 — true emission time
    │   └── contained                            ()   bool    — True iff every meaningful segment is fully inside detector; False for empty events
    ├── per_interaction/                         (1 row for non-pile-up; N rows for N-way pile-up)
    │   ├── source_type, t0, vertex_{x,y,z}       — see save_labl_event_v3 docstring
    │   ├── n_primaries, n_particles              — ints per interaction
    │   ├── neutrino_pdg, neutrino_energy_MeV     — GENIE/supernova-only, zeroed for particle-gun
    │   ├── interaction_channel, channel          — supernova-only: sntools channel code (int) + name ("ibd"/"es"/"o16e"/"o16eb")
    │   ├── contained                            (n_interactions,) bool — AND over particles attributed to this interaction; False if interaction has no particles
    │   └── CSR primary_{track_ids,pdgs,energies}_{offsets,data}
    ├── per_particle/                            (~8 rows for typical LUCiD events)
    │   ├── category                             (n_particles,) uint8
    │   ├── contained                            (n_particles,) bool — AND over meaningful segments attributed to this particle; False if particle has no segments
    │   ├── genealogy_data                       (vlen int32) — categorized chain
    │   ├── genealogy_offsets                    (n_particles+1,) uint32
    │   ├── ext_genealogy_data                   (vlen int32) — full Geant4 chain
    │   └── ext_genealogy_offsets                (n_particles+1,) uint32
    └── per_track/                               (~600 rows for typical LUCiD events)
        ├── track_id            (n_tracks,) int32   — Geant4 track ID (truth metadata)
        ├── parent_id           (n_tracks,) int32
        ├── pdg                 (n_tracks,) int32   — raw PDG code (nuclear PDGs ~1e9 do not fit in int16)
        ├── initial_energy      (n_tracks,) float32 — MeV
        ├── n_cherenkov         (n_tracks,) int32   — total Cherenkov for this track
        ├── particle_idx        (n_tracks,) int32   — FK to per_particle row
        ├── ancestor            (n_tracks,) int32   — primary track_id at root of parent chain
        └── interaction         (n_tracks,) int32   — 0-based rank of ancestor among event primaries
```

Notes:
- `label_names` (config attr) is a small list of column names within
  `per_particle/` and `per_track/` that count as classification label schemes
  (as opposed to truth metadata or join keys). For LUCiD currently:
  `["category"]`. Added if more taxonomies are introduced (e.g.,
  `["category", "shower_or_track"]`).
- The two granularities reflect a real LUCiD distinction:
  - **Particles** (`per_particle`) are Geant4-categorized objects (Primary,
    DecayElectron, SecondaryPion, Gamma) — typically ~8 per event. Used to
    decompose the PMT signal in `hits/`.
  - **Tracks** (`per_track`) are full Geant4 tracks — typically ~600 per
    event. The 3D truth in `step/` is organized at this granularity.
- Each track maps to exactly one particle via `particle_idx`. The mapping is
  derived at write time by walking the parent chain from each track until it
  reaches a categorized particle. Verified injective on production data
  (>99.9% of tracks map cleanly; orphan housekeeping tracks are flagged as
  `particle_idx = -1`).
- `genealogy_data` is the compressed chain through categorized particles only.
  `ext_genealogy_data` is the full Geant4 chain through all intermediate
  meaningful tracks. Both are kept; they answer different questions.
- `track_id` is the Geant4 ID (sparse integers up to ~10⁶). It is **not** used
  as an index anywhere — purely truth metadata.

## Cross-modality joins

For an event:

```
inst hit (h)  ──particle_idx[h]──►  labl/per_particle row (p)
seg segment (s)  ──track_idx[s]──►  labl/per_track row (k)
labl/per_track row (k)  ──particle_idx[k]──►  labl/per_particle row (p)
```

All joins are local-integer-index lookups within the same event. No offset
arithmetic, no graph walking, no cross-event references.

## Tasks → files (LUCiD-specific)

| Task | sensor | inst | seg | labl |
|---|:-:|:-:|:-:|:-:|
| SSL on raw PMT readout | x | | | |
| SSL on per-particle PMT decomposition | | x | | |
| SSL on 3D segments | | | x | |
| Per-segment Cherenkov forward simulation (resimulate photons from segments) | | | x | |
| sensor → inst denoising / deconvolution | x | x | | |
| sensor → seg reconstruction (vertex, energy, direction) | x | | x | |
| Per-PMT semantic / instance segmentation | | x | | x |
| 3D semantic / instance segmentation on segments | | | x | x |
| Event classification or regression (energy, direction, vertex via primary) | x | x | | x |
| Containment-filtered training | | x or | x or | x (filter) |

## Production-side requirements

To produce this layout, the LUCiD writer (`lucid/sources/event_io.py`) needs:

1. **PhotonSim ROOT additions** (currently missing):
   - `Segment_BetaStart` per Geant4 step — already computed by Geant4 from
     pre-step momentum and total energy; only export is missing.
   - `Segment_NCherenkov` per Geant4 step — already counted by the Geant4
     Cherenkov process; only export is missing.

2. **LUCiD writer changes**:
   - Split current `sensor_events_*.h5` into `sensor/`, `hits/`, partial `labl/`.
   - Split current `segment_events_*.h5` into `step/`, partial `labl/`.
   - Add per-event H5 group structure inside each output file.
   - Add provenance block to all four files.
   - Apply `t0` shift to `step/event_NNN/time` at write time (currently stored
     as Geant4-absolute).
   - Compute `particle_idx` per track from the genealogy chain at write time.

3. **Discontinued from previous LUCiD output**:
   - All voxel datasets (`voxel_offsets`, `voxel_flat_indices`,
     `voxel_counts`, `voxel_n_nonzero`) — superseded by per-segment data.
   - All CSR offset arrays (`event_hit_offsets`, `particle_hit_offsets`,
     `particle_event_offset`, `segment_offset`, `track_event_offset`) —
     superseded by per-event grouping plus per-row FK columns.

## Index of fields preserved from current LUCiD format

Every dataset and attribute currently emitted by LUCiD has a documented
home in this layout. Mapping table:

| Currently saved | New location |
|---|---|
| `event_hit_PE` | `sensor/event_NNN/PE` |
| `event_hit_T` | `sensor/event_NNN/T` |
| `event_hit_sensor_idx` | `sensor/event_NNN/sensor_idx` |
| `event_hit_offsets` | dropped — replaced by per-event grouping |
| `particle_hit_PE` | `hits/event_NNN/PE` |
| `particle_hit_T` | `hits/event_NNN/T` |
| `particle_hit_sensor_idx` | `hits/event_NNN/sensor_idx` |
| `particle_hit_offsets`, `particle_event_offset` | dropped — replaced by per-event grouping + `particle_idx` column |
| `sensor_positions` | `sensor/config/sensor_positions` + `hits/config/sensor_positions` |
| `t0` | `labl/event_NNN/per_event/t0` (shifted into all `T`/`time` at write time) |
| `n_particles` | attr on `event_NNN/` in inst, labl |
| `event_number` | renamed to `source_event_idx` — attr on every `event_NNN/`, plus `config/source_event_idx` array in every file |
| `particle_category` | `labl/event_NNN/per_particle/category` |
| `particle_containment` | `labl/event_NNN/per_particle/contained` (bool) |
| `overall_containment` | `labl/event_NNN/per_event/contained` (bool) |
| `genealogy_data`, `genealogy_offsets` | `labl/event_NNN/per_particle/genealogy_data`, `genealogy_offsets` |
| `ext_genealogy_data`, `ext_genealogy_offsets` | `labl/event_NNN/per_particle/ext_genealogy_data`, `ext_genealogy_offsets` |
| `start_x/y/z`, `end_x/y/z`, `dir_x/y/z` | `step/event_NNN/start_*`, `end_*`, `dir_*` |
| `edep`, `time` | `step/event_NNN/edep`, `time` (time now t0-shifted) |
| `track_id`, `parent_id`, `pdg`, `initial_energy`, `n_cherenkov` (per-track) | `labl/event_NNN/per_track/{track_id, parent_id, pdg, initial_energy, n_cherenkov}` |
| `segment_offset`, `track_event_offset` | dropped — replaced by per-event grouping + `track_idx` column |
| `master_seed` | `config/lucid_master_seed` (renamed for clarity) |
| `detector_config`, `detector_type`, `material`, `smearing_applied`, `format_version`, `source` | `sensor/config/` attrs |
| `root_file` | `source_file` provenance attr |
| Voxel datasets (`voxel_*`) | dropped — superseded by `step/` |

Newly added fields are listed in "Production-side requirements" above.
