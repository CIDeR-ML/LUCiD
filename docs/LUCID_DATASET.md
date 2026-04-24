# LUCiD Dataset Layout

This document describes the concrete HDF5 schema for LUCiD water-Cherenkov
simulation output, produced by the
[LUCiD](https://github.com/...) production pipeline using PhotonSim ROOT files
as input.

The general design principles, the role of each file, and the supported ML
task patterns are documented separately in
[DATASET_DESIGN.md](DATASET_DESIGN.md). This document specifies the
LUCiD-specific instantiation of that design.

## Overview

LUCiD output for one batch consists of four parallel HDF5 files, all sharing
the same per-event indexing:

```
dataset_root/
├── sensor/    wc_sensor_NNNN.h5    — raw PMT readout (post-smearing)
├── inst/      wc_inst_NNNN.h5      — per-particle decomposition of PMT signal
├── seg/       wc_seg_NNNN.h5       — 3D track segments (Geant4 truth)
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
- **Time convention**: all times in `sensor/`, `inst/`, and `seg/` are stored
  in the **detector frame** — the per-event `t0` offset has been added at
  write time. The truth `t0` itself lives in `labl/event_NNN/per_event/t0`.
- **Cross-file alignment**: parallel filenames (`*_NNNN.h5`) and matching
  `event_NNN/` group names align modalities. Each `event_NNN/` carries an
  `source_event_idx` attribute that loaders cross-check.

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

### `inst.h5` — per-particle PMT decomposition

Pre-smearing per-particle contributions to the PMT signal. This is
the truth-level sensor decomposition by source particle.

```
inst.h5
├── config/
│   ├── attrs: provenance + n_sensors, detector_type, material
│   ├── source_event_idx      (n_events,) uint32
│   └── sensor_positions      (n_sensors, 3) float32   — duplicated for standalone use
└── event_NNN/
    │ attrs: source_event_idx, n_particles, n_particle_hits
    ├── particle_idx          (n_particle_hits,) int32   — local FK to labl/per_particle row
    ├── sensor_idx            (n_particle_hits,) uint16
    ├── PE                    (n_particle_hits,) float32   — pre-smearing per-particle
    └── T                     (n_particle_hits,) float32   — pre-smearing per-particle, detector frame
```

Notes:
- `particle_idx` is local to the event (`0..n_particles-1`).
- Within one event, multiple rows can share the same `(particle_idx, sensor_idx)`
  is not expected — particle decomposition is one entry per (particle, sensor)
  pair where PE > 0.
- Loaders that train on `inst/` alone use `sensor_positions` from this file's
  `config/` to map `sensor_idx` to physical positions.

### `seg.h5` — 3D Geant4 track segments

Per-segment truth from Geant4: trajectory geometry, energy deposition,
direction, kinetic state.

```
seg.h5
├── config/
│   ├── attrs: provenance + detector_type, material, detector_shape,
│   │           detector_bbox (6,), detector_radius, detector_half_height, detector_axis (3,)
│   └── source_event_idx      (n_events,) uint32
└── event_NNN/
    │ attrs: source_event_idx, n_tracks, n_segments
    ├── track_idx             (n_segments,) int32   — local FK to labl/per_track row
    ├── start_x, start_y, start_z   (n_segments,) float32  — meters
    ├── end_x, end_y, end_z         (n_segments,) float32  — meters
    ├── dir_x, dir_y, dir_z         (n_segments,) float16
    ├── time                       (n_segments,) float32   — ns, detector frame (t0-shifted)
    ├── edep                       (n_segments,) float32   — MeV
    ├── beta_start                 (n_segments,) float32   — particle β at segment start
    └── n_cherenkov                (n_segments,) int32     — Cherenkov photons emitted in segment
```

Notes:
- `track_idx` is local; ranges `0..n_tracks-1`. Joins to `labl/event_NNN/per_track/`.
- `beta_start` is the particle's β at the start of the segment — sufficient to
  reconstruct Cherenkov physics (opening angle, photon yield, spectrum) outside
  Geant4. See [DATASET_DESIGN.md](DATASET_DESIGN.md) for rationale.
- `n_cherenkov` is Geant4's exact count for this segment — direct truth label
  for forward-simulation ML.
- Segments are physically ordered by track (so segments belonging to track k
  appear contiguously), but the ordering is **not** required for correctness
  — `track_idx` is the canonical link. Loaders may shuffle segments freely.

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
    │   └── edep_containment                     ()   float32 — Σ(edep inside)/Σ(primary KE); NaN if denom=0
    ├── per_interaction/                         (1 row for non-pile-up; N rows for N-way pile-up)
    │   ├── source_type, t0, vertex_{x,y,z}       — see save_labl_event_v3 docstring
    │   ├── n_primaries, n_particles              — ints per interaction
    │   ├── neutrino_pdg, neutrino_energy_MeV     — GENIE-only, zeroed for particle-gun
    │   ├── edep_containment                     (n_interactions,) float32 — NaN if primary-KE sum=0
    │   └── CSR primary_{track_ids,pdgs,energies}_{offsets,data}
    ├── per_particle/                            (~8 rows for typical LUCiD events)
    │   ├── category                             (n_particles,) uint8
    │   ├── edep_containment                     (n_particles,) float32 — edep_inside/particle_KE; NaN if denom=0
    │   ├── genealogy_data                       (vlen int32) — categorized chain
    │   ├── genealogy_offsets                    (n_particles+1,) uint32
    │   ├── ext_genealogy_data                   (vlen int32) — full Geant4 chain
    │   └── ext_genealogy_offsets                (n_particles+1,) uint32
    └── per_track/                               (~600 rows for typical LUCiD events)
        ├── track_id            (n_tracks,) int32   — Geant4 track ID (truth metadata)
        ├── parent_id           (n_tracks,) int32
        ├── pdg                 (n_tracks,) int16   — raw PDG code
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
    decompose the PMT signal in `inst/`.
  - **Tracks** (`per_track`) are full Geant4 tracks — typically ~600 per
    event. The 3D truth in `seg/` is organized at this granularity.
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
   - Split current `sensor_events_*.h5` into `sensor/`, `inst/`, partial `labl/`.
   - Split current `segment_events_*.h5` into `seg/`, partial `labl/`.
   - Add per-event H5 group structure inside each output file.
   - Add provenance block to all four files.
   - Apply `t0` shift to `seg/event_NNN/time` at write time (currently stored
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
| `particle_hit_PE` | `inst/event_NNN/PE` |
| `particle_hit_T` | `inst/event_NNN/T` |
| `particle_hit_sensor_idx` | `inst/event_NNN/sensor_idx` |
| `particle_hit_offsets`, `particle_event_offset` | dropped — replaced by per-event grouping + `particle_idx` column |
| `sensor_positions` | `sensor/config/sensor_positions` + `inst/config/sensor_positions` |
| `t0` | `labl/event_NNN/per_event/t0` (shifted into all `T`/`time` at write time) |
| `n_particles` | attr on `event_NNN/` in inst, labl |
| `event_number` | renamed to `source_event_idx` — attr on every `event_NNN/`, plus `config/source_event_idx` array in every file |
| `particle_category` | `labl/event_NNN/per_particle/category` |
| `particle_containment` | `labl/event_NNN/per_particle/edep_containment` |
| `overall_containment` | `labl/event_NNN/per_event/edep_containment` |
| `genealogy_data`, `genealogy_offsets` | `labl/event_NNN/per_particle/genealogy_data`, `genealogy_offsets` |
| `ext_genealogy_data`, `ext_genealogy_offsets` | `labl/event_NNN/per_particle/ext_genealogy_data`, `ext_genealogy_offsets` |
| `start_x/y/z`, `end_x/y/z`, `dir_x/y/z` | `seg/event_NNN/start_*`, `end_*`, `dir_*` |
| `edep`, `time` | `seg/event_NNN/edep`, `time` (time now t0-shifted) |
| `track_id`, `parent_id`, `pdg`, `initial_energy`, `n_cherenkov` (per-track) | `labl/event_NNN/per_track/{track_id, parent_id, pdg, initial_energy, n_cherenkov}` |
| `segment_offset`, `track_event_offset` | dropped — replaced by per-event grouping + `track_idx` column |
| `master_seed` | `config/lucid_master_seed` (renamed for clarity) |
| `detector_config`, `detector_type`, `material`, `smearing_applied`, `format_version`, `source` | `sensor/config/` attrs |
| `root_file` | `source_file` provenance attr |
| Voxel datasets (`voxel_*`) | dropped — superseded by `seg/` |

Newly added fields are listed in "Production-side requirements" above.
