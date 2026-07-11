# Dataset migration notes (internal)

Engineering self-audit from the schema migration — moved out of the public
dataset-schema reference. Kept for provenance.

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
| `n_particles` | attr on `event_NNN/` in hits, labl |
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
