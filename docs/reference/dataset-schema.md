# Dataset schema

`lucid-run-job` writes each batch as four HDF5 files with the same events in the same order, so
a task can load only what it needs:

```
<output-dir>/
├── sensor/  wc_sensor_NNNN.h5   recorded PMT digits (what the electronics record)
├── hits/    wc_hits_NNNN.h5     the digits' charge split by source particle
├── step/    wc_step_NNNN.h5     GEANT4 track segments, with their charge per sensor
└── labl/    wc_labl_NNNN.h5     labels and truth tables
```

`NNNN` is the batch's file index, `--job-id − 1`. Format version 6.

## Reading a batch

```python
from lucid.dataset.layout import batch_paths
from lucid.dataset.reader import (list_events, read_sensor_event, read_hits_event,
                                  read_step_event, read_labl_event)

paths = batch_paths('out/', 0)                 # {'sensor': Path, 'hits': ..., 'step': ..., 'labl': ...}
n = len(list_events(paths['sensor']))
sensor = read_sensor_event(paths['sensor'], 0)  # one event as a dict of arrays
labels = read_labl_event(paths['labl'], 0)      # nested: per_event, per_interaction, per_particle, per_track
```

The readers need only h5py. From the shell, `python scripts/inspect_dataset.py --path out/`
prints event counts, charge statistics and array shapes (`--event N` for one event). The
`viewer/` app (`python viewer/serve_viewer.py <dataset_dir> --open`) is an interactive 3D event
display of the same files; see `viewer/README.md`. `lucid/production/derived_views/` makes
derived datasets without re-simulating, such as splitting pile-up into single interactions.

## Conventions

- Groups `event_000/`, `event_001/`, ... are positions in the file. The PhotonSim entry of each
  is its `source_event_idx` attribute, also listed in `config/source_event_idx`. Events the
  trigger or selection drops are absent, so `source_event_idx` has gaps; join datasets on it,
  not on position.
- Indices inside an event run over `[0, count)` for that event. Joins use foreign-key columns
  (`particle_idx`, `track_idx`, `digit_idx`, `segment_idx`).
- Times in `sensor/`, `hits/` and `step/` are in the detector frame: the event's `t0` is
  already added. The truth `t0` is `labl/event_NNN/per_event/t0`. All absolute times are
  float64 (a supernova burst spans seconds at sub-ns resolution).
- `emission_process` (int8) on `hits/` and `step/sensor_hits/` rows: `0` Cherenkov, `1`
  scintillation, `2` dark noise (`hits/` only). A key such as `(particle_idx, sensor_idx,
  digit_idx)` can appear once per contributing process. Cherenkov-only datasets carry the column
  as all zeros.
- Lengths in m, energies in MeV, charge in photoelectrons, times in ns.

## File metadata (`config/` attributes, every file)

| attribute | meaning |
|---|---|
| `format_version` | 6 |
| `n_events` | events kept |
| `n_events_requested` | events generated (optional); the difference is the selection loss |
| `complete` | `False` while writing, `True` once every event is written; a killed job leaves `False` |
| `git_commit`, `git_dirty` | the LUCiD commit that wrote the file (from the checkout, or `/opt/VERSIONS` in the production image; `"unknown"` otherwise) and whether tracked files differed (optional) |
| `reflection_model` | `scalar_mix`, `scalar` or `angular` |
| `run_id`, `dataset_name`, `file_index` | batch identifier, dataset name, `NNNN` |
| `source_file` | the PhotonSim ROOT file |
| `lucid_master_seed` | the batch's JAX seed |
| `photonsim_seed` | the GEANT4 seed (optional; `-1` if unknown) |
| `trigger` | `"none"` without a trigger; otherwise `trigger_window_ns`, `trigger_n_thr`, `trigger_pad_before_ns`, `trigger_pad_after_ns` describe it |
| `selection`, `selection_min_physics_hits` | how interactions were kept: `"trigger"` or `"min_physics_hits"` with its threshold (optional) |
| `sn_direction`, `sn_ordering`, `sn_transformation` | supernova only: burst direction (unit vector), mass ordering, and the sntools transformation it maps to |

`config/source_event_idx` (`(n_events,)` uint32) lists the kept events' PhotonSim entries.

## `sensor/`: recorded digits

```
sensor.h5
├── config/
│   ├── attrs: metadata + n_sensors, detector_type, material, digitizer_model ("basic" | "ski" | "hk"),
│   │          smearing_applied, smearing_charge_function, smearing_time_function
│   ├── source_event_idx      (n_events,) uint32
│   └── sensor_positions      (n_sensors, 3) float32
└── event_NNN/  attrs: source_event_idx, n_hits (number of digits)
    ├── sensor_idx            (n_hits,) uint16    PMT index; repeats when a PMT has several digits
    ├── PE                    (n_hits,) float32   digit charge, after the charge resolution
    └── T                     (n_hits,) float64   digit time, detector frame
```

A digit is one recorded hit: charge integrated over a time window. One PMT can have several
digits when light arrives in separate bursts (delayed coincidence, pile-up, dark noise); the
`basic` digitizer gives one digit per hit PMT. With a trigger, only digits inside a trigger
window are kept, ordered by `(window, sensor_idx, T)`, so each window is a contiguous slice
(`labl/.../per_window`). The digitizer and trigger are described in
[digitizer and trigger](digitizer-and-trigger.md).

## `hits/`: charge by source particle

```
hits.h5
├── config/  attrs: metadata + n_sensors, detector_type, material; source_event_idx; sensor_positions
└── event_NNN/  attrs: source_event_idx, n_particles, n_particle_hits
    ├── particle_idx          (n_particle_hits,) int32    row of labl per_particle; -1 for dark noise
    ├── digit_idx             (n_particle_hits,) int32    the sensor/ digit it contributes to
    ├── sensor_idx            (n_particle_hits,) uint16
    ├── PE                    (n_particle_hits,) float32  contribution before charge resolution
    ├── T                     (n_particle_hits,) float64  first arrival of this contribution, detector frame
    ├── T_reco                (n_particle_hits,) float64  the same after the timing response (optional)
    └── emission_process      (n_particle_hits,) int8
```

Each row is one `(particle, sensor, digit, emission_process)` contribution. Summing `PE` over
the rows of one `(sensor_idx, digit_idx)` gives that digit's charge before smearing. Dark noise
rows have `emission_process = 2` and `particle_idx = -1`.

## `step/`: GEANT4 segments

```
step.h5
├── config/  attrs: metadata + detector_type, material, detector_shape, detector_bbox (6,),
│            detector_radius, detector_half_height, detector_axis (3,); source_event_idx
└── event_NNN/  attrs: source_event_idx, n_tracks, n_segments, has_segment_sensor_map
    ├── track_idx                  (n_segments,) int32    row of labl per_track
    ├── start_x, start_y, start_z  (n_segments,) float32  m
    ├── end_x, end_y, end_z        (n_segments,) float32  m
    ├── dir_x, dir_y, dir_z        (n_segments,) float16
    ├── time                       (n_segments,) float64  ns, detector frame
    ├── edep                       (n_segments,) float32  MeV
    ├── beta_start                 (n_segments,) float32  β at the segment start
    ├── n_cherenkov                (n_segments,) int32    Cherenkov photons GEANT4 emitted in it
    ├── group_id                   (n_segments,) int32    coarser grouping of consecutive segments
    ├── contained                  (n_segments,) bool     both ends inside the detector
    └── sensor_hits/  attrs: n_segment_hits
        ├── segment_idx            (n_segment_hits,) int32
        ├── digit_idx              (n_segment_hits,) int32
        ├── sensor_idx             (n_segment_hits,) uint16
        ├── PE                     (n_segment_hits,) float32
        ├── T                      (n_segment_hits,) float64
        ├── T_reco                 (n_segment_hits,) float64  (optional)
        └── emission_process       (n_segment_hits,) int8     0 or 1
```

- Each row is one GEANT4 step. Aggregating rows by `group_id` (start of the first, end of the
  last, `edep` and `n_cherenkov` summed, other fields from the first) gives the coarser
  segments; group ids are contiguous per event and never shared between tracks.
- `sensor_hits/` is the charge per `(segment, sensor, digit)`. Summing its `PE` over
  `segment_idx → track_idx → particle_idx` reproduces `hits/` exactly, and over everything a
  digit's charge; all three come from one deposit list. This allows regrouping charge by any
  particle definition without re-simulating.
- Rows are ordered by track, but `track_idx` is the link; order does not matter.

## `labl/`: labels and truth tables

```
labl.h5
├── config/  attrs: metadata + label_names (["category"]); source_event_idx
└── event_NNN/  attrs: source_event_idx, n_particles, n_tracks
    ├── per_event/
    │   ├── t0                 ()  float64   earliest interaction t0
    │   └── contained          ()  bool      every meaningful segment inside the detector
    ├── per_window/            triggered datasets only; attrs: n_windows
    │   ├── window_start       (n_win,)   float64  gate start, detector frame, ns
    │   ├── window_end         (n_win,)   float64
    │   └── digit_offsets      (n_win+1,) int32    window w is sensor/ digits [off[w], off[w+1])
    ├── per_interaction/       one row per interaction (several with pile-up)
    │   ├── source_type, t0, vertex_x, vertex_y, vertex_z
    │   ├── n_primaries, n_particles
    │   ├── neutrino_pdg, neutrino_energy_MeV     GENIE and supernova only; zero otherwise
    │   ├── interaction_channel                   sntools channel code; -1 otherwise
    │   ├── contained          (n_int,) bool
    │   └── primary_{track_ids,pdgs,energies}_{offsets,data}   per-interaction primaries, CSR
    ├── per_particle/          categorised particles, about 8 per event
    │   ├── category           (n_particles,) uint8
    │   ├── contained          (n_particles,) bool
    │   ├── genealogy_data, genealogy_offsets            chain through categorised particles
    │   ├── ext_genealogy_data, ext_genealogy_offsets    full GEANT4 chain
    │   └── interaction_idx    (n_particles,) int32      row of per_interaction
    └── per_track/             GEANT4 tracks, about 600 per event
        ├── track_id, parent_id  (n_tracks,) int32   GEANT4 ids (metadata, never used as indices)
        ├── pdg                  (n_tracks,) int32
        ├── initial_energy       (n_tracks,) float32 kinetic, MeV
        ├── dir_x, dir_y, dir_z  (n_tracks,) float32 initial direction; zero if the track has no segments
        ├── n_cherenkov          (n_tracks,) int32
        ├── particle_idx         (n_tracks,) int32   row of per_particle; -1 if none
        ├── ancestor             (n_tracks,) int32   track_id of its primary
        └── interaction          (n_tracks,) int32   rank of that primary among the event's primaries
```

Particles are GEANT4 tracks grouped into categories (primary, decay electron, secondary pion,
gamma, ...). A reconstruction target, a primary with its meaningful decay products, is a
particle, which is why `hits/` splits charge by `particle_idx`. Each track maps to the
particle reached by walking its parent chain.

Joins within an event:

```
hits row        ──particle_idx──►  labl per_particle row
step segment    ──track_idx─────►  labl per_track row
per_track row   ──particle_idx──►  labl per_particle row
```

## Which files a task needs

| task | sensor | hits | step | labl |
|---|:-:|:-:|:-:|:-:|
| models of the raw readout | x | | | |
| per-particle charge | | x | | |
| 3D segments, or re-simulating photons from segments | | | x | |
| denoising readout into per-particle charge | x | x | | |
| vertex, energy, direction from readout | x | | x | |
| per-PMT segmentation | | x | | x |
| 3D segmentation | | | x | x |
| event classification or regression | x | x | | x |
