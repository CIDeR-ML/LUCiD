# Derived views

Scripts that write a new dataset from an existing one by regrouping or relabelling its four
HDF5 files (`sensor`, `hits`, `step`, `labl`), with no PhotonSim ROOT file and no new simulation.
Each takes `SRC DST <param>`; run them in the container:

```bash
apptainer exec -B /sdf "$LUCID_IMAGE_PATH" python3 <script> SRC DST <param>
```

| script | what it does | `<param>` |
|---|---|---|
| `depile.py` | splits each pile-up event's N interactions into N single-interaction events; dark hits and trigger windows are dropped | `FILE_INDEX` (one batch per call) |
| `relabel_segments.py` | redefines particles by the deflection angle between parent and child tracks (below) | angle in degrees |
| `relabel_particles.py` | keeps the stored grouping of tracks into particles (`split`) or folds every track into its primary (`merge`) | `split` or `merge` |
| `relabel_cone.py` | labels each hit inside (1) or outside (0) a cone of the given half-angle around the primary direction, written into `emission_process` so the viewer's emission toggle shows it | half-angle in degrees |

`depile.py` writes each batch atomically and skips batches already complete, so it can be
rerun. `submit_depile_array.sbatch` runs it over a whole dataset as a Slurm array; it is a
template, so set `IMG`, `SCRIPT`, `SRC`, `DST`, the log paths and `--array` first.

The dataset format is in the [dataset schema](../../../docs/reference/dataset-schema.md).

## How views are built

The finest truth is `step/.../sensor_hits`: one row per (segment, sensor, emission process),
with `PE` and `T`. `hits/` is that grouped by (particle, sensor, process), and `sensor/` is the
digitized total per PMT. A view is therefore a filter, a group-by and a reduction (PE summed, T
minimum) over stored columns.

To change what the viewer shows, rewrite:

| to change | rewrite |
|---|---|
| Cherenkov vs scintillation per PMT | `hits/.../emission_process` and `step/.../sensor_hits/emission_process` (int8: 0 Cherenkov, 1 scintillation) |
| particle colouring of PMTs and segments | `labl/.../per_track/particle_idx` (the viewer maps segment to track to particle) |
| the particle list and species | `labl/.../per_particle/*` and the event's `n_particles` attribute |
| the particle of each hit | `hits/.../particle_idx` |

The viewer names a particle by the PDG code of the last track in its genealogy
(`per_particle/genealogy_data`), or π⁰ if any track in the chain is a π⁰.

## Constraints

- Keep the counts in step. When `labl/per_particle` gets a new number of rows, set the event's
  `n_particles` attribute to match: the viewer loops over that attribute, and a stale larger
  value shows particles that do not exist. In `hits/`, update `n_particle_hits` (and
  `n_particles` where present) the same way.
- Only what is stored can be regrouped. `step` and `per_track` hold the tracks that produced
  Cherenkov light and their ancestors; other tracks are not in the files. The Geant4 creator
  process is not stored either, so a process-based categorization cannot be reproduced; use
  geometry (`step/dir_*`) and ancestry (`per_track.parent_id`) instead.
- The medium of a dataset comes from the detector it was generated with, not the dataprod
  `material` field ([production](../README.md#medium)).

## `relabel_segments.py`

Defines particles from `step` geometry and `per_track` ancestry, with a deflection angle θ:

1. For each track, take the start and end directions (`dir_*` of its earliest and latest
   segment by `time`).
2. Walk each track to its parent (`per_track.parent_id`). An electromagnetic child (e±, γ)
   joins its parent's particle. A hadron child joins its parent's particle if the angle between
   the parent's end direction and the child's start direction is below θ, and starts a new
   particle otherwise. A primary (`parent_id == 0`) starts a particle.
3. The connected groups are the particles, indexed densely.
4. Write `per_track/particle_idx`; rebuild `per_particle` and set `n_particles`; re-aggregate
   `hits/` from `step/sensor_hits` (sum PE, minimum T) and set `n_particle_hits`.

A smaller θ gives more particles per event, a larger θ fewer.

## Viewing the result

Serving a dataset to the browser, including over an SSH tunnel, is in
[`viewer/README.md`](../../../viewer/README.md). The server reads the files on every request, so
a browser reload shows a relabelled dataset.
