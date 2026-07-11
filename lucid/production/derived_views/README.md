# Derived views from a LUCiD dataset (no ROOT, no re-sim)

A **derived view** is a new labeling/representation produced by re-aggregating or re-grouping the
four stored HDF5 files (`sensor`, `hits`, `step`, `labl`) only — no PhotonSim ROOT, no re-running
the forward. This directory holds the ROOT-free transforms plus the non-obvious gotchas so they don't
have to be re-discovered.

Each script takes `SRC DST <param>` and writes a new dataset; run inside the container:

```bash
apptainer exec -B /sdf "$LUCID_IMAGE_PATH" python3 <script> SRC DST <param>
```

| Script | What it does | `<param>` |
|---|---|---|
| `depile.py` | Splits each pile-up event's N interactions into N separate single-interaction events (~2× events for the 2-vertex configs). | `FILE_INDEX` (one file per call; resumable) |
| `relabel_segments.py` | Particle re-categorization by deflection angle θ. | `θ` in degrees |
| `relabel_particles.py` | Merge tracks into particles by ancestry. | — |
| `relabel_cone.py` | Cherenkov-cone per-hit labels. | — |

`depile.py` is dataset-agnostic and processes one file index at a time (atomic `.part`→rename,
resumable — re-running skips files whose dest event count already matches). Drive it across a whole
dataset with the SLURM array in `submit_depile_array.sbatch` (a template — edit `SRC`/`DST`/`--array`;
the example de-piles `config_000018` → `config_000018_separated`).

## The core idea
The finest truth lives in `step/sensor_hits` = one row per **(segment, sensor, emission_process)**
with `PE`/`T`. Everything coarser is an aggregation of it:
- `sensor/` = sum over everything (per PMT) — the smeared readout.
- `hits/`   = grouped by particle (per particle, sensor, process).

So any view is a **group-by / filter / scatter-sum** over stored columns. Pick: what to **filter**
(e.g. `emission_process`), what to **group by** (`sensor_idx`, `particle_idx`, `segment_idx→track_idx`),
and how to **reduce** (PE→sum, T→min). (The `step` modality was formerly called `edep`; the per-segment
`edep` energy-deposit *value* dataset inside it is unchanged.)

## What drives each thing in the viewer (so you rewrite the right field)
| To change… | Rewrite in the dataset files |
|---|---|
| per-PMT Cherenkov vs scintillation | `hits/…/emission_process` and `step/…/sensor_hits/emission_process` (int8: 0=Cher, 1=scint) |
| **Particle** label coloring (PMTs) | `labl/per_track/particle_idx` (viewer maps segment→track→particle via this) |
| Particle **list/count + species label** | `labl/per_particle/*` (genealogy, category, …) **AND the `n_particles` attr** (see gotcha) |
| hit-level particle | `hits/…/particle_idx` |

- The viewer names a particle by its **genealogy leaf pdg** (`per_particle/genealogy_data` last entry),
  overridden to π⁰ if any chain track is pdg 111. So a π⁰→γ EM shower shows as **gamma** unless you
  fold it into the parent hadron.
- Event selector is **0-based** internally (`event_NNN`, `eventKey=padStart(3)`).

## ⚠ Gotchas (hard-won)
1. **`n_particles` attribute must be kept in sync.** If you rebuild `labl/per_particle` to a new row
   count, you MUST also set `h[event].attrs["n_particles"] = N`. The viewer trusts this attr and loops
   over it — a stale (larger) value makes it read past your table and render **phantom particles**
   (garbage "gamma/other" entries). Same for `hits`: update `n_particle_hits` (and `n_particles` if
   present). This was a real bug; symptom = "extra gamma particles that aren't in the file."
2. **Geometry & medium come from `--detector`, not the dataprod `material` field.** (Generation, not
   views, but bites the same way.) Default `SK_like` ⇒ water ⇒ Cherenkov-only. See
   `lucid/production/README.md`.
3. **What you can re-derive downstream is bounded by what's persisted:**
   - `step`/`per_track` cover only **meaningful tracks** (above-threshold light producers);
     sub-threshold and **neutral** tracks (e.g. π⁰) are **not** in the files.
   - The G4 **creator-process string** (Inelastic/Deflection/…) is consumed at generation and **not
     persisted**. So you cannot reproduce the literal process-based categorization downstream; use a
     **geometry/ancestry proxy** instead (deflection angle from `step/dir_*`, branching from
     `per_track.parent_id`). EM showers from a neutral π⁰ can't be re-attributed to their parent
     hadron from the files alone (the π⁰ link is missing).

## Recipe: segment-level particle re-categorization (the θ demo)
Redefine "particle" purely from `step` geometry + `per_track` ancestry, tunable by a deflection angle θ:
1. Per track, get start/end direction = `dir_*` of its earliest/latest `step` segment (by `time`).
2. Walk each track to its parent (`per_track.parent_id`):
   - EM child (e±, γ) → **absorb** into parent's particle.
   - hadron child → **same** particle if junction angle `arccos(end_dir(parent)·start_dir(child)) < θ`,
     else a **new** particle. Primary (`parent_id==0`) = root.
3. Connected components = particles (union-find). Dense-index them.
4. Write back: `per_track/particle_idx`; **rebuild** `per_particle` (category, contained,
   interaction_idx, one-entry genealogy = representative track_id) **and set `n_particles` attr**;
   re-aggregate `hits` from `step/sensor_hits` (group by new particle, sensor, emission_process;
   sum PE, min T) and set `n_particle_hits`.

Result: smaller θ → more particles (splits at every kink), larger θ → fewer.
`relabel_segments.py SRC DST θ`. Verified on the 50-π⁺ set: θ=5°→1/8/25, θ=15°→1/6/17,
θ=45°→1/4/10 particles/event.

## Serving the viewer (host-native, no container)
```bash
python3 LUCiD/viewer/serve_viewer.py <dataset_dir> --port 8765     # binds 127.0.0.1
```
Serves its UI from its own dir and the dataset from `<dataset_dir>` (auto-detects
`{sensor,hits,step,labl}/`). It reads files fresh per request → after relabeling, just hard-reload the
browser. On an S3DF tunnel from your laptop (replace node with the one from `hostname -f`):
```bash
ssh -J <user>@s3df.slac.stanford.edu -L 8765:127.0.0.1:8765 <user>@<node>.sdf.slac.stanford.edu
# then open http://127.0.0.1:8765/ , LABEL → Particle
```
Two viewers of the same events = two ports + two relabeled copies.

## Units
meters / nanoseconds / MeV throughout.
