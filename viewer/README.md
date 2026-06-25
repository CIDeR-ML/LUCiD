# LUCiD Event Viewer

Browser-based interactive viewer for LUCiD v3 production HDF5 output:
raw PMT readout (`sensor`), per-particle decomposition (`inst`), Geant4
truth segments (`seg`), and truth labels (`labl`). Supports cylinder,
box, and sphere geometries.

Modeled on the [JAXTPC viewer](../../../JAXTPC/viewer/) — Three.js 3D +
Canvas2D unwrapped 2D — adapted to water-Cherenkov PMT displays with
particle-level correspondence.

## Files

| File | Role |
|---|---|
| `serve_viewer.py`     | Local HTTP server with byte-range support |
| `make_test_data.py`   | Stub v3 dataset generator (for pre-migration smoke tests) |
| `index.html`          | Viewer entry point |
| `viewer.css`          | Styles |
| `viewer.js`           | Main module (Three.js + Canvas2D + UI) |
| `shaders.js`          | Single point-sprite GLSL material, used for PMTs and segment trajectories |
| `colormaps.js`        | Colormap stops + hashed-hue helpers |
| `geometry_layout.js`  | Per-geometry 2D unwrap math (cylinder / box / sphere) |
| `h5_worker.js`        | Web worker: prefetches all four HDF5 files, decodes events |

## Requirements

- Python 3.8+ with stdlib only (server) or `h5py` + `numpy` (stub generator only).
- Modern browser with ES modules and WebGL.

## Quick start

```bash
# Against a real v3 dataset (production output or regenerated sample)
python3 serve_viewer.py /path/to/dataset_root --open

# Against the stub dataset in this repo
python3 serve_viewer.py ./sample_v3 --port 8765 --open
```

Then open **http://127.0.0.1:8765/**. The page shows `html loaded, waiting for viewer.js...` while modules parse; proceeds to mount/fetch status; then drops you into the viewer.

## UI

### Toolbar (left → right)

- **Event ← / → / input** — navigate events (commits on Enter or blur).
- **VIEW · `PMTs | SEG`** — exclusive 3D view: either the PMT event display or the truth-segment point cloud.
- **FIELD · `CHARGE | TIME | BETA | CHER FRAC*`** — drives continuous coloring of the active view (PE / edep for charge; earliest-hit T / segment time for time; per-segment β for beta; Cherenkov fraction f = PE_cher / (PE_cher + PE_scint) per sensor for cher frac). Colormap auto-switches (plasma for charge, viridis_r for time, viridis for β, diverging blue↔red for cher frac). `CHER FRAC*` is only shown on datasets with both Cherenkov and scintillation rows.
- **EMISSION · `All | Cherenkov | Scintillation`*** — only shown on datasets with both emission processes. Restricts the per-sensor PE/T to a single process: `All` shows the combined sensor signal (matches `sensor.h5`); `Cherenkov` and `Scintillation` derive each sensor's PE/T from `hits.h5` rows tagged with the chosen `emission_process`. The LABEL coloring + sidebar totals re-aggregate against the active slice — so `LABEL=Particle` under `EMISSION=Cherenkov` shows the dominant particle for each PMT's Cherenkov contribution. `CHER FRAC` is disabled when the filter is single-process (the ratio is trivially 1 or 0 in those slices).
- **LABEL** — categorical override.
  - `None` (default): use FIELD (continuous).
  - `Particle` — hashed hue per particle (PMTs via argmax over `inst.PE`; segs via `track → particle_idx`).
  - `Category` — fixed hue per LUCiD category (Primary / DecayElec / SecPion / Gamma / …).
  - `Ancestor` — hashed hue per primary track_id (root of each track's parent chain, from `labl/per_track/ancestor`).
  - `Interaction` — hashed hue per interaction rank (`labl/per_track/interaction` — 0-based index among the event's primaries).
- **TIME SWEEP** — fade PMTs and segments in as `simTime` crosses each primitive's arrival/emission time. Scrubber + play/pause shown at the bottom of the 3D panel. The fade also applies to the 2D panel.
- **RESET** — everything back to defaults.
- **ROTATE** — toggle auto-rotate of the 3D camera.
- **⚙ Settings** — see below.

### Particle / Group sidebar (left edge)

The sidebar contents depend on the current Label:
- `None` or `Particle`: lists every particle with its category, Σ PE, and swatch.
- `Category` / `Ancestor` / `Interaction`: lists distinct group IDs with their particle count and total PE.

Click a row to isolate that particle (or union of particles in the group): its PMT contributions pop at near-full brightness, non-contributors fade to ~8% alpha, and segments belonging to the selection stay bright. Click the same row again to deselect. Changing Label clears the selection.

Below the event-meta block, the sidebar shows a **SELECTION** info card:
- For a particle: category name, edep containment, Σ PE from inst, sensors hit, n tracks, n Cherenkov, max initial energy, PDG set, genealogy chain.
- For a group: kind, id, particle IDs in the group, Σ PE, sensors hit, n tracks.

### Settings drawer

- Continuous: log toggle, percentile slider (`perc_min`–`perc_max`), manual vmin/vmax, colormap select (auto / plasma / viridis / viridis_r / inferno_r).
- 3D: PMT disc size (default 10), show-empty-PMTs (gray silhouette), show detector outline.
- Time sweep: speed multiplier, quantile-transform T (default **on**) — replaces raw T with its rank fraction in [0, 1] for coloring AND sweep; crucial when arrivals cluster heavily.

## Data contract

Per [`docs/LUCID_DATASET.md`](../docs/LUCID_DATASET.md). The viewer reads:

**sensor.h5**
- `config/attrs`: `n_events`, `n_sensors`, `detector_type`, `dataset_name`, provenance.
- `config/sensor_positions` — (n_sensors, 3) float32, **in meters**.
- `event_NNN/attrs`: `source_event_idx`, `n_hits`.
- `event_NNN/{sensor_idx (u16), PE (f32), T (f32)}`.

**hits.h5**
- `config/…` same as sensor.
- `event_NNN/attrs`: `source_event_idx`, `n_particles`, `n_particle_hits`.
- `event_NNN/{particle_idx (i32), sensor_idx (u16), PE (f32), T (f32)}`.

**step.h5**
- `config/attrs`: `detector_shape`, `detector_radius`, `detector_half_height`, plus provenance.
- `event_NNN/attrs`: `source_event_idx`, `n_tracks`, `n_segments`.
- `event_NNN/{track_idx, start_{x,y,z}, end_{x,y,z}, dir_{x,y,z} (f16), time, edep, beta_start, n_cherenkov, contained (bool)}`.

**labl.h5**
- `config/attrs/label_names` e.g. `['category']`.
- `event_NNN/attrs`: `source_event_idx`, `n_particles`, `n_tracks`.
- `event_NNN/per_event/{t0 (f32 scalar), contained (bool scalar)}`.
- `event_NNN/per_interaction/{..., contained (bool)}` — True iff every meaningful segment of every particle in this interaction is contained; False for empty interactions.
- `event_NNN/per_particle/{category (u8), contained (bool), genealogy_data (i32), genealogy_offsets (u32), ext_genealogy_data, ext_genealogy_offsets}`.
- `event_NNN/per_track/{track_id (i32), parent_id (i32), pdg (i16), initial_energy (f32), n_cherenkov (i32), particle_idx (i32), ancestor (i32), interaction (i32)}`.

Cross-file integrity: `source_event_idx` is compared across all four files on each event load; a mismatch is logged to the browser console (non-fatal).

## Notes on real-data quirks

These caught me out during development — documenting so they don't catch you:

- **Sentinel rows in `sensor.h5`**. The writer emits a row per sensor-of-interest even when no real photon arrived, with `PE = 0` and `T = −t0` (i.e. shifted zero). The viewer filters these: a PMT is treated as "has signal" only if accumulated `PE > 0`. Sentinel sensors render as the gray silhouette.
- **`t0` can be negative**. The writer jitters each event's emission time within roughly ±15 ns so the detector reads events at random clock phases (matches real-data triggering). Sensor/step/inst times are stored as `t − t0` (detector frame, t=0 = trigger moment); the raw truth `t0` lives in `labl/per_event/t0`.
- **Units**. `sensor_positions` and all `step/*` coordinates are in **meters**. Times are in **ns**. Energies in MeV.
- **Ancestor/interaction are per-track** and consistent across all tracks of a given particle. The viewer derives per-particle ancestor/interaction by picking any one of the particle's tracks.

## Implementation notes

- **Worker** prefetches each HDF5 file whole (≤ few MB) via `fetch` at init, in parallel. Subsequent HDF5 reads are pure in-memory — no per-chunk HTTP round-trips.
- **Color modes**. Shader uniform `colorMode` flips between sampling a cmap texture (continuous) and computing HSL from a hashed/category-mapped hue (categorical). CPU writes one of `contVal` / `catVal` per mode switch.
- **Correspondence** is precomputed at event load: `particle → {sensor: PE}` map, `sensor → dominant_particle` array, plus per-particle ancestor/interaction. Selection resolves to a set of particle IDs; `hl = 0.35 + 0.65·sqrt(pe/max)` for any contributor guarantees visibility.
- **Segments as a point cloud**. WebGL draws `LineSegments` at 1 px regardless of materials, which gets lost against 10k PMT sprites. Instead each segment is expanded into 6 evenly-spaced points rendered with the PMT shader — always visible, same color/sweep/correspondence machinery.
- **2D unwrapping** per geometry (see `geometry_layout.js`):
  - **Cylinder** — barrel (θ × z) strip in the middle, top/bottom caps above and below (T-shape). Matches `good_notebooks/cylinder_2D_displays.ipynb`.
  - **Box** — four side faces unrolled as one strip (back → right → front → left), caps above/below. Face boundaries drawn as dashed seams.
  - **Sphere** — equirectangular Plate carrée. 2:1 aspect.
- **Time sweep** uses per-vertex `arrivalT` + `simTime` uniform; shader `smoothstep` fades each primitive as simTime crosses arrivalT. Quantile transform replaces `arrivalT` (and continuous `contVal` in TIME mode) with rank fraction. The 2D panel re-renders each frame while the sweep plays.

## Known deferrals

- No GPU picking on 3D PMTs/segments (hover → tooltip); selection is sidebar-click only.
- Multi-batch navigation — the manifest exposes only batch `0000` even when more exist.
- No light theme.
- No 2D pan/zoom/expand; panels are fixed-fit.
- Categorical Ancestor/Interaction uses golden-ratio hashed hues; can turn out similar for distinct small ids. For typical 2-primary events (ancestor ∈ {1,2}, interaction ∈ {0,1}) the colors are unambiguous.

## Development

Stub dataset generator for smoke tests before real production output is available:

```bash
python3 make_test_data.py --out /tmp/stub --geom cylinder --events 5
python3 serve_viewer.py /tmp/stub --port 8766 --open
```

The stub honors the v3 schema but has synthetic sensor hits/segments; won't match any real physics. For debugging the viewer shell it's sufficient.

Diagnostic logging is sprinkled through module init and worker startup; open DevTools → Console and look for lines prefixed `[viewer]` / `[h5_worker]`. On a JS parse error the page stalls on `html loaded, waiting for viewer.js...` — check the Console for the exception (that's how I caught the pdgName object-literal bug).
