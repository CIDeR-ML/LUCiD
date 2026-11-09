# Event viewer

A browser viewer for LUCiD datasets: the PMT readout (`sensor`), its decomposition by particle
(`hits`), the Geant4 truth segments (`step`) and the truth labels (`labl`), for cylinder, box and
sphere detectors. It shows a 3D view (three.js) and an unrolled 2D view, and links each PMT to
the particles that lit it. The files it reads are described in the
[dataset schema](../docs/reference/dataset-schema.md).

## Running it

The server needs only the Python standard library; `make_test_data.py` needs `h5py` and
`numpy`. A browser with ES modules and WebGL is required.

```bash
python3 serve_viewer.py /path/to/dataset_root --open             # a real dataset
python3 make_test_data.py --out ./test_data                      # or a synthetic one
python3 serve_viewer.py ./test_data --port 8765 --open
```

Then open http://127.0.0.1:8765/. `serve_viewer.py` binds `127.0.0.1` on port 8765 by default
(`--host`, `--port`); `--dataset` picks one dataset when the directory holds several. It reads
the files on every request, so reloading the page shows a rewritten dataset. Only the first
batch of a dataset is served.

`make_test_data.py` writes a dataset in the right format with synthetic hits and segments
(`--geom cylinder|box|sphere`, `--events`, `--sensors`, `--seed`); it is for checking the viewer,
not the physics.

### A dataset on a cluster node

Run the server on a node that sees the data and forward its port from your laptop. On S3DF
(use the node name that `hostname -f` prints):

```bash
python3 LUCiD/viewer/serve_viewer.py <dataset_dir> --port 8765       # on the node
ssh -J <user>@s3df.slac.stanford.edu -L 8765:127.0.0.1:8765 <user>@<node>.sdf.slac.stanford.edu
```

then open http://127.0.0.1:8765/ on the laptop. Two viewers need two ports.

## Controls

Toolbar, left to right:

- EVENT: previous, next, or type an index.
- VIEW: `PMTs` (the event display) or `EDEP` (the truth segments as a point cloud).
- FIELD: what colours the active view: `CHARGE` (PE, or energy deposit for segments), `TIME`
  (first-hit time, or segment time), `BETA` (segment β), and `CHER FRAC`, the Cherenkov
  fraction PE_cher / (PE_cher + PE_scint) per sensor, shown only when the dataset has both
  processes.
- EMISSION, shown when the dataset has more than one emission process: `All` is the sensor
  signal of `sensor.h5`; `Cherenkov`, `Scintillation` and `Dark` rebuild each sensor's PE and
  time from the `hits.h5` rows of that `emission_process` (0, 1, 2). Dark rows have
  `particle_idx = -1` and appear as a separate dark category under LABEL `Particle`.
- TRIGGER WINDOW, on triggered datasets: restricts the digits to one readout window.
- HIT, shown when some PMT has more than one digit: `All` sums each PMT's digits at its first
  time; `1st`, `2nd`, ... show only PMTs with that many digits, coloured by that digit (for
  example a muon's prompt digit and its decay electron's delayed one).
- LABEL: colour by category instead of FIELD: `Particle` (the particle with most PE on each
  PMT), `PDG` (fixed colours per particle type), `Interaction`, or `Segment` (needs
  `step/sensor_hits`).
- TIME SWEEP: fades PMTs and segments in as the time passes their arrival, with a scrubber under
  the 3D view; it applies to the 2D view too.
- RESET, ROTATE (auto-rotate), and the settings drawer (⚙): log scale, percentile clip, manual
  limits, colour map, PMT disc size, empty PMTs, detector outline, sweep speed, and the quantile
  transform of times (on for the PMT display by default), which helps when arrivals cluster.

The sidebar lists the particles (or the groups of the current LABEL) with their total PE.
Clicking a row isolates it: its PMTs and segments stay bright and the rest fade. The SELECTION
card shows the selected particle's category, PE, sensors hit, tracks, energy, PDG codes and
genealogy.

## Notes on the data

- Positions are in metres, times in ns, energies in MeV.
- Event times include a random per-interaction offset `t0` (the event's `t0` is in
  `labl/.../per_event/t0`; supernova events carry their true times). The viewer shows and colours
  `T − t0`, subtracted in float64.
- A PMT counts as hit only if its summed PE is positive.
- `source_event_idx` is compared across the four files on each event; a mismatch is logged to
  the browser console.
- The 2D layouts (`geometry_layout.js`): a cylinder is the barrel with the caps above and below,
  as in the paper's `fig_cylinder_2d_displays`; a box is the four side faces in a strip with the
  caps above and below; a sphere is an equirectangular map.

## Limitations

- Selection is by sidebar click only; there is no picking or hover in the 3D view.
- Only the first batch of a dataset can be viewed.
- The 2D view has no pan or zoom.
- `Particle`, `Interaction` and `Segment` colours are hashed hues, so two small ids can look
  alike.

## Debugging

The page shows `html loaded, waiting for viewer.js...` until its modules load. If it stays
there, a script failed to parse: open the browser console. Log lines from the viewer start with
`[viewer]` and `[h5_worker]`.
