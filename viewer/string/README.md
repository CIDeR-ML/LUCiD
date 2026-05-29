# String Detector Event Viewer

3D viewer for IceCube-86 string detector events simulated by LUCiD.

## Quick start

```bash
# 1. Generate events (requires download_data.sh to have been run first)
python scripts/run_string_siren_tracks.py --particle muon
python scripts/run_string_siren_tracks.py --particle electron

# 2. Serve the viewer
bash viewer/string/serve.sh
```

Open http://localhost:8766/viewer/string/viewer.html in a browser.

## Datasets

The dropdown switches between:

| Dataset | Source script |
|---------|-------------|
| Muon tracks | `scripts/run_string_siren_tracks.py --particle muon` |
| Electron tracks | `scripts/run_string_siren_tracks.py --particle electron` |
| Cascades | `scripts/run_string_cascades.py` |

Output lands in `output/siren_tracks_{muon,electron}/` and `output/cascades_sim/`.

## Controls

- **Point size / Emphasis** — adjust DOM rendering
- **Play / Pause** — animate the Cherenkov wavefront
- **Speed** — animation speed
- **Rotation** — auto-rotate camera
- **LOG** — toggle log/linear charge scale
- **Unhit / Track** — show/hide unhit DOMs and track line
- **GIF / MP4** — export animation

## Events

Each event in `events.json` contains:

- `track_origin`, `track_direction` — particle vertex and direction
- `energy_gev` — particle energy
- `track_length_m` — emission extent (p95 of SIREN intensity along track)
- `hit_dom_ids`, `hit_charges`, `hit_times_ns` — per-DOM hit data
