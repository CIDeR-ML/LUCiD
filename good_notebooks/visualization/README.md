# visualization/ — event & detector displays (shared)

Display utilities, agnostic to reconstruction vs calibration. Forward = `setup_event_simulator`.

- **cylinder_2D_displays** — 2D unrolled-cylinder hit displays (prediction vs data).
- **geometry_and_events_3D_visualization** — 3D Plotly disc displays (multi-detector via `detector.visualize_event_data_plotly_discs`).
- **event_hit_animation** — animated GIF of PMT hits arriving over time.
