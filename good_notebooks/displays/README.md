# displays/ — event & detector displays

All views from the library — **no inline display code** (the old per-notebook display helpers now
live in `lucid.visualization`).

- **event_displays** ⭐ — one event, three ways: 2D unrolled cylinder
  (`visualization.create_detector_display`), time-evolution animation
  (`visualization.animate_event`), and interactive 3D discs
  (`detector.visualize_event_data_plotly_discs`). (GPU-validated.)

Seam: `lucid.visualization` = `create_detector_display` (2D) + `animate_event` (GIF) +
`unroll_layout` (the shared 2D barrel-unroll). The old `cylinder_2D_displays` /
`geometry_and_events_3D_visualization` / `event_hit_animation` (which re-defined displays inline)
are in `../archive/`.
