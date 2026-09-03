## [2026-09-03] JUNO 532 nm laser simulation discrepancy diagnosis

### Environment
- Machine: remote Linux server
- Environment: `/opt/gentoo/opt/PyMC/bin/python`; NumPy-based read-only diagnostics
- Git commit: not recorded because repository status command did not return

### Input
- Three-way comparison summary PDF and channel PDF under
  `/junofs/C/liukainan/LUCiD/outputs/compare_sim_center_zpm15m/15040/`
- `aligned_data_1000_tts_on_dark_on_fsmpfix.npz`
- `comparison_stats_1000_tts_on_dark_on_fsmpfix.csv`
- JUNO laser simulator, nested-sphere forward model, material and geometry JSON

### Operation
- Read `README.md`, `proposal.md`, `diffsim.tex`, `AGENTS.md`, simulator,
  forward transport, comparison and run scripts.
- Checked normalized time-shape arrays and per-channel comparison fields.
- Reviewed monitor alignment, TTS/IRF convolution and dark subtraction semantics.
- No Slurm job, full simulation, code modification or raw-data write was run.

### Result
- LUCiD is high near 80--100 ns and low after roughly 130 ns relative to the
  dark-corrected FSMP shape.
- Mean normalized density ratio LUCiD/FSMP for Hamamatsu: 1.375, 0.654,
  0.582, 0.591, 0.357, 0.294 in 80--100, 100--130, 130--200,
  200--300, 300--400 and 400--500 ns; NNVT: 1.357, 1.018, 0.709,
  0.613, 0.339, 0.222 (diagnostic ratios, not likelihood estimates).
- The output is normalized by channel for shape comparison; absolute PE totals
  have different exposure/intensity scales and are not directly comparable.

### Notes
- Leading hypotheses are missing delayed optical paths (PMT/glass internal
  reflection, acrylic and engineering structures, non-scalar angular response),
  finite transport depth `K`, and possible comparison time-response mismatch.
- Next step: add deterministic simulator component toggles and path diagnostics,
  then compare prompt/late integrals and held-out channel/source groups before
  adding free parameters.
