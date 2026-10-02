# LUCiD vs FSMP/JUNOsw Comparison Tools

Usage: `make help` (all figures are reproducible Make targets; caches
and PDFs land under the data directory).

- `compare_sim_vs_real.py`: main three-way comparison (time / summary /
  amplitude PDFs, cached stages).
- `plot_tts_vs_irf.py`: per-channel TTS vs measured-IRF cross-check.
- `plot_pe_scale.py`: detector-wide PE/event scale comparison.
- `plot_theta_intensity.py`: per-PMT relative PE counts vs theta.
- `plot_source_scale_vs_z.py`: scale-vs-z summary across source
  positions.
- `compare_sim.py`, `laser_compare_*.py`, `pe_scale_io.py`: internal
  modules.
