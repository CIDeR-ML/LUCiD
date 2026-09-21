# Comparison Tools

This directory contains the FSMP, LUCiD, and JUNOsw comparison tools. It
keeps the time-response workflow separate from the detector-wide raw PE-scale
diagnostic because they use different numerical definitions.

`compare_sim.py`
: Shared readers, time-histogram utilities, PMT metadata, and legacy
  simulation comparison helpers. It is also a stand-alone legacy script.

`compare_sim_vs_real.py`
: Main response comparison. It reads the three sources, builds caches, and
  produces per-channel time, summary, and amplitude PDFs.

`laser_compare_channels.py`
: Selects representative channels from PMT geometry and type using a fixed
  seed.

`laser_compare_metrics.py`
: Builds time-response metrics, alignment, TTS/IRF response, dark-noise
  treatment, and amplitude diagnostics.

`laser_compare_plotting.py`
: Writes response, summary, and amplitude diagnostics as vector PDFs.

`laser_compare_csv.py`
: Writes selected-channel, FSMP-read audit, and per-channel response CSV
  files.

`pe_scale_io.py`
: Validates and loads raw all-channel caches; builds absent raw caches; matches
  PMT theta; and prepares PE-scale arrays.

`plot_pe_scale.py`
: Stand-alone raw PE-scale plotter. With `--build-missing`, it reuses valid
  caches or builds only missing source caches, then writes one theta-binned PDF
  per simulation source.

The time-response workflow is normally submitted with
`run/run_compare_multi_sim_vs_real_tts_dark_on.sh`. The independent PE-scale
workflow is submitted with `run/run_plot_pe_scale_raw_cache.sh`; it does not
produce response, summary, or amplitude PDFs.
