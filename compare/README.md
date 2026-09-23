# Comparison Tools

This directory contains the FSMP, LUCiD, and JUNOsw comparison tools. It
keeps the time-response workflow separate from the detector-wide raw PE-scale
diagnostic because they use different numerical definitions.

`compare_sim.py`
: Shared readers, time-histogram utilities, PMT metadata, and legacy
  simulation comparison helpers. It is also a stand-alone legacy
  script. Histogram readers accept first_pe=True to keep only the
  earliest PE of every (event, PMT) record.

`compare_sim_vs_real.py`
: Main response comparison. It reads the three sources, builds caches, and
  produces per-channel time, summary, and amplitude PDFs with automatic
  source-intensity scaling. All three sources load only the first
  (earliest) PE of every waveform record.

`laser_compare_metrics.py`
: Builds time-response metrics, alignment, TTS/IRF response, dark-noise
  addition to simulations, and amplitude diagnostics with per-source
  intensity scales.

`laser_compare_plotting.py`
: Writes response, summary, and amplitude diagnostics as vector PDFs,
  annotating the c=1/mu normalization coefficient on the spectrum pages.

`laser_compare_csv.py`
: Writes selected-channel, FSMP-read audit, and per-channel response CSV
  files, including the dark-noise estimate and applied source scales.

`pe_scale_io.py`
: Validates and loads raw all-channel caches; builds absent raw caches;
  matches PMT theta; and prepares PE-scale arrays.

`plot_pe_scale.py`
: Stand-alone raw PE-scale plotter. With `--build-missing`, it reuses valid
  caches or builds only missing source caches, then writes one theta-binned
  PDF per simulation source.

## Usage

Both entry points are plain command-line tools runnable from the project
root. See `python compare/compare_sim_vs_real.py --help` for the
time-response comparison and `python compare/plot_pe_scale.py --help` for
the raw PE-scale diagnostic.
