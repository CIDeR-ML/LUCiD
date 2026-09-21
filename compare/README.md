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
: Selects the channel panel: a fixed 72-PMT geometry/type representative
  set, a seeded random subset, all channels, or a seeded theta/type
  stratified panel (`--stratum-size` PMTs per 10-degree theta bin per PMT
  type).

`laser_compare_metrics.py`
: Builds time-response metrics, alignment, TTS/IRF response, dark-noise
  treatment, and amplitude diagnostics.

`laser_compare_plotting.py`
: Writes response, summary, and amplitude diagnostics as vector PDFs. For
  off-center sources the summary pages pool channels into populated
  10-degree theta bins.

`laser_compare_csv.py`
: Writes selected-channel, FSMP-read audit, and per-channel response CSV
  files.

`pe_scale_io.py`
: Validates and loads raw all-channel caches; builds absent raw caches;
  matches PMT theta; and prepares PE-scale arrays.

`plot_pe_scale.py`
: Stand-alone raw PE-scale plotter. With `--build-missing`, it reuses valid
  caches or builds only missing source caches, then writes one theta-binned
  PDF per simulation source.

## Time-response comparison

Run the comparison from the project root:

```bash
python compare/compare_sim_vs_real.py \
    -z 0 \
    --sim-dir /path/to/lucid/simulation \
    --junosw-sim-dir /path/to/junosw/simulation \
    --fsmp-dir /path/to/fsmp/pe \
    --output-root /path/to/output
```

`-z` selects the laser run through `--run-table`. FSMP `PEt` lists are
converted with the monitor correction `t = PEt + offset - t_monitor`; every
PE in each stored record is kept. Simulation amplitudes are scaled to
PE/event/ns: the LUCiD source is multiplied by the common 532 nm QE
(`--no-lucid-qe-correction` keeps the raw QE=1 scale), and each simulation
accepts a manual `--source-intensity-scale` factor (default `1 1`).

Channel panel (`--channel-mode`):

- `representative` (default): the fixed 72-PMT geometry/type set;
- `stratified`: a seeded panel with `--stratum-size` PMTs per 10-degree
  theta bin per PMT type (`--seed` fixes the draw);
- `random`: `-n` channels drawn with `--seed`;
- `all`: every channel.

Caches and outputs live under `--output-root/<run>/`; PDFs, CSVs, and the
aligned-data store carry `--output-tag` in their names, cache files do not.
Cache names carry only the selection parameters (channel mode, seed,
windows, file limits), so changing the tag or the LUCiD source never
renames the FSMP/JUNOsw caches. Each cache stores a content signature of
its input files (path, size, mtime): a mismatch rebuilds only the stale
source, while unchanged FSMP/JUNOsw caches are reused across LUCiD
iterations. `--force` rebuilds everything; `--plot-only` reuses caches and
aligned data and only redraws, with `--cache-root` pointing at a copied
cache tree when needed; `--brief` and the `--*-file-limit` options cap the
input volume for smoke tests.

Defaults worth knowing: per-PMT TTS convolution is on (`--no-tts`
disables), per-type IRF widths are set with `--irf-sigma-hamamatsu` and
`--irf-sigma-nnvt` (0 disables the IRF), FSMP dark noise is kept with its
sideband estimate added to the simulations (`--no-add-dark-to-simulation`
keeps it in FSMP only), alignment is a single global L1 shift
(`--alignment peak` for peak matching), and the analysis window is
`--data-range -100 500` (`--long` extends both data and plot ranges to
1000 ns).

## Raw PE-scale diagnostic

The PE scale `(N_FSMP - N_dark) / (q_s * N_MC)` uses separate raw
all-channel caches and never touches the response caches:

```bash
python compare/plot_pe_scale.py \
    --run 15040 \
    --fsmp-cache /path/to/fsmp_data_all_raw_pe_scale_all.npz \
    --simulation-cache /path/to/sim_raw_cache.npz \
    --pmt-map /path/to/junoid.parquet \
    --output-dir /path/to/output \
    --build-missing
```

`--build-missing` fills absent or stale caches from `--fsmp-dir`,
`--lucid-sim-dir`, and `--junosw-sim-dir`. By default the FSMP numerator is
dark-subtracted with the pre-peak sideband (`--dark-margin-ns`;
`--no-fsmp-dark-subtract` keeps raw counts) and channels below 0.01
PE/event are dropped (`--no-fsmp-min-pe-filter` disables the filter).
`--pe-per-event` switches both sides to per-event averages. One
theta-binned PDF is written per simulation source.