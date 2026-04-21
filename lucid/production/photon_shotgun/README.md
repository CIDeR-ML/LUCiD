# Photon-shotgun Production Pipeline

Runs LUCiD with hand-specified per-photon origins / directions / wavelengths
(no particle track, no SIREN) across many cases and streams the result to
HDF5. Designed for large position / source-parameter scans.

---

## Modes

| Output | Contents | Typical size |
|---|---|---|
| `waveform` (accumulated) | dense `(n_cases, num_sensors, n_time_bins)` histogram, sparsified to COO `(case, sensor, time_bin, charge)` on save | ~24 KB/case at SK (0.08 % bins nonzero) |
| `per_photon` | per-photon `(detected, sensor_id, hit_time)` arrays of length `n_photons` | ~900 KB/case uncompressed; ~17 KB/case gzipped |

Both modes use MC sampling (`temperature=None`, hard sensor kernel) →
binary detection. TTS Gaussian smearing + SK-like gain smearing are
applied per detected photon.

### Defaults

```
K             = 12     # scattering iterations (covers tail for all detectors tested)
temperature   = None   # hard sensor edges (required for binary detection)
window_ns     = 500
bin_width_ns  = 1.0    # 1 GHz FADC sampling
tts_sigma_ns  = 1.0    # roughly SK R3600 single-PE time resolution
wavelength    = "cherenkov"   # per-photon sampled from 1/λ² over the QE-curve
                              # range intersected with [300, 700] nm
                              # (e.g. SK → [300, 648], HK → [300, 651]; default
                              # [300, 700] when no QE curve is loaded)
direction     = isotropic     # one random direction per case, shared by all photons
position      = uniform       # random inside detector volume, shrunk by 0.9
```

Every default is overridable via CLI flags (see `--help`).

---

## Quick start

```bash
# 1k random positions × 100k photons in SK, waveform mode
python -m lucid.production.photon_shotgun.run \
    --detector config/SK_geom_config.json \
    --physics-config config/SK_physics_config.json \
    --n-cases 1000 --n-photons 100000 \
    --output-mode waveform \
    --chunk 20 \
    -o runs/shotgun_SK_1k_waveform.h5
```

End-to-end on an RTX 4090: ~90 s, ~11 MB on disk for the above.

```bash
# Same geometry, per-photon mode (saves raw hit list)
python -m lucid.production.photon_shotgun.run \
    --detector config/SK_geom_config.json \
    --physics-config config/SK_physics_config.json \
    --n-cases 1000 --n-photons 100000 \
    --output-mode per_photon \
    --save-source \
    -o runs/shotgun_SK_1k_per_photon.h5
```

`--save-source` stores per-case origin / direction arrays so downstream
notebooks can plot detection fraction vs position.

---

## CLI flags (grouped)

```
Geometry
  --detector              PATH      Detector geometry JSON (required)
  --physics-config        PATH      Physics config JSON (required)
  --detector-type         {Cylinder, Sphere, Box}

Scale
  --n-cases               INT       Number of source configurations
  --n-photons             INT       Photons per case (JIT cache key)
  --chunk                 INT       Cases per vmap batch (GPU memory knob)

Source
  --position-mode         {uniform, center}
  --position-fraction     FLOAT     Shrink factor for uniform sampling (default 0.9)
  --direction-mode        {isotropic, fixed}
  --origin                X Y Z     For --position-mode center
  --direction             DX DY DZ  For --direction-mode fixed
  --wavelength            FLOAT | "cherenkov"
  --wavelength-sampling   {cherenkov, cherenkov_qe}
                                    'cherenkov' (default) = λ~1/λ², per-photon
                                    QE weight. 'cherenkov_qe' = λ~QE(λ)/λ²,
                                    scalar <QE>_C weight — lower variance at
                                    fixed photon count but the output becomes
                                    a density estimate rather than a literal
                                    per-shot realization.
  --intensity             FLOAT

Simulator
  --K                     INT       Max scattering iterations (default 12)
  --output-mode           {waveform, per_photon}

Waveform / smearing
  --window-ns             FLOAT
  --bin-ns                FLOAT
  --tts-sigma-ns          FLOAT
  --no-smear-time
  --no-smear-charge

IO
  --seed                  INT       Drives positions, directions, Cherenkov sampling, and QE Bernoulli
  -o, --output            PATH
  --save-source                     Persist per-case origin / direction arrays
```

---

## Common recipes

**Uniform-position + isotropic-direction beam survey (default pattern)**

```bash
python -m lucid.production.photon_shotgun.run \
    --detector config/HK_geom_config.json --physics-config config/HK_physics_config.json \
    --n-cases 10000 --n-photons 500000 \
    --output-mode waveform --chunk 10 \
    -o runs/shotgun_HK_10k_500k.h5
```

**Fixed-beam study — all cases share an origin and a direction**

```bash
python -m lucid.production.photon_shotgun.run \
    --detector config/SK_geom_config.json --physics-config config/SK_physics_config.json \
    --n-cases 1000 --n-photons 1000000 \
    --position-mode center --origin 0 0 0 \
    --direction-mode fixed --direction 0 0 1 \
    --wavelength 400 \
    --output-mode waveform \
    -o runs/shotgun_SK_vertical_beam.h5
```

**IWCD / WCTE (small detector)**

```bash
python -m lucid.production.photon_shotgun.run \
    --detector config/IWCD_geom_config.json --physics-config config/IWCD_physics_config.json \
    --n-cases 1000 --n-photons 100000 \
    --K 10 \
    -o runs/shotgun_IWCD_1k.h5
```

**Single-λ scan** — repeat the command above with `--wavelength 350`, `400`,
`450`, … and pass distinct `-o` paths to compare wavelength dependence.

---

## Loading outputs

```python
from lucid.production.photon_shotgun.io import load_shotgun_waveform, load_shotgun_per_photon

# Waveform mode
out = load_shotgun_waveform('runs/shotgun_SK_1k_waveform.h5')
# out['case_idx'], out['sensor_id'], out['time_bin'], out['charge']
# out['n_detected'], out['n_dropped']
# out['meta'] = {'n_cases', 'num_sensors', 'n_time_bins', ...}

# Dense reconstruction (careful — SK 10k cases dense ≈ 223 GB)
out = load_shotgun_waveform(path, dense=True)
waveform = out['waveform']   # (n_cases, num_sensors, n_time_bins)

# Per-photon mode
out = load_shotgun_per_photon('runs/shotgun_SK_1k_per_photon.h5')
# out['detected'], out['sensor_id'], out['hit_time']  (each shape (n_cases, n_photons))
```

---

## Parallelization — sharding by seed

Running one `n-cases = 100000` job serialises all work. Better: shard into
`N` independent sub-jobs with distinct seeds and non-overlapping case
counts, then merge. Each shard is a standalone HDF5 file — runnable on
separate GPUs, hosts, or just sequentially if that's what you have.

**Launch in parallel (separate machines / GPUs)**

```bash
# On machine 0:
python -m lucid.production.photon_shotgun.run \
    --detector config/SK_geom_config.json --physics-config config/SK_physics_config.json \
    --n-cases 2500 --n-photons 100000 \
    --seed 1 \
    -o runs/shard_0.h5

# On machine 1:
python -m lucid.production.photon_shotgun.run ... --seed 2 -o runs/shard_1.h5
# ...on machine 2, 3:  seeds 3, 4
```

**Or launch N in background on one box (useful for CPU fallback, not for
a single GPU that they'd contend for)**

```bash
for i in 0 1 2 3; do
    python -m lucid.production.photon_shotgun.run \
        --detector config/SK_geom_config.json --physics-config config/SK_physics_config.json \
        --n-cases 2500 --n-photons 100000 \
        --seed $((1 + i)) \
        -o runs/shard_${i}.h5 &
done
wait
```

**Merge into one HDF5**

```python
from lucid.production.photon_shotgun.io import merge_waveform_shards

merge_waveform_shards(
    [f'runs/shard_{i}.h5' for i in range(4)],
    'runs/shotgun_SK_10k_merged.h5',
)
```

`merge_per_photon_shards(...)` exists for per-photon outputs. The merger
validates that shards agree on `num_sensors`, `n_time_bins`, `n_photons`,
`window_ns`, `bin_width_ns`; cases are renumbered globally (shard 0 →
[0, n0), shard 1 → [n0, n0+n1), …). Total `n_cases` in the merged file is
the sum of shard case counts.

**Different source configurations in parallel** — a single merged file
won't make sense if shards have different detectors or source setups. Keep
them as separate files and run analysis per file.

---

## Streaming writes

Large runs never hold the dense `(n_cases, num_sensors, n_time_bins)`
tensor in memory. `StreamingWaveformWriter` / `StreamingPerPhotonWriter`
sparsify and append each chunk to HDF5 as the simulator produces it.
`run.py` uses these automatically; `io.save_shotgun_*` and
`io.load_shotgun_*` are the small-data one-shot convenience APIs.

---

## Notebooks

Three starter notebooks live in `notebooks/`:

1. `01_waveform_single_event.ipynb` — pick a case, plot the detector-summed
   waveform, 3D sensor scatter coloured by charge (linear + log), and
   first-hit time. Requires a waveform HDF5.
2. `02_per_photon_hit_map.ipynb` — detection fraction / time distributions,
   3D per-sensor hit counts. Requires a per-photon HDF5.
3. `03_detection_rate_scan.ipynb` — detection fraction vs source position
   (ρ, z, 2D heatmap, wall distance, 3D scatter). Requires `--save-source`.

---

## Files

```
lucid/production/photon_shotgun/
├── __init__.py     # re-exports
├── run.py          # CLI entry point (python -m ...)
├── io.py           # COO sparsify/densify, save/load, streaming writers,
│                   # shard merger
├── utils.py        # position samplers (uniform-in-cylinder, isotropic)
├── viz.py          # 3D sensor scatter + lin/log histogram helpers
├── README.md       # this file
└── notebooks/      # 01, 02, 03 + _build_notebooks.py
```
