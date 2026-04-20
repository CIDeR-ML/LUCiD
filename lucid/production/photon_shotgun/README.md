# Photon-shotgun Production Pipeline

Runs LUCiD with hand-specified per-photon origins / directions / wavelengths
(no particle track, no SIREN) across many cases and saves the result to HDF5.

## Modes

| Output | Contents | Storage |
|---|---|---|
| `waveform` (accumulated) | dense `(n_cases, num_sensors, n_time_bins)` — sparsified to COO `(case, sensor, time_bin, charge)` on save | tiny; compresses ~1000× vs dense |
| `per_photon` | per-photon `(detected, sensor_id, hit_time)` arrays length `n_photons` | fixed per case |

Both modes use MC sampling (`temperature=None`) → binary detection. TTS and
SK-like gain smearing are applied per detected photon.

## Default settings (override via CLI / `setup_shotgun_simulator` kwargs)

- `K = 12` (max scattering iterations; covers tail for all detectors tested)
- `window_ns = 500`, `bin_width_ns = 1`, `tts_sigma_ns = 1.0`
- Cherenkov-spectrum wavelengths sampled per photon

## Quick start

```bash
# 10k random positions × 100k photons in SK, waveform mode, streamed to disk
python -m lucid.production.photon_shotgun.run \
    --detector config/SK_geom_config.json \
    --physics-config config/SK_physics_config.json \
    --n-cases 10000 --n-photons 100000 \
    --position-mode uniform --direction-mode isotropic \
    --output-mode waveform \
    --chunk 20 \
    -o runs/shotgun_SK_10k.h5
```

## Load results

```python
from lucid.production.photon_shotgun.io import load_shotgun_waveform
out = load_shotgun_waveform('runs/shotgun_SK_10k.h5')
# out['case_idx'], out['sensor_id'], out['time_bin'], out['charge']
# out['n_detected'], out['n_dropped']
# out['meta'] = {'n_cases', 'num_sensors', 'n_time_bins', 'window_ns', ...}
```

Dense reconstruction via `load_shotgun_waveform(path, dense=True)` → adds
`out['waveform']`, shape `(n_cases, num_sensors, n_time_bins)`. Careful for
large runs: SK 10k cases dense ≈ 223 GB.

## Streaming writes

For very large runs (> ~1k cases × 100k photons) the driver uses
`StreamingWaveformWriter` / `StreamingPerPhotonWriter` which sparsify and
append each chunk to HDF5 without holding the dense tensor in RAM.

## Files

- `run.py` — CLI driver
- `io.py` — save/load + streaming writers + COO helpers
- `utils.py` — position/direction samplers, source construction
- `__init__.py` — re-exports
