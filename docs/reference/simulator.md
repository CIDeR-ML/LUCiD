# Simulator: `setup_event_simulator`

```python
from lucid.simulation import setup_event_simulator

sim = setup_event_simulator(json_filename, n_photons=1_000_000, temperature=0.2, K=7, *,
                            source, transport=None, detection=None, hits=None,
                            hits_config=None, particle='muon', charge_resolution=None,
                            physics_config=None, default_detector_params=False,
                            wavelength_mode=True, wavelength_sampling='cherenkov',
                            reflection_model='scalar_mix', reflection_wavelength=400.0,
                            cherenkov_emission_band=None, emission_processes=None,
                            **grid_params)
```

It returns a `Simulator`: call it to run the forward model. It also records what it was built
from: `.det_geom` (the detector, medium and propagator), `.reflection_model`,
`.default_detector_params` (None unless baked in) and, for `source='particle'`, `.emitter`, the
Cherenkov emitter (`.emitter.n_photons(E)` is its photon yield).

## Arguments

| argument | meaning |
|---|---|
| `json_filename` | the geometry config, `config/<name>_geom_config.json` |
| `n_photons` | photons (rays) per event |
| `temperature` | soft sensor assignment: kernel width as a fraction of the sensor radius. `None` is a hard step (no geometric gradient) |
| `K` | scattering iterations; light scattering more often is dropped |
| `source` | required: `'particle'`, `'light_source'` or `'photons'` (below) |
| `transport` | `'expected'` (mean weights, no shot noise) or `'sampled'` (Monte Carlo). Default: `'sampled'` for `source='photons'`, which accepts only that; `'expected'` otherwise |
| `detection` | `'expected'` or `'sampled'` for QE, transit-time spread and charge resolution at the sensor. Same default as `transport` |
| `hits` | what the sensors report (below). Default: `'photon_list'` for a particle, `'first_arrival'` otherwise |
| `hits_config` | options of the hits function, e.g. `{'window_ns': ...}` for `'waveform'`; an unknown key raises |
| `particle` | the emitter to load: `'muon'` or `'electron'` |
| `charge_resolution` | per-sensor charge smearing of sampled hits: `None` (raw counts), `'Abe_2013'` (SK fractional Gaussian) or `'Bellamy_94'` (per-photoelectron SPE spectrum) |
| `physics_config` | the physics config path; required when `default_detector_params=True` |
| `default_detector_params` | `False`: the call takes a `DetectorParams`. `True`: load it from `physics_config` and fix it. A `DetectorParams`: fix that one |
| `wavelength_mode` | `True`: each photon carries a wavelength and the optics are evaluated per photon. `False`: one set of scalar optics ([optics](../concepts/optics.md)). A scintillating medium needs `True` |
| `wavelength_sampling` | `'cherenkov'` (default) or `'cherenkov_qe'` ([optics](../concepts/optics.md#how-wavelengths-are-sampled)) |
| `reflection_model` | `'scalar_mix'` (default; scalar rates with a specular/diffuse split), `'scalar'` (scalar rates; walls diffuse, sensors specular) or `'angular'` (Schlick wall, Fresnel PMT cathode) |
| `reflection_wavelength` | wavelength (nm) at which the angular model evaluates the cathode's dispersion |
| `cherenkov_emission_band` | the band (nm) Cherenkov wavelengths are sampled over. Default: the emitter's `emission_band_nm` |
| `emission_processes` | a subset of `('cherenkov', 'scintillation')`; default: the medium's own |
| `**grid_params` | propagation-grid keywords (below) |

The detector class comes from the geometry config's `detector_type`.

## Sources and call signatures

| `source` | photons | call |
|---|---|---|
| `'particle'` | a track (`ParticleParams`) through its SIREN emitter | `sim(track, detector_params, key)` |
| `'light_source'` | a source object: `laser_source`, `isotropic_source`, `cascade_source`, `shotgun_source` (`lucid.sources`) | `sim(source, detector_params, key)` |
| `'photons'` | given photons, e.g. from PhotonSim | `sim(detector_params, key, photon_data)` |

With `default_detector_params` set, the `detector_params` argument is dropped:
`sim(track, key)`, `sim(source, key)`, `sim(key, photon_data)`.

For given photons, `lucid.sources.data_event.simulate_data_event(sim, photons, key, nbuf=...,
t0=...)` pads the photon arrays to the buffer `sim` was built for (`n_photons`), runs it, adds
the event start time `t0` (ns) and returns the observed `(charge, time)` per sensor as numpy
arrays. `examples/seed_reconstruct.py --geant4` shows the full path from a ROOT file.

## Hits

Each call returns a NamedTuple. `hits` and `detection` together select the function from
`lucid.simulation.hits.HITS`; an unsupported pair raises and names the supported ones.

| `hits` | `detection` | returns |
|---|---|---|
| `first_arrival` | expected | `FirstArrival(charge, time)` per sensor: expected charge, and the mean of the predicted first-arrival time |
| `first_arrival` | sampled | `FirstArrival(charge, time)`: sampled charge (smeared by `charge_resolution`) and the earliest detected photon's time, with the transit-time spread |
| `photon_list` | expected | `PhotonList(log_w, times, indices, charge)`: per-deposit log-weights, arrival times and sensor indices, and the per-sensor charge. The reconstruction's input |
| `segment_truth` | sampled | `SegmentTruth`: per-sensor charge and times plus per-deposit weights, times, sensors and track segments. Dataset production |
| `waveform` | expected, sampled | `Waveform(waveform, n_dropped, n_detected)`: time-binned waveforms per sensor |
| `photon_hits` | sampled | `PhotonHits(detected, sensor_id, hit_time, indirect)`: one record per photon |

Charge is in photoelectrons, times in ns. The time resolution comes only from the per-photon
transit-time spread, `DetectorParams.response.tts`.

The expected first-arrival time is the mean of the first-arrival distribution: each deposit
`i` carries Poisson(`w_i`) photoelectrons at time `t_i`, smeared by the transit-time spread
`σ`, so the expected count before `t` is `R(t) = Σ w_i Φ((t − t_i)/σ)` and
`P(first > t | at least one) = (e^(−R(t)) − e^(−μ)) / (1 − e^(−μ))` with `μ = Σ w_i`. A dim
sensor reports roughly the weighted mean arrival time; a bright one its earliest light.

## Propagation grid

Sensors are found through a spatial grid. `**grid_params` sets it:

- `max_candidates_per_ray` (default 4): sensor slots per grid cell. If the geometry puts more
  sensors in one cell, it is raised to that count with a warning.
- cell counts: `n_cap`, `n_angular`, `n_height` (cylinder); `n_divisions` (sphere);
  `n_x`, `n_y`, `n_z` (box). Without them the grid is sized from the sensor spacing and
  `max_candidates_per_ray`.

The paper's reconstruction uses `n_cap=80, n_angular=120, n_height=80,
max_candidates_per_ray=8` on `SK_like`; with 4 candidates, vertex and t0 accuracy suffer at low
photon counts. A warning starting `Sensor map:` at setup reports a sensor in no cell or low cell
coverage.

## Choosing `n_photons`, `K` and `temperature`

- `n_photons`: enough that photon statistics do not limit the quantity you study. The examples
  use 200k for a display; the published reconstruction uses 250k; calibration 1M. Increase it
  until your result stops moving.
- `K`: displays use 4 to 6; reconstruction and calibration use 8 or more. Strongly scattering
  media need more (`examples/hello_telescope.py` uses 15 in ice). `scripts/find_optimal_k.py`
  finds the smallest `K` that keeps a target fraction of the charge.
- `temperature`: `None` for data or truth (hard assignment). For a model you fit, a small value
  (0.1 to 0.2; the published reconstruction uses 0.1). Lower is more faithful, with noisier
  gradients.

Published working points are `MODEL_RECIPE` (reconstruction) in `lucid.fitting` and `CALIB` in
`analysis/paper/utils/calibration.py`.
