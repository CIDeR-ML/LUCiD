# Digitizer & trigger

The [photon pipeline](../concepts/photon-pipeline.md) is differentiable end to end, but that
pipeline stops at `sensor_response`: a per-PMT charge and a soft-min first-arrival time. Real
front-end electronics don't record that directly — they integrate charge in time windows, apply
a threshold, and a separate DAQ decides whether to read the event out at all. `lucid/simulation/digitizer.py`
and `lucid/simulation/trigger.py` model that stage. They are plain NumPy (no JAX, no gradients)
and only run in the production path ([`lucid-run-job`](cli.md#lucid-run-job-produce-a-dataset),
see also the [local production guide](../guides/production/local.md)), not in the reconstruction
or calibration forward model.

## The digitizer models

`digitizer.py` turns a flat per-photon list `(sensor_idx, time, charge)` into **digits** — recorded
hits, one row per integration window that clears threshold. A sensor can produce more than one
digit per event if light arrives in separated time clusters (delayed coincidence, pile-up, dark
noise). Every model shares the same windowing code path; `basic` is not special-cased, it is
simply the preset with an infinite integration window, so every sensor collapses to exactly one
digit — the historical one-hit-per-sensor behavior, reproduced for backward compatibility.

The three presets, from `MODEL_PRESETS` in `lucid/simulation/digitizer.py`:

| model | integration window | deadtime | threshold | charge model | time model | dark rate | WCSim provenance |
|-------|---------------------|----------|-----------|--------------|------------|-----------|-------------------|
| `basic` | infinite (`None`) | 0 ns | 0 pe | `legacy` (the historical `sk_like` fractional smear) | `none` | 0 kHz | none — idealized passthrough, kept for parity |
| `ski` | 200 ns | 0 ns | 0.25 pe | `spe` (sampled SK SPE spectrum) | `sk_gauss` | 4.2 kHz | `WCSimWCDigitizer.hh:97-98` (window/deadtime), `WCSimPMTObject.cc` `Getqpe` + HitTimeSmearing `:88-99` (SK charge/time), `:262` (dark rate) |
| `hk` | 200 ns | 0 ns | 0.25 pe | `spe` (sampled HK SPE spectrum) | `hk_emg` | 4.2 kHz | same digitizer as `ski` (WCSim ships one digitizer, SKI — there is no separate QBEE model); `WCSimPMTObject.cc` HitTimeSmearing `:2124-2138` (HK Box&Line time response, `R12860`), `:2295` (dark rate) |

Both `ski` and `hk` also apply a 0.4 ns TDC truncation after the time-jitter model (WCSim SKI
timing precision, `WCSimWCDigitizer.hh:99`), and sample charge from a fitted Gaussian+exponential
single-photoelectron spectrum (Bellamy et al. 1994, NIM A 339 468) rather than a Gaussian
resolution function. `ski`'s SPE fit and `hk`'s SPE fit differ (`_SPE_SK` vs `_SPE_HK` in the
module) — `hk`'s HitTimeSmearing is also a different functional form (exponentially-modified
Gaussian vs pure Gaussian for `ski`).

Algorithm, per sensor (`digitize_event`, mirroring WCSim's SKI integrator): sort hits by time,
open an integration window at the first hit, sum charge over `[t0, t0+window]`, emit a digit if
the integrated charge clears `threshold`, veto hits for the following `deadtime`, and repeat from
the next surviving hit. `window=None` makes every group "simple" (fits in one window), which is
what makes `basic` collapse to one digit per sensor.

## Config keys

A dataset config (the JSON passed to `lucid-run-job --config ...`) enables the digitizer and
trigger with two optional top-level blocks, `"digitizer"` and `"trigger"`. Both are read by
`_read_digitizer_cfg` / `_read_trigger_cfg` in `lucid/production/run_job.py`: the dataset config
owns the block if present; otherwise the loader falls back to the detector's
`*_physics_config.json` (see the [config reference](config.md) for that file's other keys —
backward compatibility with older configs that put digitizer/trigger settings there); if neither
has it, digitizer defaults to `basic` and the trigger is off.

`"digitizer"` may be a bare model name or a dict with a `"model"` key plus overrides — any extra
key (e.g. `dark_rate_khz`, `threshold_pe`) is merged on top of the preset
(`resolve_model_config` in `digitizer.py` does `dict(MODEL_PRESETS[name]).update(overrides)`).

`lucid/production/configs/GeV/01_mu.json` (single muon, HK detector) enables the `hk` digitizer
and a real readout trigger (excerpted — the file has the usual particle/energy/output keys too):

```json
{
  "detector": "HK",
  "digitizer": {
    "model": "hk"
  },
  "selection": {
    "mode": "trigger"
  },
  "trigger": {
    "window_ns": 200.0,
    "n_thr": 45,
    "pad_before_ns": 300.0,
    "pad_after_ns": 300.0
  }
}
```

`lucid/production/configs/GeV/11_mu_pi_plus_pi_minus.json` (three-particle event on the same
detector) uses the identical digitizer/trigger block — the block is copied per dataset config, not
inherited automatically, so multi-particle and single-particle configs for the same detector must
each declare it if they want the same electronics model.

The `"selection"` block is a separate, truth-level knob (`_read_min_physics_hits` in
`run_job.py`) that decides which interactions are written at all: `{"mode": "min_physics_hits",
"n": N}` keeps interactions with at least `N` real (non-dark) hits — a cheap stand-in trigger that
doesn't touch digits — while `{"mode": "trigger"}` defers entirely to the real readout trigger
below (in which case `_read_min_physics_hits` returns `None` and the digit-level gate is the only
filter).

## The trigger

`trigger.py` implements a sliding-window hit-sum trigger over the detector-wide digit times — the
SK-like N200 trigger, applied after digitization rather than to raw photon hits. `TriggerConfig`
holds four fields:

- `window_ns` — the coincidence window width (SK-like default 200 ns; distinct from the
  digitizer's own per-sensor integration window).
- `n_thr` — the detector-wide hit multiplicity that must be reached inside the window to fire
  (default 30).
- `pad_before_ns` / `pad_after_ns` — how far the recorded gate extends before the up-crossing and
  after the down-crossing (default 30 ns each; a config may also set a single symmetric `pad_ns`,
  which `TriggerConfig.from_block` expands to both).

`find_trigger_gates` computes a trailing-window multiplicity `m(t) = #hits in (t-window, t]`,
opens a gate when it crosses `n_thr` **up** (minus `pad_before_ns`), closes it on the following
down-crossing (plus `pad_after_ns`), and merges overlapping gates. `apply_trigger` keeps only
in-gate digits, canonically re-sorts them by `(window, sensor_idx, T)`, remaps the `digit_idx`
foreign keys in the hits/segment decompositions, and returns the per-window gate list alongside
the filtered digits — or `None` if nothing in the event crossed threshold.

Selection interacts with the trigger at the event level, not just the digit level: when a config
sets `"selection": {"mode": "trigger"}`, an event that never crosses `n_thr` produces no windows
at all, and `apply_trigger` returning `None` means the whole event is dropped from the dataset —
it doesn't appear as an empty entry, it is simply **absent**, leaving gaps in the batch's
`source_event_idx`. The recorded per-window gates themselves (`window_start`, `window_end`, and
the CSR `digit_offsets` into the sensor digit list) are written to `labl.h5`'s `per_window` group
— see the [dataset schema](dataset-schema.md) for the exact layout rather than duplicating it
here.

## Dark noise

Dark counts are generated directly in `digitizer.py` (`generate_dark_noise`): a Poisson number of
hits per sensor with mean `rate_khz · Δt · 1e-3` (rate in kHz, `Δt` in ns), spread uniformly across
the event's readout span (the true/reco time range of the real hits, padded by
`readout_pad_ns`, default 100 ns), each worth `charge_pe=1.0` p.e. before the digitizer sees it.
These synthetic hits are concatenated onto the real photon list before windowing, so a dark hit
can extend a digit's integration window or is absorbed into one if it falls inside a real pulse.

Dark digits are labelled in the `hits.h5` decomposition, not hidden: `digitize_and_decompose`
tags every dark-noise row with `emission_process = EMISSION_PROCESS_DARK` (value `2`, alongside
`0` = Cherenkov and `1` = scintillation) and `particle_idx = -1` / `segment_idx = -1` (no owning
particle or track), so `sensor.h5` stays a clean `(sensor_idx, PE, T)` list while `hits.h5` still
records which contribution was noise.

Both real and dark deposits pass through a shared readout time cap: anything later than
`_MAX_DIGIT_TIME_NS = 1e5` ns (100 μs) relative to the event's reference time is dropped before
windowing. This exists for late nuclear-channel light (neutron capture, de-excitation), which can
legitimately arrive ms–s after the primary interaction — far outside anything a real detector
reads out — and keeps the digit list, the decomposition, and the dark-noise window bounded.

## Try it without PhotonSim

The digitizer and trigger are pure NumPy functions over `(sensor_idx, time, charge)` arrays, so
you can exercise them on a synthetic event without running PhotonSim or importing JAX:

```python
import numpy as np
from lucid.simulation.digitizer import resolve_model_config, digitize_and_decompose
from lucid.simulation.trigger import TriggerConfig, apply_trigger

rng = np.random.default_rng(0)

# A synthetic "event": 40 photon hits on distinct PMTs (of 5000 in the detector),
# arriving in a fast ~15 ns burst around t=1000 ns, one photoelectron each, all
# from the same truth particle/segment.
n_sensors = 5000
n_hits = 40
sensor_idx = np.arange(n_hits)
t_true = 1000.0 + rng.uniform(0, 15, n_hits)
t_reco = t_true.copy()               # no TTS in this toy example
charge = np.ones(n_hits)
particle_idx = np.zeros(n_hits, dtype=np.int64)
segment_idx = np.zeros(n_hits, dtype=np.int64)
emission_process = np.zeros(n_hits, dtype=np.int64)   # 0 = Cherenkov

model = resolve_model_config("hk")   # HK preset: 200 ns window, 0.25 pe thr, 4.2 kHz dark

sensor_digits, hits_sparse, seg_hits = digitize_and_decompose(
    sensor_idx=sensor_idx, charge=charge, t_true=t_true, t_reco=t_reco,
    particle_idx=particle_idx, segment_idx=segment_idx, emission_process=emission_process,
    n_sensors=n_sensors, model=model, rng=rng,
    dark_rate_khz=model["dark_rate_khz"], readout_pad_ns=100.0)

n_dark = int((hits_sparse["emission_process"] == 2).sum())
print(f"digits: {sensor_digits['sensor_idx'].shape[0]} "
      f"(physics hits: {n_hits}, dark-tagged hit rows: {n_dark})")
print(f"mean digit charge: {sensor_digits['PE'].mean():.3f} pe "
      f"(sample sigma: {sensor_digits['PE'].std():.3f})")

trig_cfg = TriggerConfig()  # window_ns=200, n_thr=30, pad_before/after=30 ns
result = apply_trigger(sensor_digits, hits_sparse, seg_hits, trig_cfg)
if result is None:
    print("no trigger fired")
else:
    triggered_sd, triggered_hits, triggered_seg, per_window = result
    print(f"trigger fired: {per_window['window_start'].shape[0]} window(s), "
          f"kept {triggered_sd['sensor_idx'].shape[0]}/{sensor_digits['sensor_idx'].shape[0]} digits")
    print(f"window: [{per_window['window_start'][0]:.1f}, {per_window['window_end'][0]:.1f}] ns")
```

Output (seeded, deterministic):

```
digits: 44 (physics hits: 40, dark-tagged hit rows: 4)
mean digit charge: 0.987 pe (sample sigma: 0.436)
trigger fired: 1 window(s), kept 42/44 digits
window: [983.2, 1237.6] ns
```

44 digits from 40 physics hits because 4.2 kHz of dark noise over the ~215 ns readout span across
5000 sensors adds a handful of extra single-pe digits (here, 4); the trigger's default `n_thr=30`
clears easily on the 40-hit burst and produces one gate, dropping the 2 digits that fell outside
`window_ns + pad` of the burst.

## Choosing a model

Use `basic` for anything where the electronics response itself isn't the point — quick
turnaround datasets, ML training sets where you want the historical one-hit-per-sensor charge/time
pair, or comparisons against older LUCiD datasets. Switch to `ski` or `hk` when you need the SPE
charge resolution, PMT time jitter, and TDC granularity that a real SK- or HK-like readout
contributes, e.g. when validating against WCSim or characterizing low-charge/near-threshold
behavior. Only turn on the trigger block if you specifically care about DAQ acceptance and
dark-noise-driven false triggers; if you just want to drop low-activity interactions, the cheaper
`selection.mode = "min_physics_hits"` truth-level cut avoids running the digit-level trigger
machinery entirely.
