# Digitizer and trigger

Dataset production (`lucid-run-job`) turns the simulated photon deposits into what a detector's
electronics record. Two NumPy stages, with no gradients, run after the simulator:
`lucid.simulation.digitizer` (charge integration, single-photoelectron charge, time jitter, dark
noise) and `lucid.simulation.trigger` (a sliding-window multiplicity trigger). Reconstruction
and calibration do not use them.

## Where the settings come from

Two top-level blocks, `"digitizer"` and `"trigger"`. Each is taken from the dataset config when
it has the key (`null` included), otherwise from the detector's physics config. A block in
neither stops the job before PhotonSim runs.

Only `config/SK_WAND_physics_config.json` and `config/HK_WAND_physics_config.json` carry the
blocks, so a `GeV` dataset config (which states none) runs only with `--detector SK_WAND` or
`HK_WAND`. To use another detector, state the readout in the dataset config:
`"digitizer": "basic"` (or `"ski"`, `"hk"`) and `"trigger": null` (or a block).
`"digitizer": null` is refused. The `Solar`, `SN` and `Test` configs state their own.

The shipped blocks:

| detector | `digitizer` | `trigger` |
|---|---|---|
| `SK_WAND` | `{"model": "ski"}` | `window_ns` 200, `n_thr` 25, `pad_before_ns` 300, `pad_after_ns` 300 |
| `HK_WAND` | `{"model": "hk"}` | `window_ns` 200, `n_thr` 45, `pad_before_ns` 300, `pad_after_ns` 300 |

## Digitizer

`"digitizer"` is a model name, or `{"model": name, ...}` where any other key overrides that
preset value (for example `"dark_rate_khz"` or `"threshold_pe"`).

| model | window | threshold | charge | time | TDC step | dark rate |
|---|---|---|---|---|---|---|
| `basic` | infinite | 0 pe | SK-like fractional smearing | none | none | 0 |
| `ski` | 200 ns | 0.25 pe | sampled SK single-photoelectron spectrum | SK Gaussian jitter | 0.4 ns | 4.2 kHz |
| `hk` | 200 ns | 0.25 pe | sampled HK single-photoelectron spectrum | HK exponentially modified Gaussian | 0.4 ns | 4.2 kHz |

Dead time is 0 in all three. `ski` and `hk` follow WCSim's SKI digitizer and its SK and HK PMT
models; `basic` is an idealised pass-through with one digit per hit PMT.

Per sensor: sort the hits by time, open a window at the first, sum the charge in it, record a
digit if the sum clears the threshold, then repeat from the next hit after the window (and dead
time). A digit's time is its first hit's. A sensor can therefore record several digits per
event.

Dark noise: a Poisson number of 1 pe hits per sensor at `dark_rate_khz`, uniform over the span
of the event's real hits, added before windowing. The span is padded on each side by
`max(pad_before_ns, pad_after_ns) + window_ns` of the trigger (500 ns for the WAND blocks), so
every possible gate is filled with dark noise, or by 100 ns without a trigger. Dark hits are labelled
`emission_process = 2` with `particle_idx = -1` in `hits/`. An event with no detected photons
gets no dark hits.

Deposits later than 100 μs after the event start (late neutron-capture or de-excitation light)
are dropped before digitizing.

## Trigger

`"trigger"`: `window_ns`, `n_thr`, `pad_before_ns`, `pad_after_ns` (or `pad_ns` for both).
Omitted keys take the `TriggerConfig` defaults: 200 ns, 30, 30 ns, 30 ns. `null` turns the
trigger off.

The trigger counts digits in a trailing window over the whole detector. A gate opens where the
count reaches `n_thr` (moved earlier by `pad_before_ns`) and closes where it drops below
(moved later by `pad_after_ns`); overlapping gates merge. Only digits inside a gate are kept.
An event with no gate is dropped from the dataset entirely, leaving a gap in
`source_event_idx`. Gates are written to `labl/.../per_window`
([schema](dataset-schema.md)).

Digit times are rounded to the 0.4 ns TDC step, and 200 ns is a multiple of it, so an event
exactly at `n_thr` can flip on a one-tick change in one arrival.

## Selection

The dataset config's `"selection"` block decides which interactions are written:

- `{"mode": "trigger"}`: those the trigger keeps (all `GeV` configs and `Test`).
- `{"mode": "min_physics_hits", "n": N}`: those with at least `N` real (non-dark) hits, a
  cheap truth-level cut in place of the trigger (`Solar`, `SN`). It needs `"trigger": null`:
  a trigger block together with it is refused. A supernova config must use it.

## Choosing

Use `basic` with no trigger when the electronics are not the point, or to compare with
one-hit-per-PMT datasets. Use `ski` or `hk` for realistic charge resolution, time jitter, TDC
granularity and dark noise. Add a trigger when acceptance and dark-noise triggers matter; to
only drop near-empty interactions, `min_physics_hits` is cheaper.
