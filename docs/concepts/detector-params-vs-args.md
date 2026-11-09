# Parameters: DetectorParams vs arguments

## The litmus test
- **Would you ever take a gradient w.r.t. it / fit it?** → it's a **fittable pytree leaf** (DetectorParams, or
  ParticleParams for reconstruction).
- **Is it a configuration / runtime choice fixed for a given setup, or a known physical constant we never calibrate?**
  → it's an **arg** (passed to `setup_event_simulator` / the forward), not a pytree leaf.

Corollary: a leaf only earns a place in a fit pytree if some run might fit it (name it in `FieldParams(dp, fields)`). Everything that is
always-fixed (geometry, the medium *reference* shapes, the soft-overlap correction, the medium light
speed `c/n`) is an arg.

## Two fittable pytrees (the optimization state)
| pytree | holds | free in |
|---|---|---|
| **DetectorParams** | the detector's physical/response state (below) | calibration, joint |
| **ParticleParams** | track (energy, position, direction, t0) | reconstruction, joint |

`free = DetectorParams` → calibration; `free = ParticleParams` → recon; `free = both` → joint self-calibration.
Calibration sources (laser direction, flasher position, intensity) are not fitted: there is no
source-nuisance pytree. A fit with uncertain sources would add one beside `DetectorParams`.

## DetectorParams (nested by physics; leaves = exactly the fittable detector DOF)
```
DetectorParams(
  scattering    = ScatteringParams(scatter_length, mie_scatter_length, g,
                                   rayleigh_dev[n_ctrl], mie_dev[n_ctrl]),
  absorption    = AbsorptionParams(absorption_length, abs_dev[n_ctrl]),
  reflection    = ReflectionParams(wall_reflection_rate, sensor_reflection_rate,
                                   wall_R0, wall_p, wall_fspec,
                                   cathode_nr, cathode_nk, sensor_fspec),
  response      = ResponseParams(qe, tts, qe_dev[n_ctrl]),
  per_pmt       = PerPmtParams(qe_corrections[NS], t0[NS]),
  scintillation = ScintillationParams(S, kB, C, tau_rise, tau_fall),
)
```
The `*_dev` leaves are the fittable λ-deviation curves — one value per control wavelength
(see [wavelength physics](wavelength.md)); `dev ≡ 1` reproduces the pure medium
reference.

Notes / decisions per field:
- **tts** is a DetectorParams field (`response`): a calibrated PMT property. The data
  generation and the timing observable both read `dp.response.tts`. The charge spread is not a
  parameter: it is the simulator's `charge_resolution` argument and production's digitizer.
- **g, cathode_nk** are physical and fittable-in-principle but usually FROZEN (hard to measure) — present as fields,
  default just not among the fitted fields (SK fixes n_imag).
- **reflection holds the SUPERSET** of model params; the chosen reflection model (an arg,
  `reflection_model='scalar_mix'` (default)`|'scalar'|'angular'`) reads the subset it needs (scalar → the rate constants;
  Schlick/Fresnel angular → `wall_R0`+`wall_p`, `cathode_nr`+`cathode_nk`; `*_fspec` for
  specular/diffuse direction). Unused fields stay frozen. One DetectorParams structure, the
  model selects — no per-model pytree variant.
- **scintillation** fields default to 0 (no light), so a non-scintillating medium's forward is
  unchanged by them; they are read only when the medium scintillates.
- **Bounds and normalization** walk the whole pytree (`default_bounds`, `normalize_params` /
  `denormalize_params`), so any subset of leaves can be optimized directly.
- **per_pmt** are all `(NS,)`, gauge-fixed. The calibration fit (`lucid.fitting.calibrate` on a `CalibrationProblem`) profiles the gains in closed form each step; the joint charge-and-time fit (`fit_charge_time`) and the CRB carry the group as a Schur block. `qe_corrections`=rate
  efficiency (the per-PMT gain the calibration fits), `t0`=first-arrival offset (not yet applied
  by the hits).

## Args (config / runtime — NOT in any fit pytree)
| arg | why it's an arg, not a leaf |
|---|---|
| **geometry** (sensor positions, detector shape) | fixed per detector; via the geometry registry |
| **medium** (reference λ-shapes: Rayleigh 1/λ⁴, pure-water absorption, bench QE) | the FIXED reference the DetectorParams deviations multiply; never fit |
| **control_λ** (the deviation-curve grid, SK 337/375/398/405/445) | parameterization metadata paired with the `*_dev` lengths; not a value to fit |
| **sources** (nominal laser/iso positions + spectra; or the SIREN net) | the run's emission setup; not fitted (no source-nuisance pytree exists) |
| **reflection_model** | the reflection *model* choice; its *params* are DetectorParams |
| **N (photons), K (bounces), temperature (soft overlap), rng/keys** | runtime/differentiability knobs |
| **`transport`, `detection`, `hits`** (`'expected'` / `'sampled'`; what the sensors report) | which forward and readout, not a parameter |
| **soft-overlap correction** | a differentiability correction constant, depends on geometry+temperature, not a detector property |
| **medium light speed** `c/n` (timing velocity) | the single non-dispersive index `n` sets one light speed per event; a known medium constant, not calibrated. NB this is the *bulk* index — distinct from the fittable Fresnel cathode indices `cathode_nr`/`cathode_nk` in the reflection block |
| **numerical guards** | numerical config, not physics |
| **medium material name, wavelength band [lo,hi]** | source/medium config |

> The forward/physics path contains **no environment-variable reads** — every knob above is
> either a pytree leaf or an explicit argument, and this is enforced by a repo test
> (`test_no_env_reads_in_lucid_package`).
