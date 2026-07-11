# DetectorParams (fittable pytree) vs args (config/runtime) — the delineation

## The litmus test
- **Would you ever take a gradient w.r.t. it / fit it?** → it's a **fittable pytree leaf** (DetectorParams, or
  SourceParams / ParticleParams for the other two fit modes).
- **Is it a configuration / runtime choice fixed for a given setup, or a known physical constant we never calibrate?**
  → it's an **arg** (passed to `setup_event_simulator` / the forward), not a pytree leaf.

Corollary: a leaf only earns a place in a fit pytree if some run might list it in `trainable_fields`. Everything that is
always-fixed (geometry, the medium *reference* shapes, the soft-overlap correction, the medium light
speed `c/n`) is an arg.

## Three fittable pytrees (the optimization state) — `trainable_fields` selects across them
| pytree | holds | free in |
|---|---|---|
| **DetectorParams** | the detector's physical/response state (below) | calibration, joint |
| **SourceParams** | calibration source nuisances (laser dir, iso pos, per-source intensity) | calibration (optional), joint |
| **ParticleParams** | track (energy, position, direction, t0) | reconstruction, joint |

`free = DetectorParams` → calibration; `free = ParticleParams` → recon; `free = both` → joint self-calibration.
SourceParams is the calibration analogue of ParticleParams, used only when sources are uncertain.

## DetectorParams (nested by physics; leaves = exactly the fittable detector DOF)
```
DetectorParams(
  scattering    = ScatteringParams(scatter_length, mie_scatter_length, g,
                                   rayleigh_dev[n_ctrl], mie_dev[n_ctrl]),
  absorption    = AbsorptionParams(absorption_length, abs_dev[n_ctrl]),
  reflection    = ReflectionParams(wall_reflection_rate, sensor_reflection_rate,
                                   wall_R0, wall_p, wall_fspec,
                                   cathode_nr, cathode_nk, sensor_fspec),
  response      = ResponseParams(qe, spe_width, tts, qe_dev[n_ctrl]),
  per_pmt       = PerPmtParams(qe_corrections[NS], gain[NS], t0[NS], walk[NS]),
  scintillation = ScintillationParams(S, kB, C, tau_rise, tau_fall,
                                      moyal_amp, moyal_loc, moyal_scale),
)
```
The `*_dev` leaves are the fittable λ-deviation curves — one value per control wavelength
(see [wavelength physics](wavelength.md)); `dev ≡ 1` reproduces the pure medium
reference.

Notes / decisions per field:
- **tts, spe_width** are DetectorParams (`response`): calibrated PMT properties. The data
  generation and the timing observable both read `dp.response.tts`.
- **g, cathode_nk** are physical and fittable-in-principle but usually FROZEN (hard to measure) — present as fields,
  default just not in `trainable_fields` (SK fixes n_imag).
- **reflection holds the SUPERSET** of model params; the chosen reflection model (an arg,
  `reflection_model='scalar_mix'` (default)`|'scalar'|'angular'`) reads the subset it needs (scalar → the rate constants;
  Schlick/Fresnel angular → `wall_R0`+`wall_p`, `cathode_nr`+`cathode_nk`; `*_fspec` for
  specular/diffuse direction). Unused fields stay frozen. One DetectorParams structure, the
  model selects — no per-model pytree variant.
- **per_pmt** are all `(NS,)`; the fitter Schur-marginalizes this group structurally (gauge-fixed). `qe_corrections`=rate
  efficiency, `gain`=charge-per-PE, `t0`=offset, `walk`=TQ-map slope.

## Args (config / runtime — NOT in any fit pytree)
| arg | why it's an arg, not a leaf |
|---|---|
| **geometry** (sensor positions, detector shape) | fixed per detector; via the geometry registry |
| **medium** (reference λ-shapes: Rayleigh 1/λ⁴, pure-water absorption, bench QE) | the FIXED reference the DetectorParams deviations multiply; never fit |
| **control_λ** (the deviation-curve grid, SK 337/375/398/405/445) | parameterization metadata paired with the `*_dev` lengths; not a value to fit |
| **sources** (nominal laser/iso positions + spectra; or the SIREN net) | the run's emission setup; the fittable *nuisance* lives in SourceParams |
| **reflection_model** | the reflection *model* choice; its *params* are DetectorParams |
| **observables** (which of charge_mean / charge_var / first_arrival) | a fit-config list, not a physical value |
| **N (photons), K (bounces), temperature (soft overlap), rng/keys** | runtime/differentiability knobs |
| **response mode** (expected-value model vs sampled truth) | which response, not a parameter |
| **soft-overlap correction** | a differentiability correction constant, depends on geometry+temperature, not a detector property |
| **medium light speed** `c/n` (timing velocity) | the single non-dispersive index `n` sets one light speed per event; a known medium constant, not calibrated. NB this is the *bulk* index — distinct from the fittable Fresnel cathode indices `cathode_nr`/`cathode_nk` in the reflection block |
| **numerical guards** | numerical config, not physics |
| **medium material name, wavelength band [lo,hi]** | source/medium config |

## Borderline calls (resolved)
- **TTS / SPE width** → DetectorParams (calibrated). Not config.
- **Soft-overlap correction** → arg/config (a differentiability correction, not a physical detector parameter).
- **medium light speed `c/n` (timing velocity)** → arg (known). The medium currently uses one
  non-dispersive index `n`, so timing rides on a single light speed per event (see
  [wavelength physics](wavelength.md)); it is *not* a group velocity and there is no dispersion
  yet. Promote to a frozen DetectorParams field only if a future study fits it. The bulk `n` stays
  frozen — the only refractive indices a fit ever touches are the Fresnel cathode `cathode_nr`/
  `cathode_nk` (and those default frozen too).
- **g, cathode n_imag** → DetectorParams fields, default frozen.
- **control_λ** → arg/config (the basis grid), must stay consistent with the `*_dev` field lengths.
- **Source position/intensity** → SourceParams (fittable nuisance) when uncertain; the *nominal* setup is an arg.

> The forward/physics path contains **no environment-variable reads** — every knob above is
> either a pytree leaf or an explicit argument, and this is enforced by a repo test
> (`test_no_env_reads_in_lucid_package`).

## In short

- A leaf belongs in a fit pytree **iff some run might calibrate it**.
- `DetectorParams` = the detector's physical + response degrees of freedom, nested by physics
  (`per_pmt` is the Schur block; `tts`/`spe_width` are real fields; reflection is a superset
  the chosen model picks from).
- Everything else — geometry, medium reference shapes, `control_λ`, the source setup, the
  reflection-model choice, `N`/`K`/`temperature`, the overlap correction, the medium light speed
  `c/n`, numerical guards — is an **arg**.
