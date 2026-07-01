# What the unified framework looks like in practice (sources, wavelength, make_hits, QE)

## The pipeline (one data flow, every stage a composable, registry-dispatched component)

```
Source.emit(n, key)                       -> PhotonBatch(origins, dirs, weights, WAVELENGTHS)   # spectrum lives in the source
        |                                                                   |
        v                                                                   v
OpticalModel(DetectorParams, medium)(λ)   -> OpticalArrays(scatter_len, mie_len, abs_len,       # per-photon, pre-scan
        |                                      R0_wall(λ), nr_cathode(λ))                        # λ resolved here, ONCE
        v
PhotonStep scan (vmap over photons)        -> Deposits(weight, sensor_idx, time, wavelength)     # uses ReflectionModel in-step
        |   reflection_fn(cth_inc, R0_λ, nr_λ, ...) for the ANGLE part
        v
ResponseModel(DetectorParams)(deposits)    -> Observables(charge_mean, charge_var, t_first)      # QE(λ), per-PMT QE/gain/w/t0/walk/TTS
        |
        v
Loss / Fisher  (reads ParamRegistry: which leaves are free, transform, gauge, prior, observable)
```

Nothing in this flow is hard-coded: each stage is a registered/declared component. The six extensibility points are:
**source** (register a class), **spectrum** (register a sampler), **optical property** (a DetectorParams field + its
evaluator), **reflection model** (a function), **observable** (a ResponseModel method), **fit param** (a registry row).

---

## 1. Sources + wavelength — the spectrum belongs to the source

Today wavelength sampling is SPLIT off in `setup_event_simulator` (`sample_cherenkov_wavelengths`,
`build_qe_weighted_cherenkov_sampler`). That's the wrong seam. A photon's wavelength is a property of the SOURCE.
Put it there, in `lucid/sources/`:

```python
class PhotonBatch(NamedTuple):           # the universal emission contract
    origins: f32[n,3]; directions: f32[n,3]; weights: f32[n]; wavelengths: f32[n]

class Spectrum(Protocol):
    def sample(self, n, key) -> f32[n]                      # wavelengths
    def log_pdf(self, wl) -> f32[n]                         # for importance reweighting (optional)

# concrete spectra (registry):
Monochromatic(λ0)                # laser: delta  -> sample = full(n, λ0)
PowerLaw(alpha, lo, hi)          # Cherenkov 1/λ²  (alpha=2)  [BARE — calibration-safe]
Tabulated(grid, pdf)             # Xe lamp / POPOP / any measured spectrum
QEWeighted(base, qe_curve)       # importance-sampled — PRODUCTION ONLY (bakes QE into λ → CANNOT fit QE)

class Source(Protocol):
    spectrum: Spectrum
    def emit(self, n, key) -> PhotonBatch                   # samples geometry AND wavelengths

# concrete sources (registry, in lucid/sources/calibration_sources.py + siren_rays.py):
LaserSource(origin, direction, divergence, spectrum=Monochromatic(λ0))
IsotropicSource(origin, spectrum=PowerLaw(2, lo, hi))      # diffuser ball
CherenkovTrackSource(siren_net, particle_params, spectrum=PowerLaw(2,...))   # recon
DataSource(root_file)                                       # wavelengths come from file; spectrum=None
```

**Why this is best / what it fixes:**
- The currently-split wavelength sampling collapses into `source.emit` → one seam, the photon carries its own λ.
- **The cherenkov_qe trap is made explicit by type:** calibration sources MUST use a BARE spectrum (`PowerLaw`,
  `Tabulated`) so QE stays fittable in the ResponseModel; `QEWeighted` is flagged production-only. (This is the
  memory finding that the qe-importance-sampler can't fit qe, now enforced by the source's spectrum choice.)
- Add a source = register a class; add a spectrum (e.g. a real Xe/POPOP curve, SK 337/375/398/405/445 laser ladder) =
  register a `Spectrum`. No edits to the simulator.
- **Multi-intensity ladder** (the TQ map needs low+high occupancy) = a source with a list of intensities, or two
  source instances at scaled `weights` — a config choice, not engine code.

---

## 2. OpticalModel — λ resolved once, before the scan (Section in WAVELENGTH_DESIGN.md)

```python
class OpticalArrays(NamedTuple):         # per-photon (n,), passed into the scan
    scatter_len; mie_len; abs_len; R0_wall; nr_cathode

def optical_model(dp: DetectorParams, medium, control_λ):
    def eval(wl):                        # wl = batch.wavelengths (n,)
        ray = medium.rayleigh(wl) / dp.rayleigh_amp                          # amplitude × known 1/λ⁴
        mie = medium.mie(wl)      / (dp.mie_amp * interp(wl, control_λ, dp.mie_curve))  # amp × free curve (optional)
        ab  = medium.abs(wl)      / interp(wl, control_λ, dp.abs_curve)      # FREE curve × pure-water ref
        R0  = interp(wl, control_λ, dp.wall_R0_curve)                        # per-photon blacksheet R0(λ)
        nr  = interp(wl, control_λ, dp.cathode_nr_curve)                     # per-photon cathode n_real(λ)
        return OpticalArrays(1/ray, 1/mie, 1/ab, R0, nr)
    return eval
```
Reference shapes (medium, glass, g, n_imag) are FIXED constants here — NOT pytree fields. `control_λ` is a config
constant. Add a property = one field + one line. The two-mode `wavelength_mode` branch disappears: monochromatic is
just "all wl equal".

---

## 3. ReflectionModel — pluggable function, λ pre-resolved + angle in-step (your point #2)

```python
def reflection_fn(cth_inc, R0_λ, nr_λ, dp, hit_sensor, normal, dir, key) -> (refl_prob, new_dir, logp_refl):
    ...   # cth_inc from sg(normal) (magnitude pathwise); spec/diff sampled here; DiCE score for fspec returned

# interchangeable models (registry):
scalar_reflection      # wall/sensor_reflection_rate (recon today)
schlick_reflection     # R0_λ + (1-R0_λ)(1-cth)^wall_p ; spec/diff mix fspec
fresnel_reflection     # multilayer water/glass/cathode(nr_λ, nk) ; spec/diff mix
```
The step calls `reflection_fn`; it is agnostic to which. Magnitude params (R0, p, nr) are pathwise-exact; direction
fraction (fspec) carries a DiCE score — both verified in `refl_check2`. Add a reflection physics = add a function.

---

## 4. ResponseModel — make_hits done right: QE, per-PMT, gain, observables

This is the make_hits rewrite. **Separate TRANSPORT (deposits) from DETECTOR RESPONSE (observables).** Today make_hits
mixes them and bakes QE in. The response model owns QE(λ), per-PMT QE/gain, SPE width, timing response — all from
DetectorParams — and EXPOSES the observables.

**QE has three independent factors that must not be conflated** (the Phase-2 finding):
```
rate[p]   = Σ_{photons→p}  deposit_weight · qe_λ(λ) · qe_corr[p]      # EXPECTED PE count (the "rate")
                              └ global curve ┘   └ per-PMT efficiency ┘
charge_mean[p] = rate[p] · gain[p]                                    # gain = per-PMT charge-per-PE  (≠ efficiency!)
charge_var[p]  = rate[p] · gain[p]² · (1 + w²)                        # compound-Poisson  -> splits QE↔gain, gives w
t_first[p]     = softmin_{photons→p}(time ; weighted by deposit_weight·qe_λ·qe_corr)
                   + t0[p] + walk[p]·f(rate[p]) + TTS-smear            # per-PMT timing response
```
So the ResponseModel methods ARE the observables:
```python
class ExpectedResponse:        # the MODEL (differentiable, mean + analytic variance)
    charge_mean(deposits, dp) -> (NS,)
    charge_var(deposits, dp)  -> (NS,)        # NEW vs recon — the QE↔gain/w splitter
    first_arrival(deposits, dp)-> (NS,)
class SampleResponse:          # the TRUTH (shot noise): Bernoulli QE = qe_λ·qe_corr per photon, hard-min timing + TTS
    charge(deposits, dp) -> (NS,) integer-PE × sampled-SPE
    first_arrival(deposits, dp) -> (NS,)
```
Both read the SAME DetectorParams. Truth = SampleResponse; model = ExpectedResponse — the validated implicit==sample
equivalence. The old hit_modes (`aggregated/per_photon/realistic/shotgun/waveform`) become a CHOICE of response method,
not five code paths.

**Where QE(λ) is applied:** in the Response(detection), per-photon, NOT in the source (so it stays fittable; QEWeighted
source would bake it in — forbidden for calibration). qe_corr and gain are per-PMT (`dp.qe_corrections`, `dp.gain`).
Add an observable (e.g. charge-skewness, waveform) = add a ResponseModel method + a noise model.

---

## 5. DetectorParams + ParamRegistry — the only place "which is free" lives

`DetectorParams` (extended pytree, leaves = fittable DOF; from WAVELENGTH_DESIGN.md). The **ParamRegistry** declares,
per field, everything the fitter needs — so nothing is hard-coded in the optimizer:

```python
REGISTRY = {
 'abs_curve':       Spec(kind='curve', ref='pure_water', transform='log', prior='curvature', obs=['charge'], free=True),
 'mie_amp':         Spec(kind='scalar', transform='log', obs=['timing'], free=True),   # timing-measurable, charge-thin
 'wall_R0_curve':   Spec(kind='curve', transform='log', prior='curvature', obs=['charge','timing']),
 'cathode_nr_curve':Spec(kind='curve', ..., obs=['timing']),
 'qe_corrections':  Spec(kind='per_pmt', gauge='mean_log0', obs=['charge_mean'], marginalize=True),
 'gain':            Spec(kind='per_pmt', gauge='mean_log0', obs=['charge_var'],  marginalize=True),
 't0':              Spec(kind='per_pmt', gauge='mean_zero', obs=['timing'],      marginalize=True),
 'walk':            Spec(kind='per_pmt', obs=['timing_multilevel']),
 'spe_width':       Spec(kind='scalar', obs=['charge_var']),
 # recon adds: ParticleParams rows (energy:charge, pos/dir/t0:timing) — SAME machinery
}
```
The fitter: `θ, unravel = ravel_pytree(free_leaves)`; build the forward `θ -> unravel -> dp -> optical_model ->
step -> response -> observables`; GN/Schur marginalizes the `marginalize=True` per-PMT leaves; applies `prior`/`gauge`.
**Calibration vs reconstruction vs joint = which rows have `free=True`.** Nothing else changes.

---

## 6. Everything affected (the concrete change list)

| Component | Now | Becomes |
|---|---|---|
| `sources/calibration_sources.py`, `siren_rays.py` | emit geometry; λ sampled elsewhere | `Source.emit -> PhotonBatch(...,wavelengths)`; `Spectrum` registry |
| `simulation/simulator.py::_get_optical_arrays` | 60-line closure, 2 modes, medium baked | extract `optical_model(dp,medium,control_λ)`; ONE path |
| `simulation/photon_step.py` | scalar reflection in-step | per-photon `R0_λ,nr_λ` in; `reflection_fn(...)` pluggable |
| `overlap.py` reflection | scalar rate | n/a (moved into `reflection_fn`) |
| `simulation/sensor_response.py` / make_hits | mixes transport+QE; 5 hit_modes | `ResponseModel` (Expected/Sample); QE(λ)/per-PMT/gain/w/t0/walk; observables = methods |
| `detector_params.py::DetectorParams` | 7 scalars + qe_corrections | + amplitude scalars + curve arrays + gain/t0/walk/w/tts; bounds via `tree.map` already work |
| (new) `fitting/registry.py` | — | `ParamRegistry` (transform/gauge/prior/obs/free/marginalize) |
| (new) `fitting/gauss_newton.py`, `fisher.py` | recon `gnrec` / calib `gn_fast` separately | one GN+Schur, one CRB, read the registry |
| `setup_event_simulator` | orchestrates with flags | orchestrates source→optical→step→response; flags become component choices |

## 7. Design decisions / options, and the recommendation
- **λ in the source vs the simulator:** → SOURCE (the photon owns its λ; collapses the split sampling; makes QEWeighted
  vs bare a typed, calibration-safe choice).
- **QE in source vs response:** → RESPONSE, per-photon (keeps QE fittable; never bake into λ-sampling for calibration).
- **One response with observable-methods vs five hit_modes:** → observable methods (charge_mean / charge_var / first
  arrival); Expected (model) vs Sample (truth) is the only real fork.
- **Reflection in-step:** → pluggable `reflection_fn`, λ pre-resolved (R0_λ, nr_λ), angle in-step.
- **DetectorParams curves as fields vs external:** → FIELDS (fittable, bounded, pytree); reference shapes stay external
  constants; `ravel_pytree` bridges to the GN flat vector.
- **Where "which is free" lives:** → the ParamRegistry ONLY (calibration/recon/joint differ by `free` flags).
- **custom_vjp:** keep as the 2nd-order/recon backstop; make `nan_to_num` opt-in+logged; fix NaNs at source.

**One-line:** a photon is `(geometry, weight, wavelength)` from a `Source`; `OpticalModel` turns its λ into per-photon
optical scalars; the `PhotonStep` transports it with a pluggable `reflection_fn`; the `ResponseModel` turns deposits
into observables applying QE(λ)/per-PMT-QE/gain/w/t0/walk; `DetectorParams` holds exactly the fittable DOF and the
`ParamRegistry` says which are free — so calibration, reconstruction, and joint self-calibration are the same code with
different `free` flags, and every physics addition is a new registered component, nothing hard-coded.
```
