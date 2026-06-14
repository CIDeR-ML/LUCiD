# Merged `DetectorParams` — proposal for refactor-v2 reconciliation

**Audience:** the refactor-v2 author. **Purpose:** agree the merged `DetectorParams` shape *before*
any forward code lands (it touches your scintillation/ice loaders). This is the keystone of the
`unification`→`refactor-v2` merge (base = unification's AD-clean forward + `lucid/fitting/`; features
from refactor-v2 added on top). See `docs/RECONCILIATION_PLAN.md`.

## The two current shapes

**unification — NESTED** (5 physics sub-tuples; the optimizer tree-walks them generically; 23 leaves,
order pinned by `tests/reconciliation/` tripwire):

| sub-tuple | leaves |
|---|---|
| `ScatteringParams` | scatter_length, mie_scatter_length, g, rayleigh_dev, mie_dev |
| `AbsorptionParams` | absorption_length, abs_dev |
| `ReflectionParams` | wall_reflection_rate, sensor_reflection_rate, wall_R0, wall_p, wall_fspec, cathode_nr, cathode_nk, sensor_fspec |
| `ResponseParams` | qe, spe_width, tts, qe_dev |
| `PerPmtParams` | qe_corrections, gain, t0, walk |

**refactor-v2 — FLAT** + 8 scintillation scalars (NaN-default, read only when the medium's
`emission_processes` includes `"scintillation"`): `S, kB, C` (Chou light yield), `tau_rise, tau_fall`
(hypoexp timing), `moyal_amp, moyal_loc, moyal_scale` (emission spectrum).

## Proposal

**Keep unification's nested tree; add a 6th sub-tuple `ScintillationParams`, appended LAST.**

```python
class ScintillationParams(NamedTuple):       # NEW — neutral default NaN (only read when scintillating)
    S:          jax.Array = jnp.nan          # Chou light yield  dL/dx = S·(dE/dx)/(1+kB·(dE/dx)+C·(dE/dx)²)
    kB:         jax.Array = jnp.nan
    C:          jax.Array = jnp.nan
    tau_rise:   jax.Array = jnp.nan          # hypoexp emission timing (ns), differentiable
    tau_fall:   jax.Array = jnp.nan
    moyal_amp:  jax.Array = jnp.nan          # emission-spectrum shape (loc/scale closed at setup today)
    moyal_loc:  jax.Array = jnp.nan
    moyal_scale:jax.Array = jnp.nan

_SUBTUPLES = ( ('scattering', ScatteringParams), ('absorption', AbsorptionParams),
               ('reflection', ReflectionParams), ('response', ResponseParams),
               ('per_pmt', PerPmtParams), ('scintillation', ScintillationParams) )  # <- appended LAST
```

### Why these choices (and the three things to confirm)

1. **Appended LAST, not interleaved.** This preserves the existing 23-leaf flatten order, so every
   `ravel_pytree`/`normalize`/`make_optimization_mask` consumer and the Phase-0 tripwire leaf-order pin
   stay valid; the 8 scint leaves become 24–31. **Water-mode forward stays byte-identical** (scint
   leaves never read for non-scintillating media). ✅ low-risk.
   → *Confirm:* you're OK with scintillation as the last group (no preference for grouping it with
   emission/timing elsewhere).

2. **NaN default + EXCLUDED from optimization by default.** I keep your `NaN`-for-non-scintillating
   convention at the *physics* level. ⚠️ But NaN on a JAX pytree is dangerous for the *optimizer* —
   unification's `from_flat`/`default_bounds`/`make_optimization_mask` deliberately use neutral
   (finite) defaults so `normalize`/ravel never see NaN and gradients can't be poisoned. So the merged
   tree will: keep `ScintillationParams` NaN by default, **but exclude it from `default_bounds` and the
   trainable set unless a scintillating-medium fit is explicitly requested** (then the fields are
   populated finite from the material/medium and become trainable). This keeps water/Cherenkov fits
   exactly as today and only activates scint params for ice/WbLS scintillation calibration.
   → *Confirm:* the scint fields are never normalized/masked while NaN in your current forward (i.e.
   they're only consumed by the emission closure, never by the optimizer) — so excluding-by-default is
   purely additive for you.

3. **`from_flat` flat-JSON wire format stays canonical; your material loader feeds it.** unification's
   `DetectorParams.from_flat(**flat_kwargs)` + `_nest_flat_kwargs` is the bridge JSON authors use (no
   nesting knowledge required). Your ice/WbLS material `"scintillation"` block + `_scintillation_
   defaults_from_medium` populate the 8 scint kwargs through `from_flat` (NaN when absent). The abs/
   scatter material blocks already share `water.json`'s schema, so they load unchanged.
   → *Confirm:* the material-JSON `"scintillation"` block maps 1:1 to `{S,kB,C,tau_rise,tau_fall,
   moyal_amp,moyal_loc,moyal_scale}` (units: S [ph/MeV], kB [mm/keV], C [(mm/keV)²], τ [ns]) and that
   `moyal_loc/scale` being "closed over at setup" is compatible with also carrying them on the pytree.

## Downstream touch-list (FYI, not asking)
`from_flat` / `_nest_flat_kwargs` / `_FLAT_FIELDS` / `default_bounds` / `create_default` / `grad_scales`
/ `make_optimization_mask` all extend mechanically over the new sub-tuple (generic tree-walks). The
forward's `_get_optical_arrays` / emission closure reads `dp.scintillation.*` where it reads the flat
scint scalars today. The tripwire (`tests/reconciliation/`) re-captures once after the change with the
27→31-leaf order as the new pinned reference.

## One open question that affects YOU directly
The medium/optics seam: unification keeps `wavelength/optical_model.py` canonical (the λ-deviation seam
`lucid/fitting` depends on) and bridges your scintillation **emission** (which lives in your
`MediumProperties.emission_processes`) into it — we are *not* adopting `make_medium`+`spectrum` as the
fitting optics seam. Does any scintillation-emission code you wrote assume the `make_medium` path such
that bridging it onto `optical_model.py` would lose behavior? If so, flag it now.
