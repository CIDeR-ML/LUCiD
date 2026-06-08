"""Pluggable reflection models for the differentiable photon transport step.

A reflection model is a function with the signature::

    reflection_fn(direction, normal, hit_sensor, refl_params, lam, key)
        -> (refl_prob, refl_dir, lr_score)

where

- ``direction`` : (3,) incoming photon direction (LIVE — carries the pathwise track gradient)
- ``normal``    : (3,) outward surface normal (the model detaches it where curvature would blow up)
- ``hit_sensor``: bool, sensor (True) vs wall (False)
- ``refl_params``: a model-specific pytree of the fittable reflection parameters
                   (for the scalar model, the two reflection rates)
- ``lam``       : per-photon wavelength (nm); used by wavelength-dependent models, ignored otherwise
- ``key``       : a PRNGKey slot for the reflection's stochastic direction choice

and returns

- ``refl_prob`` : reflection probability (used as ``1 - refl_prob`` in the implicit-capture
                  deposit and ``refl_prob`` in the continuation weight)
- ``refl_dir``  : the post-reflection direction
- ``lr_score``  : a DiCE score increment for any DISCRETE reflection branch — 0.0 for the
                  scalar model; the specular/diffuse-mix log-prob for angular models

The model is chosen at setup and captured in the photon step's closure, so the
``custom_vjp`` step signature stays fixed (``refl_params`` is a single packed pytree
argument) — adding a new model never reshapes it. ``scalar_reflection`` is the default
and reproduces the legacy angle-independent behaviour byte-for-byte.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lucid.simulation.optics import (
    compute_reflection_direction, sample_cosine_hemisphere,
)

sg = jax.lax.stop_gradient


# ---------------------------------------------------------------------------
# Scalar model (default) — angle/λ-independent rates, hard direction.
# ---------------------------------------------------------------------------

class ScalarReflection(NamedTuple):
    """``refl_params`` for the scalar reflection model.

    Fields
    ------
    wall_rate : jnp.ndarray     scalar reflection probability at walls
    sensor_rate : jnp.ndarray   scalar reflection probability at sensors
    """
    wall_rate: jnp.ndarray
    sensor_rate: jnp.ndarray


def scalar_reflection(direction, normal, hit_sensor, refl_params, lam, key):
    """Angle/λ-independent reflection: scalar wall/sensor rates, hard direction
    (walls diffuse, sensors specular). Byte-identical to the legacy photon step.

    The normal is detached (``sg``) for the reflected direction — the Igehy (1999)
    curvature term through a live normal compounds ~1/r across bounces. ``lam`` is
    unused.
    """
    refl_prob = jnp.where(hit_sensor, refl_params.sensor_rate, refl_params.wall_rate)
    normal_refl = sg(normal)
    specular_dir = compute_reflection_direction(direction, normal_refl)
    diffuse_dir = sample_cosine_hemisphere(-normal_refl, key)
    refl_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    lr_score = jnp.zeros_like(refl_prob)
    return refl_prob, refl_dir, lr_score


# ---------------------------------------------------------------------------
# Angular model — Schlick blacksheet (wall) + multilayer-Fresnel cathode (sensor).
# Ported from the validated mie_hunter/refl_engine2.py. The reflectivity MAGNITUDE
# (R0w, pw, nr, nk) is PATHWISE-exact because cth_inc uses sg(normal); the spec/diff
# DIRECTION mix (fractions fw, fs) is a DISCRETE branch carried by a DiCE score lr.
# ---------------------------------------------------------------------------

N_WATER = 1.33


def n_glass(lam):
    """SK PMT-glass dispersion n_g(λ), λ in nm."""
    return 1.472 + 3670.0 / (lam * lam)


def fresnel_rr(ci, n_i, n_t):
    """Unpolarised Fresnel reflectance, real→real (ci = cos incidence).

    Returns (R, cos_transmit); total-internal-reflection clamps R→1.
    """
    s2t = (n_i / n_t) ** 2 * (1.0 - ci * ci)
    ct = jnp.sqrt(jnp.clip(1.0 - s2t, 0.0, 1.0))
    rs = (n_i * ci - n_t * ct) / (n_i * ci + n_t * ct + 1e-12)
    rp = (n_t * ci - n_i * ct) / (n_t * ci + n_i * ct + 1e-12)
    R = 0.5 * (rs * rs + rp * rp)
    return jnp.clip(jnp.where(s2t >= 1.0, 1.0, R), 0.0, 1.0), ct


def fresnel_rc(ci, n_i, n_c):
    """Unpolarised Fresnel reflectance, real→COMPLEX (absorbing cathode)."""
    ci = ci.astype(jnp.complex64)
    n_i = jnp.asarray(n_i, jnp.complex64)
    n_c = jnp.asarray(n_c, jnp.complex64)
    ct = jnp.sqrt(1.0 - (n_i / n_c) ** 2 * (1.0 - ci * ci))
    rs = (n_i * ci - n_c * ct) / (n_i * ci + n_c * ct)
    rp = (n_c * ci - n_i * ct) / (n_c * ci + n_i * ct)
    return jnp.clip(0.5 * (jnp.abs(rs) ** 2 + jnp.abs(rp) ** 2).real, 0.0, 1.0)


def pmt_reflectance(cth_inc, lam, n_r, n_k):
    """Effective 4-level PMT reflectance: water→glass(λ)→cathode(n_r+i·n_k),
    two incoherent Fresnel interfaces summed over multi-bounce."""
    ng = n_glass(lam)
    R1, ctg = fresnel_rr(cth_inc, N_WATER, ng)              # water→glass (real)
    R2 = fresnel_rc(ctg, ng, jnp.asarray(n_r) + 1j * jnp.asarray(n_k))  # glass→cathode (complex)
    return jnp.clip(R1 + (1.0 - R1) ** 2 * R2 / (1.0 - R1 * R2 + 1e-9), 0.0, 0.999)


class AngularReflection(NamedTuple):
    """``refl_params`` for the angular reflection model.

    Fields
    ------
    R0w : jnp.ndarray   blacksheet normal-incidence reflectance (Schlick)
    pw : jnp.ndarray    blacksheet Schlick angular exponent
    fw : jnp.ndarray    blacksheet specular fraction (1-fw diffuse)
    nr : jnp.ndarray    cathode real refractive index
    nk : jnp.ndarray    cathode imaginary refractive index (absorption)
    fs : jnp.ndarray    cathode specular fraction (1-fs diffuse)
    """
    R0w: jnp.ndarray
    pw: jnp.ndarray
    fw: jnp.ndarray
    nr: jnp.ndarray
    nk: jnp.ndarray
    fs: jnp.ndarray


def angular_reflection(direction, normal, hit_sensor, refl_params, lam, key):
    """Schlick blacksheet (wall) + multilayer-Fresnel cathode (sensor) reflection.

    Magnitude is angle/λ-dependent and pathwise-exact (cth_inc uses sg(normal));
    the reflected direction is a specular/diffuse mixture whose discrete branch is
    carried by the returned DiCE score ``lr``.
    """
    normal_refl = sg(normal)
    cth_inc = jnp.clip(jnp.abs(jnp.sum(direction * normal_refl)), 0.0, 1.0)

    Rw = refl_params.R0w + (1.0 - refl_params.R0w) * (1.0 - cth_inc) ** jnp.clip(refl_params.pw, 0.5, 12.0)
    Rs = pmt_reflectance(cth_inc, lam, refl_params.nr, refl_params.nk)
    refl_prob = jnp.clip(jnp.where(hit_sensor, Rs, Rw), 0.0, 0.999)

    # Direction: specular/diffuse mixture, fraction f_eff per surface. is_spec is a
    # DISCRETE branch → DiCE-scored (reflected photons only; score detaches f_eff).
    kd, ks = jax.random.split(key)
    f_eff = jnp.clip(jnp.where(hit_sensor, refl_params.fs, refl_params.fw), 1e-3, 1.0 - 1e-3)
    is_spec = jax.random.uniform(ks) < sg(f_eff)
    specular_dir = compute_reflection_direction(direction, normal_refl)
    diffuse_dir = sample_cosine_hemisphere(-normal_refl, kd)
    refl_dir = jnp.where(is_spec, specular_dir, diffuse_dir)
    lr_score = jnp.where(is_spec, jnp.log(f_eff), jnp.log1p(-f_eff))
    return refl_prob, refl_dir, lr_score


# ---------------------------------------------------------------------------
# Model registry — name → (reflection_fn, build_refl_params(detector_params)).
# build_refl_params extracts the model's parameters from a DetectorParams pytree.
# The scalar model is the byte-identical default; 'angular' needs per-photon λ
# (wavelength_mode=True), threaded by the simulator.
# ---------------------------------------------------------------------------

def _build_scalar_params(detector_params):
    return ScalarReflection(
        wall_rate=detector_params.reflection.wall_reflection_rate,
        sensor_rate=detector_params.reflection.sensor_reflection_rate,
    )


def _build_angular_params(detector_params):
    r = detector_params.reflection
    return AngularReflection(R0w=r.wall_R0, pw=r.wall_p, fw=r.wall_fspec,
                             nr=r.cathode_nr, nk=r.cathode_nk, fs=r.sensor_fspec)


REFLECTION_MODELS = {
    'scalar': (scalar_reflection, _build_scalar_params),
    'angular': (angular_reflection, _build_angular_params),
}

# Reflection models that require a per-photon wavelength (→ wavelength_mode=True).
WAVELENGTH_REFLECTION_MODELS = frozenset({'angular'})


def get_reflection_model(name):
    """Return ``(reflection_fn, build_refl_params)`` for a registered model name."""
    if name not in REFLECTION_MODELS:
        raise ValueError(
            f"Unknown reflection model {name!r}; available: {sorted(REFLECTION_MODELS)}")
    return REFLECTION_MODELS[name]
