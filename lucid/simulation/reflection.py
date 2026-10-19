"""Pluggable reflection models for the differentiable photon transport step.

A reflection model is a function with the signature::

    reflection_fn(direction, normal, hit_sensor, refl_params, lam, key,
                  polarization=None)
        -> (refl_prob, refl_dir, lr_score)

where

- ``direction`` : (3,) incoming photon direction (LIVE — carries the pathwise track gradient)
- ``normal``    : (3,) outward surface normal (the model detaches it where curvature would blow up)
- ``hit_sensor``: bool, sensor (True) vs wall (False)
- ``refl_params``: a model-specific pytree of the fittable reflection parameters
                   (for the scalar model, the two reflection rates)
- ``lam``       : per-photon wavelength (nm); used by wavelength-dependent models, ignored otherwise
- ``key``       : a PRNGKey slot for the reflection's stochastic direction choice
- ``polarization``: optional (3,) linear-polarization vector; SK tabulated
                    models use it for their S/P reflection probabilities

and returns

- ``refl_prob`` : reflection probability (used as ``1 - refl_prob`` in the implicit-capture
                  deposit and ``refl_prob`` in the continuation weight; its sensor value at
                  normal incidence also sets how QE converts a deposit, see
                  ``sensor_normal_reflectance``)
- ``refl_dir``  : the post-reflection direction
- ``lr_score``  : a DiCE score increment for any DISCRETE reflection branch — 0.0 for the
                  scalar model; the specular/diffuse-mix log-prob for angular models

The model is chosen at setup and captured in the photon step's closure, so the
``custom_vjp`` step signature stays fixed (``refl_params`` is a single packed pytree
argument) — adding a new model never reshapes it. ``scalar_mix`` is the DEFAULT (scalar
rates + a specular/diffuse direction mixture); ``scalar_reflection`` is the legacy model
that reproduces the pre-mixture angle-independent behaviour byte-for-byte.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.optics import (
    compute_reflection_direction, create_local_frame, sample_cosine_hemisphere,
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


def scalar_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
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
# Scalar-mix model — angle/λ-independent MAGNITUDE (wall/sensor rates) but a
# specular/diffuse DIRECTION mixture (fractions fw, fs). The magnitude is
# pathwise (scalar rates); the spec/diff direction is a DISCRETE branch carried
# by a DiCE score. Needs no per-photon wavelength, so it drops straight into the
# scalar (wavelength_mode=False) calibration.
# ---------------------------------------------------------------------------

class ScalarMixReflection(NamedTuple):
    """``refl_params`` for the scalar specular/diffuse-mixture reflection model.

    Fields
    ------
    wall_rate : jnp.ndarray     scalar wall reflection probability
    sensor_rate : jnp.ndarray   scalar sensor reflection probability
    wall_fspec : jnp.ndarray    wall specular fraction (1-fspec diffuse)
    sensor_fspec : jnp.ndarray  sensor specular fraction (1-fspec diffuse)
    """
    wall_rate: jnp.ndarray
    sensor_rate: jnp.ndarray
    wall_fspec: jnp.ndarray
    sensor_fspec: jnp.ndarray


def scalar_mix_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
    """Angle/λ-independent reflection MAGNITUDE with a specular/diffuse DIRECTION mixture.

    Reflection probability is the scalar wall/sensor rate (pathwise). Each reflected
    photon goes specular with probability ``fspec`` and diffuse (cosine-hemisphere)
    otherwise — a DISCRETE branch carried by the DiCE score ``lr`` (the score detaches
    ``f_eff`` so only the discrete choice, not the magnitude, is score-corrected).
    ``lam`` is unused.
    """
    refl_prob = jnp.where(hit_sensor, refl_params.sensor_rate, refl_params.wall_rate)
    normal_refl = sg(normal)
    kd, ks = jax.random.split(key)
    f_eff = jnp.clip(jnp.where(hit_sensor, refl_params.sensor_fspec, refl_params.wall_fspec),
                     1e-3, 1.0 - 1e-3)
    is_spec = jax.random.uniform(ks) < sg(f_eff)
    specular_dir = compute_reflection_direction(direction, normal_refl)
    diffuse_dir = sample_cosine_hemisphere(-normal_refl, kd)
    refl_dir = jnp.where(is_spec, specular_dir, diffuse_dir)
    lr_score = jnp.where(is_spec, jnp.log(f_eff), jnp.log1p(-f_eff))
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


def angular_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
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
# Tabulated tuned reflection model.
#
# This model combines a tuned black-sheet magnitude and
# Lambert/Phong/specular direction mixture with an externally supplied PMT
# reflectance table.  The unpolarized average remains the fallback; when the
# simulator carries polarization, the SK-IV S/P convention is applied per ray.
# ---------------------------------------------------------------------------

class PmtOpticalTable(NamedTuple):
    """Unpolarised SK PMT optical probabilities on the 108x200 table grid."""
    reflectance: jnp.ndarray
    transmittance: jnp.ndarray
    absorptance: jnp.ndarray
    reflectance_s: jnp.ndarray | None = None
    reflectance_p: jnp.ndarray | None = None


class PmtOpticalOutcome(NamedTuple):
    """Mutually exclusive SK-like PMT optical-outcome probabilities."""
    reflectance: jnp.ndarray
    absorptance: jnp.ndarray
    transmittance: jnp.ndarray


@lru_cache(maxsize=None)
def _load_pmt_optical_table_cached(path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.shape != (108 * 200, 8):
        raise ValueError(
            f"PMT reflection table {path!r} has shape {data.shape}; "
            "expected (21600, 8)")
    data = data.reshape(108, 200, 8)
    # Columns: wavelength, angle, Rs, Rp, Ts, Tp, As, Ap. Preserve the
    # component reflectances as well as their unpolarized averages.
    averages = tuple(
        np.asarray(0.5 * (data[:, :, a] + data[:, :, b]), dtype=np.float32)
        for a, b in ((2, 3), (4, 5), (6, 7)))
    return averages + (
        np.asarray(data[:, :, 2], dtype=np.float32),
        np.asarray(data[:, :, 3], dtype=np.float32),
    )


def load_pmt_optical_table(path=None):
    """Load user-supplied SK-like PMT reflectance, transmittance and absorptance.

    Pass ``path`` explicitly or set ``LUCID_PMT_REFLECTANCE_TABLE``. The
    expected format is a whitespace-delimited table on a 108-wavelength by
    200-angle grid, ordered by wavelength then angle. Each row contains eight
    columns: wavelength, angle, Rs, Rp, Ts, Tp, As, Ap.
    """
    if path is None:
        env_path = os.environ.get("LUCID_PMT_REFLECTANCE_TABLE")
        if not env_path:
            raise FileNotFoundError(
                "No PMT optical table was configured. Pass path=... or set "
                "LUCID_PMT_REFLECTANCE_TABLE.")
        path = env_path
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    reflection, transmission, absorption, reflection_s, reflection_p = (
        _load_pmt_optical_table_cached(str(path)))
    return PmtOpticalTable(*(jnp.asarray(x) for x in
                             (reflection, transmission, absorption,
                              reflection_s, reflection_p)))


def load_pmt_reflectance_table(path=None):
    """Load an externally supplied unpolarised PMT reflectance table.

    Pass ``path`` explicitly or set ``LUCID_PMT_REFLECTANCE_TABLE``. The table
    itself is detector-specific input data and is intentionally not bundled
    with LUCiD.
    """
    return load_pmt_optical_table(path).reflectance


class TabulatedTunedReflection(NamedTuple):
    """Parameters for the tabulated tuned reflection model."""
    pmt_reflectance_table: jnp.ndarray
    pmt_reflectance_s: jnp.ndarray | None = None
    pmt_reflectance_p: jnp.ndarray | None = None


class TabulatedScaledReflection(NamedTuple):
    """SK tabulated/tuned reflection with differentiable magnitude scales.

    A scale of one reproduces :class:`TabulatedTunedReflection`.  Separate wall
    and sensor fields keep the model useful for later fits, while a calibration
    parameterisation may tie them to one common reflection coordinate.
    """
    pmt_reflectance_table: jnp.ndarray
    wall_scale: jnp.ndarray
    sensor_scale: jnp.ndarray
    pmt_reflectance_s: jnp.ndarray | None = None
    pmt_reflectance_p: jnp.ndarray | None = None


def _sk4_p_table_weight(direction, normal, polarization):
    """Return SK-IV's probability of selecting the table's p column.

    The physical s-polarization weight is ``|E dot s|^2``.  SK-IV's legacy
    branch swaps the physical labels when choosing the optical-table column,
    so physical s selects the p column and physical p selects the s column.
    This is the same convention used by ``pmt_detection.py``.
    """
    s_axis_raw = jnp.cross(direction, normal)
    s_norm2 = jnp.sum(s_axis_raw * s_axis_raw)
    s_axis = s_axis_raw / jnp.sqrt(s_norm2 + 1e-12)
    physical_s_weight = jnp.sum(polarization * s_axis) ** 2
    return jnp.where(
        s_norm2 > 1e-10, jnp.clip(physical_s_weight, 0.0, 1.0), 0.5)


def tuned_blacksheet_reflectance(
        cth_inc, lam, direction=None, normal=None, polarization=None):
    """Tuned wavelength- and incidence-dependent black-sheet reflectance.

    The Fresnel magnitude is multiplied by a bilinear angle/wavelength
    correction and a fixed 0.5 material scale.
    """
    cth_inc = jnp.clip(cth_inc, 0.0, 1.0)
    theta = jnp.arccos(cth_inc)
    nbs = jnp.asarray(2.01 / 1.34, dtype=cth_inc.dtype)
    theta_t = jnp.arcsin(jnp.clip(jnp.sin(theta) / nbs, 0.0, 1.0))

    rp = ((nbs * jnp.cos(theta) - jnp.cos(theta_t)) /
          (nbs * jnp.cos(theta) + jnp.cos(theta_t) + 1e-12)) ** 2
    rs = ((jnp.cos(theta) - nbs * jnp.cos(theta_t)) /
          (jnp.cos(theta) + nbs * jnp.cos(theta_t) + 1e-12)) ** 2
    if polarization is None or direction is None or normal is None:
        fresnel = 0.5 * (rp + rs)
    else:
        p_table_weight = _sk4_p_table_weight(
            direction, normal, polarization)
        fresnel = p_table_weight * rp + (1.0 - p_table_weight) * rs

    wavelengths = jnp.asarray([337.0, 400.0, 420.0], dtype=cth_inc.dtype)
    angles = jnp.asarray([0.0, 30.0, 45.0, 60.0, 90.0], dtype=cth_inc.dtype)
    corrections = jnp.asarray([
        [2.00000, 2.00000, 2.00000],
        [1.60904, 1.24411, 1.24317],
        [1.36482, 1.13971, 1.14026],
        [1.20353, 0.970104, 0.951023],
        [1.00000, 1.00000, 1.00000],
    ], dtype=cth_inc.dtype)

    angle_deg = theta * (180.0 / jnp.pi)
    angle_values = jax.vmap(lambda column: jnp.interp(
        angle_deg, angles, column))(corrections.T)
    correction = jnp.interp(lam, wavelengths, angle_values)
    return jnp.clip(0.5 * fresnel * correction, 0.0, 1.0)


def _pmt_table_lookup(cth_inc, lam, table, *, sk_forward=True):
    """Evaluate one SK PMT table component with useful forward/reverse AD.

    ``sk_forward=True`` uses the SK-like hard-bin lookup in the numerical
    forward pass, but uses a bilinear surrogate derivative. The expression
    ``smooth + stop_gradient(hard-smooth)`` gives the same finite derivative
    to JAX forward-mode JVP and reverse-mode VJP without changing the hard-bin value.
    ``sk_forward=False`` returns ordinary bilinear interpolation.
    """
    cth_inc = jnp.clip(cth_inc, 0.0, 1.0)
    angle_deg = jnp.arccos(cth_inc) * (180.0 / jnp.pi)

    # SK-like hard bins: 1 nm in wavelength and 0.45 degrees in angle.
    wi_hard = jnp.clip(jnp.floor(lam - 337.0).astype(jnp.int32), 0, 107)
    ai_hard = jnp.clip(jnp.floor(angle_deg / 0.45).astype(jnp.int32), 0, 199)
    hard = table[wi_hard, ai_hard]

    # Bilinear surrogate on the physical table centres (337.5+iwav nm and
    # 0.225+0.45*iang degrees). Clamp at the first/last centre.
    wf = jnp.clip(lam - 337.5, 0.0, 107.0)
    af = jnp.clip((angle_deg - 0.225) / 0.45, 0.0, 199.0)
    w0 = jnp.floor(wf).astype(jnp.int32)
    a0 = jnp.floor(af).astype(jnp.int32)
    w1 = jnp.minimum(w0 + 1, 107)
    a1 = jnp.minimum(a0 + 1, 199)
    dw = wf - w0.astype(wf.dtype)
    da = af - a0.astype(af.dtype)
    smooth = ((1.0 - dw) * ((1.0 - da) * table[w0, a0] + da * table[w0, a1])
              + dw * ((1.0 - da) * table[w1, a0] + da * table[w1, a1]))
    return hard + smooth - sg(smooth) if sk_forward else smooth


def tabulated_pmt_reflectance(cth_inc, lam, table, *, sk_forward=True):
    """Evaluate unpolarised PMT reflectance from an SK-like regular table.

    By default, the value is the SK-like hard-bin value and the derivative is the
    local bilinear-table slope; see :func:`_pmt_table_lookup`.
    """
    return jnp.clip(_pmt_table_lookup(
        cth_inc, lam, table, sk_forward=sk_forward), 0.0, 1.0)


def tabulated_pmt_optical_outcome(cth_inc, lam, table, *, sk_forward=True):
    """Return the SK-like PMT reflection/absorption/transmission budget.

    The SK-like model treats tabulated ``R`` and ``A`` as separate outcomes
    and assigns the remaining probability to transmission. We calculate
    ``T`` as ``1-R-A`` to conserve the probability budget. The separately
    tabulated transmittance remains available in :class:`PmtOpticalTable` for closure
    diagnostics.
    """
    reflection = tabulated_pmt_reflectance(
        cth_inc, lam, table.reflectance, sk_forward=sk_forward)
    absorption = jnp.clip(_pmt_table_lookup(
        cth_inc, lam, table.absorptance, sk_forward=sk_forward), 0.0, 1.0)
    transmission = jnp.clip(1.0 - reflection - absorption, 0.0, 1.0)
    return PmtOpticalOutcome(reflection, absorption, transmission)


def _sample_phong_about_specular(specular_dir, inward_normal, key):
    """Sample a power-10 Phong lobe inside the detector.

    This is the same eight-try rejection distribution as drawing complete
    Phong directions and rejecting the outward ones, but marginalises the
    polar angle during rejection. A proposed azimuth has an analytic
    acceptance probability; after accepting it, draw the polar cosine from
    the corresponding truncated power law. Only the selected proposal needs
    a 3-D direction and local-frame transform. This preserves the target
    distribution and the 2**-8 worst-case fallback probability while avoiding
    eight direction constructions per photon per propagation step.
    """
    n_proposals = 8
    phi_key, accept_key, theta_key = jax.random.split(key, 3)
    phi = 2.0 * jnp.pi * jax.random.uniform(phi_key, (n_proposals,))

    # Express the inward normal in a frame whose z axis is specular_dir and
    # whose x axis points toward its transverse projection. For cos(phi)>=0
    # every polar angle is inward. On the far half-plane, inwardness requires
    # cos(theta)>x_min. With a power-10 Phong density, the marginal acceptance
    # is therefore 1-x_min**11.
    c_inc = jnp.clip(jnp.sum(specular_dir * inward_normal), 0.0, 1.0)
    s_inc = jnp.sqrt(jnp.clip(1.0 - c_inc * c_inc, 0.0, 1.0))
    cos_phi = jnp.cos(phi)
    far = jnp.maximum(-cos_phi, 0.0)
    x_min_denominator2 = jnp.maximum(
        c_inc * c_inc + s_inc * s_inc * far * far, 0.0)
    smooth_x_min = s_inc * far / jnp.sqrt(x_min_denominator2 + 1e-12)
    exact_x_min = jnp.where(
        far > 0.0,
        s_inc * far / jnp.maximum(jnp.sqrt(x_min_denominator2), 1e-12),
        0.0,
    )
    # Keep the exact grazing-incidence forward boundary while differentiating
    # through the finite denominator. This avoids sqrt'(0) in reverse mode.
    x_min = smooth_x_min + sg(exact_x_min - smooth_x_min)
    x_min_power = x_min ** 11
    accepted = jax.random.uniform(accept_key, (n_proposals,)) < (
        1.0 - x_min_power)
    first_accepted = jnp.argmax(accepted.astype(jnp.int32))
    found = jnp.any(accepted)

    selected_phi = phi[first_accepted]
    selected_min_power = x_min_power[first_accepted]
    u = jax.random.uniform(theta_key)
    cos_theta = (selected_min_power + u * (1.0 - selected_min_power)) ** (
        1.0 / 11.0)
    sin_theta = jnp.sqrt(jnp.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))

    default_frame = create_local_frame(specular_dir)
    transverse_normal = inward_normal - c_inc * specular_dir
    aligned_x = transverse_normal / jnp.sqrt(
        jnp.sum(transverse_normal * transverse_normal) + 1e-12)
    frame_x = jnp.where(s_inc > 1e-6, aligned_x, default_frame[0])
    frame_y = jnp.cross(specular_dir, frame_x)
    proposal = (cos_theta * specular_dir
                + sin_theta * (jnp.cos(selected_phi) * frame_x
                               + jnp.sin(selected_phi) * frame_y))
    return jnp.where(found, proposal, specular_dir)


def _tabulated_tuned_reflection_components(
        direction, normal, hit_sensor, refl_params, lam, key, *,
        enable_wall=True, enable_sensor=True, polarization=None):
    """Tabulated/tuned reflection with static wall/PMT ablation switches."""
    normal_refl = sg(normal)
    inward_normal = -normal_refl
    cth_inc = jnp.clip(jnp.abs(jnp.sum(direction * normal_refl)), 0.0, 1.0)

    wall_reflectance = tuned_blacksheet_reflectance(
        cth_inc, lam, direction, normal_refl, polarization)
    sensor_reflectance = tabulated_pmt_reflectance(
        cth_inc, lam, refl_params.pmt_reflectance_table)
    pmt_s = getattr(refl_params, 'pmt_reflectance_s', None)
    pmt_p = getattr(refl_params, 'pmt_reflectance_p', None)
    if polarization is not None and pmt_s is not None and pmt_p is not None:
        p_table_weight = _sk4_p_table_weight(
            direction, normal_refl, polarization)
        reflectance_s = tabulated_pmt_reflectance(cth_inc, lam, pmt_s)
        reflectance_p = tabulated_pmt_reflectance(cth_inc, lam, pmt_p)
        sensor_reflectance = (
            p_table_weight * reflectance_p
            + (1.0 - p_table_weight) * reflectance_s)
    wall_reflectance = wall_reflectance * getattr(
        refl_params, 'wall_scale', jnp.asarray(1.0))
    sensor_reflectance = sensor_reflectance * getattr(
        refl_params, 'sensor_scale', jnp.asarray(1.0))
    if not enable_wall:
        wall_reflectance = jnp.zeros_like(wall_reflectance)
    if not enable_sensor:
        sensor_reflectance = jnp.zeros_like(sensor_reflectance)
    refl_prob = jnp.where(hit_sensor, sensor_reflectance, wall_reflectance)

    specular_dir = compute_reflection_direction(direction, normal_refl)
    branch_key, diffuse_key, phong_key = jax.random.split(key, 3)

    # Fixed Lambert component, angle-dependent specular component, and the
    # remainder in a power-10 Phong lobe.
    specular_fraction = jnp.where(cth_inc <= (2.0 / 3.0),
                                  1.0 - 1.2 * cth_inc, 0.2)
    lambert_fraction = jnp.minimum(
        0.015 / jnp.maximum(wall_reflectance, 1e-12),
        1.0 - specular_fraction)
    phong_fraction = jnp.maximum(1.0 - lambert_fraction - specular_fraction, 0.0)

    branch = jax.random.uniform(branch_key)
    lambert_dir = sample_cosine_hemisphere(inward_normal, diffuse_key)
    phong_dir = _sample_phong_about_specular(specular_dir, inward_normal, phong_key)
    wall_dir = jnp.where(
        branch <= lambert_fraction,
        lambert_dir,
        jnp.where(branch <= lambert_fraction + phong_fraction,
                  phong_dir, specular_dir),
    )

    # Reflected sensor photons are specular.
    refl_dir = jnp.where(hit_sensor, specular_dir, wall_dir)
    return jnp.clip(refl_prob, 0.0, 0.999), refl_dir, jnp.zeros_like(refl_prob)


def tabulated_tuned_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
    """Full SK PMT-table plus tuned-black-sheet reflection model."""
    return _tabulated_tuned_reflection_components(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=polarization)


def tabulated_scaled_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
    """SK tabulated/tuned reflection with differentiable wall/PMT scales."""
    return _tabulated_tuned_reflection_components(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=polarization)


def tabulated_pmt_only_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
    """Diagnostic ablation: retain PMT reflection and disable blacksheet reflection."""
    return _tabulated_tuned_reflection_components(
        direction, normal, hit_sensor, refl_params, lam, key,
        enable_wall=False, polarization=polarization)


def tuned_wall_only_reflection(
        direction, normal, hit_sensor, refl_params, lam, key,
        polarization=None):
    """Diagnostic ablation: retain blacksheet reflection and disable PMT reflection."""
    return _tabulated_tuned_reflection_components(
        direction, normal, hit_sensor, refl_params, lam, key,
        enable_sensor=False, polarization=polarization)


# ---------------------------------------------------------------------------
# Model registry — name → (reflection_fn, build_refl_params(detector_params)).
# build_refl_params extracts the model's parameters from a DetectorParams pytree.
# 'scalar_mix' is the default; 'scalar' is the byte-identical legacy model; 'angular'
# needs per-photon λ (wavelength_mode=True), threaded by the simulator.
# ---------------------------------------------------------------------------

def _build_scalar_params(detector_params):
    return ScalarReflection(
        wall_rate=detector_params.reflection.wall_reflection_rate,
        sensor_rate=detector_params.reflection.sensor_reflection_rate,
    )


def _build_scalar_mix_params(detector_params):
    r = detector_params.reflection
    return ScalarMixReflection(
        wall_rate=r.wall_reflection_rate, sensor_rate=r.sensor_reflection_rate,
        wall_fspec=r.wall_fspec, sensor_fspec=r.sensor_fspec)


def _build_angular_params(detector_params):
    r = detector_params.reflection
    return AngularReflection(R0w=r.wall_R0, pw=r.wall_p, fw=r.wall_fspec,
                             nr=r.cathode_nr, nk=r.cathode_nk, fs=r.sensor_fspec)


def _build_tabulated_tuned_params(detector_params):
    del detector_params
    table = load_pmt_optical_table()
    return TabulatedTunedReflection(
        pmt_reflectance_table=table.reflectance,
        pmt_reflectance_s=table.reflectance_s,
        pmt_reflectance_p=table.reflectance_p)


def _build_tabulated_scaled_params(detector_params):
    r = detector_params.reflection
    table = load_pmt_optical_table()
    return TabulatedScaledReflection(
        pmt_reflectance_table=table.reflectance,
        wall_scale=r.wall_reflection_rate,
        sensor_scale=r.sensor_reflection_rate,
        pmt_reflectance_s=table.reflectance_s,
        pmt_reflectance_p=table.reflectance_p)


REFLECTION_MODELS = {
    'scalar': (scalar_reflection, _build_scalar_params),
    'scalar_mix': (scalar_mix_reflection, _build_scalar_mix_params),
    'angular': (angular_reflection, _build_angular_params),
    'tabulated_tuned': (tabulated_tuned_reflection, _build_tabulated_tuned_params),
    'tabulated_scaled': (tabulated_scaled_reflection, _build_tabulated_scaled_params),
    'tabulated_pmt_only': (tabulated_pmt_only_reflection, _build_tabulated_tuned_params),
    'tuned_wall_only': (tuned_wall_only_reflection, _build_tabulated_tuned_params),
}

# Reflection models that require a per-photon wavelength (→ wavelength_mode=True).
WAVELENGTH_REFLECTION_MODELS = frozenset({
    'angular', 'tabulated_tuned', 'tabulated_scaled',
    'tabulated_pmt_only', 'tuned_wall_only',
})

# The SK-like model treats photocathode absorption (direct PE response) and reflection
# as separate outcomes.  For this model, pmt_detection.py applies A(theta)
# per candidate PMT while photon transport independently continues R(theta).
INDEPENDENT_SENSOR_DETECTION_MODELS = frozenset({
    'tabulated_tuned', 'tabulated_scaled',
    'tabulated_pmt_only', 'tuned_wall_only',
})


def get_reflection_model(name):
    """Return ``(reflection_fn, build_refl_params)`` for a registered model name."""
    if name not in REFLECTION_MODELS:
        raise ValueError(
            f"Unknown reflection model {name!r}; available: {sorted(REFLECTION_MODELS)}")
    return REFLECTION_MODELS[name]


def sensor_normal_reflectance(reflection_fn, refl_params, lam):
    """Sensor reflection probability at normal incidence under ``reflection_fn``.

    PMT QE is quoted per photon arriving at normal incidence, so this is the reflection loss
    a quoted QE already contains. The key only drives the reflected direction, which is unused.
    """
    normal = jnp.array([0.0, 0.0, 1.0])
    refl_prob, _, _ = reflection_fn(-normal, normal, True, refl_params, lam,
                                    jax.random.PRNGKey(0))
    return refl_prob
