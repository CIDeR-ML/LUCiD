"""Event simulator factory (setup_event_simulator).

``setup_event_simulator`` assembles the pieces below, each a module-level function of explicit
inputs: the detector-parameter resolution, the optics (wavelength and QE sampling), the photon
loop shared by every source, and one forward per source (given photons, a particle, a light
source); the hits come from :mod:`lucid.simulation.hits`. What a piece needs from setup travels in a frozen record (``_Optics``,
``_Transport``, ``_Forward``, ``_Emitters``) that is bound into the jitted functions, so it
enters the trace as constants, exactly as the enclosing-scope closures it replaces did.
"""
from lucid.sources.siren_rays import (
    make_cherenkov_surrogate_fn,
    make_scintillation_surrogate_fn,
    predict_t0,
)
from lucid.siren.emitter import Emitter, load_emitter
from lucid.detector_params import DetectorParams, load_physics_config
from lucid.wavelength import DEFAULT_WAVELENGTH_NM
from lucid.wavelength.medium import medium_on_grid
from lucid.wavelength.optical_model import evaluate_optical_model
from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths, build_qe_weighted_cherenkov_sampler,
)

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional

from lucid.simulation.photon_step import (
    photon_iteration_sample, make_photon_iteration_update_factors_safe,
)
from lucid.simulation.reflection import get_reflection_model, sensor_normal_reflectance
from lucid.simulation.hits import bind_hits

if TYPE_CHECKING:
    from lucid.geometry.detector_geometry import DetectorGeometry

# ===================================================================
# Event simulator factory
# ===================================================================

@dataclass(frozen=True, eq=False)
class Simulator:
    """What ``setup_event_simulator`` returns: the simulation function and what it was built from.

    Calling it calls ``fn``. ``eq=False`` keeps identity hashing: ``DetectorParams`` holds arrays,
    so a field-wise hash would fail, and jit caches the object by identity anyway.
    """
    fn: Callable
    det_geom: 'DetectorGeometry'                        # the medium is det_geom.medium
    reflection_model: str
    default_detector_params: Optional[DetectorParams]   # None: detector_params is an argument
    emitter: Optional[Emitter]                          # track mode: the Cherenkov emitter

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# ===================================================================
# Setup records: what the jitted pieces read besides their call arguments
# ===================================================================

@dataclass(frozen=True, eq=False)
class _Optics:
    """The fixed inputs of the per-photon optical evaluation (built by :func:`_build_optics`)."""
    wavelength_mode: bool
    medium_wl: object                   # the medium on the wavelength grid; None in scalar mode
    qe_fn: Optional[Callable]           # the QECurve; None without one or in scalar mode
    qe_ref: Optional[float]             # qe_fn at DEFAULT_WAVELENGTH_NM
    wl_lo: float                        # the medium/QE window, [300, 700] nm clamped to the curve
    wl_hi: float
    sample_lo: float                    # the band LUCiD samples Cherenkov λ over (both methods)
    sample_hi: float
    wavelength_sampling: str            # 'cherenkov' or 'cherenkov_qe'
    qe_sampler: Optional[Callable]      # 'cherenkov_qe' only
    mean_qe_c: Optional[float]          # 'cherenkov_qe' only


@dataclass(frozen=True, eq=False)
class _Transport:
    """The fixed inputs of the photon loop (:func:`_common_propagation`)."""
    propagate_fn: Callable              # the detector's propagator
    photon_update_fn: Callable          # the per-photon step, sampled or expected-value
    hits_fn: Callable                   # an entry of hits.HITS, bound by bind_hits
    reflection_fn: Callable
    build_refl_params: Callable
    reflection_wavelength: float
    bounds_check: Callable              # the detector's inside test
    speed_of_light: float               # m/ns in the medium
    num_sensors: int
    K: int
    detach_directions: bool             # given photons: no gradient through the sampled directions
    is_volume: bool                     # string telescope: the volume step, no reflection


@dataclass(frozen=True, eq=False)
class _Forward:
    """What every source's forward reads besides its call arguments."""
    propagate: Callable                 # the jitted photon loop, bound to its _Transport
    optics: _Optics
    n_photons: int
    n_medium: float                     # the medium's refractive index, for the sources
    hits: str
    has_cherenkov: bool
    has_scintillation: bool
    n_cher: int                         # the photon split between the two emission processes
    n_scint: int


@dataclass(frozen=True, eq=False)
class _Emitters:
    """The track mode's emitters, built from the trained SIREN models (:func:`_load_emitters`)."""
    cherenkov_get_rays: Callable
    t0: tuple                           # the Cherenkov emitter's t0 fit (predict_t0)
    c_mm_per_ns: float                  # the t0 fit's speed of light
    scint_ray_fn: Optional[Callable]    # scintillating media only


# ===================================================================
# Configuration
# ===================================================================

def _resolve_detector_params(default_detector_params, physics_config, reflection_model):
    """``(baked DetectorParams or None, medium or None, QECurve or None)``.

    See ``default_detector_params`` in :func:`setup_event_simulator`. The medium and QE curve
    come from ``physics_config`` whenever one is given, loaded once; params baked from it must
    state what ``reflection_model`` reads.
    """
    if not isinstance(default_detector_params, (bool, DetectorParams)):
        raise TypeError(
            f"default_detector_params must be bool or DetectorParams, got {type(default_detector_params)}")
    if default_detector_params is True and physics_config is None:
        raise ValueError("physics_config is required when default_detector_params=True")
    baked_model = reflection_model if default_detector_params is True else None
    loaded_dp, medium, qe_curve = (
        load_physics_config(physics_config, reflection_model=baked_model)
        if physics_config is not None else (None, None, None))
    if default_detector_params is False:
        return None, medium, qe_curve
    if default_detector_params is True:
        return loaded_dp, medium, qe_curve
    return default_detector_params, medium, qe_curve


def _broadcast_qe_corrections(default_dp, num_sensors):
    """``default_dp`` with a scalar ``qe_corrections`` broadcast to every sensor."""
    if default_dp is None:
        return None
    qe_corr = default_dp.per_pmt.qe_corrections
    if qe_corr.ndim == 0:
        return default_dp._replace(
            per_pmt=default_dp.per_pmt._replace(
                qe_corrections=jnp.ones(num_sensors) * qe_corr))
    if len(qe_corr) != num_sensors:
        raise ValueError(
            f"qe_corrections has {len(qe_corr)} elements "
            f"but detector has {num_sensors} sensors")
    return default_dp


_SOURCES = ('particle', 'light_source', 'photons')
_TRANSPORTS = ('expected', 'sampled')


# ===================================================================
# Optics: wavelength and QE sampling
# ===================================================================

def _build_optics(medium, wavelength_mode, wavelength_sampling, qe_curve,
                  cherenkov_emission_band, given_photons):
    """The :class:`_Optics` for these options, validated (see :func:`setup_event_simulator`).

    ``cherenkov_emission_band`` is the band Cherenkov λ is sampled over (track mode: the emitter's
    unless the caller overrides it); None samples over the medium/QE window (calibration).
    """
    # ---- Wavelength-dependent medium (when wavelength_mode=True) -----
    # Sampling/grid bounds: clamp the QE curve's knot range into the
    # [300, 700] nm floor/ceiling that LUCiD's water medium covers.
    # Without a QE curve, default to the full [300, 700] nm range.
    if wavelength_mode:
        if qe_curve is not None:
            qe_lo, qe_hi = qe_curve.bounds
            wl_lo = max(300.0, qe_lo)
            wl_hi = min(700.0, qe_hi)
            if wl_hi <= wl_lo:
                raise ValueError(
                    f"QE curve range [{qe_lo:.1f}, {qe_hi:.1f}] nm does not "
                    f"overlap the water medium range [300, 700] nm. Check "
                    "the qe_curve in physics_config.")
        else:
            wl_lo, wl_hi = 300.0, 700.0
        wl_grid = jnp.linspace(wl_lo, wl_hi, 200)
        medium_wl = medium_on_grid(medium, wl_grid)
        qe_fn = qe_curve
    else:
        wl_lo, wl_hi = 300.0, 700.0
        medium_wl = None
        qe_fn = None

    # ---- Cherenkov λ-SAMPLING band ----------------------------------------
    # The medium grid + QE machinery stay clamped to [wl_lo, wl_hi] (the
    # [300,700]∩QE window the water medium covers). But the emitter's N_photons(E)
    # counts photons over its generator's EMISSION band (PhotonSim: [274.91, 673.83] nm),
    # which extends past that window. Sampling only over [wl_lo, wl_hi] would treat
    # every photon as detectable and over-count the charge by the QE-dead fringe
    # (+19% with the SK QE). Over the emission band, QE(λ)=0 (outside the knots) and the
    # medium interp-clamp make the fringe photons inert, with no scalar norm.
    if cherenkov_emission_band is not None:
        sample_lo = float(cherenkov_emission_band[0])
        sample_hi = float(cherenkov_emission_band[1])
    else:
        sample_lo, sample_hi = wl_lo, wl_hi

    # ---- Wavelength sampling mode ----------------------------------------
    # 'cherenkov'   : Method A — λ ~ 1/λ², per-photon QE weight = qe_fn(λ).
    # 'cherenkov_qe': Method B — λ ~ QE(λ)/λ² (importance sampling),
    #                 per-photon QE weight collapses to the scalar <QE>_C.
    if wavelength_sampling not in ('cherenkov', 'cherenkov_qe'):
        raise ValueError(
            f"wavelength_sampling must be 'cherenkov' or 'cherenkov_qe'; "
            f"got {wavelength_sampling!r}")
    if wavelength_sampling == 'cherenkov_qe':
        if not wavelength_mode:
            raise ValueError(
                "wavelength_sampling='cherenkov_qe' requires wavelength_mode=True.")
        if qe_fn is None:
            raise ValueError(
                "wavelength_sampling='cherenkov_qe' requires a QE curve — set "
                "qe_curve in the physics_config.")
        if given_photons:
            raise ValueError(
                "wavelength_sampling='cherenkov_qe' is incompatible with "
                "source='photons': PhotonSim ROOT photons carry their own "
                "wavelengths; LUCiD does not sample them.")
        qe_sampler, mean_qe_c = build_qe_weighted_cherenkov_sampler(
            qe_fn, sample_lo, sample_hi)
    else:
        qe_sampler = None
        mean_qe_c = None

    # response.qe is the QE at the scalar reference wavelength: the value scalar mode
    # applies to every photon, and the one _project_missing_scalars reads off the curve
    # when a config gives no scalar. In wavelength mode the curve supplies only the
    # wavelength dependence around it, so _get_optical_arrays divides the QE weights by
    # qe_fn(λ_ref) and the call sites multiply by response.qe. A photon at λ is then
    # detected with response.qe * qe_fn(λ) / qe_fn(λ_ref): exactly qe_fn(λ) for the
    # projected scalar, and the two modes agree at λ_ref for any scalar.
    qe_ref = None
    if qe_fn is not None:
        qe_ref = float(qe_fn(DEFAULT_WAVELENGTH_NM))
        if not qe_ref > 0.0:
            raise ValueError(
                f"The QE curve is zero at the scalar reference wavelength "
                f"({DEFAULT_WAVELENGTH_NM} nm), so response.qe cannot be anchored to it.")

    return _Optics(wavelength_mode=wavelength_mode, medium_wl=medium_wl, qe_fn=qe_fn,
                   qe_ref=qe_ref, wl_lo=wl_lo, wl_hi=wl_hi, sample_lo=sample_lo,
                   sample_hi=sample_hi, wavelength_sampling=wavelength_sampling,
                   qe_sampler=qe_sampler, mean_qe_c=mean_qe_c)


def _check_peak_qe(default_dp, reflection_fn, build_refl_params, reflection_wavelength,
                   reflection_model, optics):
    """Raise unless the baked-in peak QE is at most 1 - R0.

    A deposited photon converts at QE / (1 - R0) (see :func:`_common_propagation`), a probability
    only while QE <= 1 - R0: QE counts photons arriving at the PMT, so it cannot exceed the
    fraction that is not reflected. Checkable at setup when the parameters are baked in.
    """
    r0 = float(sensor_normal_reflectance(
        reflection_fn, build_refl_params(default_dp), jnp.asarray(reflection_wavelength)))
    qe_peak = (float(default_dp.response.qe)
               * float(jnp.max(default_dp.per_pmt.qe_corrections)))
    if optics.wavelength_mode and optics.qe_fn is not None:
        qe_peak *= (optics.qe_fn.peak / optics.qe_ref
                    * float(jnp.max(default_dp.response.qe_dev)))
    if not qe_peak <= 1.0 - r0:
        raise ValueError(
            f"Peak QE {qe_peak:.4f} exceeds 1 - R0 = {1.0 - r0:.4f}, R0 being the "
            f"{reflection_model!r} sensor reflectance at normal incidence: QE counts "
            "photons arriving at the PMT, so it cannot exceed the fraction not reflected.")


def _get_optical_arrays(optics, n, detector_params, key, wavelengths=None):
    """Compute per-photon (n,) scatter/absorption arrays and QE weights.

    When wavelength_mode=True: uses medium coefficients at given wavelengths.
    When wavelength_mode=False: broadcasts DetectorParams scalars.

    Returns (scatter_lengths, absorption_lengths, qe_weights, key).
    qe_weights is (n,) relative to the QE at λ_ref (see _Optics.qe_ref), or None.
    """
    if optics.wavelength_mode and optics.medium_wl is None:
        raise RuntimeError(
            "wavelength_mode=True but no medium model was loaded. "
            "Provide a physics_config with 'medium_model' or set wavelength_mode=False.")
    if not optics.wavelength_mode:
        # Monochromatic / scalar mode: the pure optical model broadcasts the
        # DetectorParams scalars (wavelengths=None path).
        oa = evaluate_optical_model(detector_params, None, optics.medium_wl, n)
        return (oa.scatter_len, oa.mie_len, oa.abs_len, oa.qe, key)

    # --- λ sampling (a SOURCE concern; the optical model only evaluates) ---
    # Normalize wavelength input to per-photon array:
    #   None   → sample Cherenkov spectrum (Method A or B depending on
    #            wavelength_sampling)
    #   scalar → broadcast to (n,)
    #   (n,)   → use as-is
    sampled_via_qe_importance = False
    if wavelengths is None:
        key, wl_key = jax.random.split(key)
        if optics.wavelength_sampling == 'cherenkov_qe':
            wavelengths = optics.qe_sampler(wl_key, n)
            sampled_via_qe_importance = True
        else:
            wavelengths = sample_cherenkov_wavelengths(
                wl_key, n, lambda_min=optics.sample_lo, lambda_max=optics.sample_hi)
    else:
        wavelengths = jnp.asarray(wavelengths)
        if wavelengths.ndim == 0:
            wavelengths = jnp.full(n, wavelengths)

    # --- per-photon optical evaluation (pure seam) ---
    # QE-weight convention:
    #   • Method B sampled here  → the λ-dependence of QE is already in the
    #     sampling distribution, so the per-photon weight collapses to the
    #     scalar <QE>_C (override below; qe_fn not threaded here).
    #   • Otherwise (explicit wavelengths, PhotonSim data, or Method A):
    #     the per-photon weight must include qe_fn(λ).
    oa = evaluate_optical_model(
        detector_params, wavelengths, optics.medium_wl, n,
        qe_fn=None if sampled_via_qe_importance else optics.qe_fn)
    qe_weights = jnp.full(n, optics.mean_qe_c) if sampled_via_qe_importance else oa.qe
    if qe_weights is not None:
        qe_weights = qe_weights / optics.qe_ref
    return oa.scatter_len, oa.mie_len, oa.abs_len, qe_weights, key


def _qe_per_photon(qe_weights, detector_params, n):
    """Per-photon QE: response.qe x the curve relative to λ_ref (passed to the hits, not baked
    into weights); response.qe alone without a curve."""
    if qe_weights is not None:
        return qe_weights * detector_params.response.qe
    return jnp.full(n, detector_params.response.qe)


# ===================================================================
# Core propagation (shared by all modes)
# ===================================================================

def _gradient_cut(t, positions, directions):
    """The photon state carried to the next scan step: positions always carry gradient, directions
    carry it except for given photons (detached on every step)."""
    if t.detach_directions:
        return positions, jax.lax.stop_gradient(directions)
    return positions, directions


def _common_propagation(t, positions, directions, intensities, times,
                 scatter_lengths, mie_scatter_lengths, absorption_lengths,
                 qe_per_photon, n_rays, detector_params, key, segment_idx=None):
    """Core photon propagation loop; ``t`` is its :class:`_Transport`.

    Jitted per simulator with ``t`` bound and ``n_rays`` static (see
    :func:`setup_event_simulator`).

    Parameters
    ----------
    scatter_lengths : jnp.ndarray
        Per-photon scattering lengths, shape (n_rays,).
    mie_scatter_lengths : jnp.ndarray
        Per-photon Mie scattering lengths, shape (n_rays,).
    absorption_lengths : jnp.ndarray
        Per-photon absorption lengths, shape (n_rays,).
    qe_per_photon : jnp.ndarray
        Per-photon quantum efficiency for light arriving at a sensor, its reflection
        included, shape (n_rays,).
    """

    # Packed reflection params for the pluggable reflection model (default scalar_mix →
    # ScalarMixReflection(rates + fspec)). build_refl_params is chosen at setup.
    refl_params = t.build_refl_params(detector_params)
    # Wavelength fed to the reflection model. Scalar reflection ignores it; the angular
    # (Fresnel) model uses it for the cathode/glass dispersion. A scalar reflection
    # wavelength is exact for monochromatic-laser calibration (the validated case); a
    # per-photon λ array can be threaded here later for broadband sources.
    refl_lam = jnp.asarray(t.reflection_wavelength)
    # QE is per photon arriving at a PMT, its reflection included, as quoted at normal
    # incidence. The photon step reflects first and deposits the rest, so a deposited
    # photon converts at QE / (1 - R0), R0 the model's sensor reflectance at normal
    # incidence. String telescopes have no reflection.
    if not t.is_volume:
        qe_per_photon = qe_per_photon / (
            1.0 - sensor_normal_reflectance(t.reflection_fn, refl_params, refl_lam))
    qe_corrections = detector_params.per_pmt.qe_corrections
    g = detector_params.scattering.g

    from lucid.simulation.types import PhotonState

    # The surface detectors are closed (the ID blacksheet), so light emitted outside never
    # reaches an ID sensor; a vertex shift, or a track's range, can put emission there.
    # String telescopes are open media.
    initial_survival = (jnp.ones(n_rays) if t.is_volume
                        else t.bounds_check(positions).astype(jnp.float32))

    def propagation_step(carry, _):
        state = carry
        key, prop_key = jax.random.split(state.key)

        prop_results = t.propagate_fn(state.positions, state.directions)
        depositions = prop_results['sensor_weights']
        sensor_indices = prop_results['sensor_indices']
        inside_sensor = prop_results['inside_sensor']

        if t.is_volume:
            # ── Volume model (string telescope): per-DOM survival, NO reflection ──
            # Each candidate DOM gets an independent detection weight from its distance
            # along the ray; the photon always scatters in the open medium (no wall).
            # Two-channel Rayleigh+Mie scatter (forward-peaked Mie needed for ice), as a
            # DiCE-forward citizen mirroring the water step: the optical (λ_R/λ_M/g)
            # charge-gradient rides the per-step score (logp_increment) folded via dice_dep,
            # NOT the high-variance pathwise route through the discrete DOM-candidate
            # selection. Track params flow pathwise; λ_abs flows pathwise (deterministic).
            from lucid.simulation.photon_step_volume import photon_step_volume
            sensor_distances = prop_results['sensor_distances']     # (n_cand, n_rays, 1)
            seg_lengths = jnp.maximum(prop_results['envelope_exit_t'], 1.0)   # (n_rays,)
            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)
            (new_positions, new_directions, new_times,
             per_dom_charges, continuing_factors, logp_increments,
             step_indirect) = jax.vmap(
                photon_step_volume,
                in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0, 0, None, None)
            )(state.positions, state.directions, state.times,
              sensor_distances.squeeze(-1), depositions,
              scatter_lengths, mie_scatter_lengths, absorption_lengths, seg_lengths,
              rng_keys, t.speed_of_light, g)

            inside_detector = t.bounds_check(new_positions)
            safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)
            new_survival = state.survival * safe_continuing

            # DiCE magic box from the PRE-step log_p: forward value = 1 (charge unchanged),
            # reverse-mode injects the accumulated optical score so the deposit carries the
            # gradient of the probability the photon's trajectory reached this step.
            dice_dep = jnp.exp(state.log_p - jax.lax.stop_gradient(state.log_p))   # (n_rays,)
            physical_intensities = intensities * state.survival * dice_dep
            updated_weights = per_dom_charges.T * physical_intensities[None, :]
            total_times = sensor_distances / t.speed_of_light + state.times[:, None]

            next_pos, next_dir = _gradient_cut(t, new_positions, new_directions)
            new_log_p = state.log_p + logp_increments   # accumulate optical score for FUTURE deposits
            new_state = PhotonState(
                positions=next_pos, directions=next_dir, times=new_times,
                survival=new_survival, key=key, log_p=new_log_p)
            # Same 4-tuple shape as the surface branch: the scan unpacks one
            # output signature for both models.
            outputs = (updated_weights, sensor_indices,
                       total_times.squeeze(-1), step_indirect)
            return new_state, outputs

        # ── Surface model (cylinder/sphere/box) — UNCHANGED, byte-identical ──
        hit_times_meters = prop_results['times']
        hit_positions = prop_results['positions']
        normals = prop_results['normals']

        normals = jax.lax.stop_gradient(normals)   # detach the reflection normal (Igehy 1/r curvature term compounds across bounces)

        hit_sensor = jnp.max(inside_sensor, axis=0)
        # SAFE norm: eps INSIDE the sqrt so the gradient is finite when a photon sits on a surface
        # (hit==pos -> |Δ|=0 -> jnp.linalg.norm has a 0/0 NaN gradient). This is the 2nd-order-AD NaN.
        surface_distances = jnp.sqrt(jnp.sum((hit_positions - state.positions)**2, axis=1) + 1e-12) - 1e-6

        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, n_rays)

        # vmap: 14 args — per-photon scatter/absorption, scalar g, packed refl_params
        # pytree (broadcast, in_axes None), per-photon hit_sensor + rng, scalar lam
        # (a wavelength placeholder for scalar reflection; per-photon λ is threaded when
        # a wavelength-dependent reflection model is used). Step returns a 7-tuple; the
        # 7th is the per-photon DiCE score increment (lf+la+lr for the differentiable
        # step, 0.0 for the sampling step). PRE-step log_p drives the implicit deposit.
        (new_positions, new_directions, new_times,
         detect_probs, reflection_attenuations,
         continuing_factors, logp_increments, step_indirect) = jax.vmap(
            t.photon_update_fn,
            in_axes=(0, 0, 0, 0, 0,
                     0, 0, None, None, 0,
                     0, None, 0, None)
        )(state.positions, state.directions, state.times,
          surface_distances, normals,
          scatter_lengths, mie_scatter_lengths, g, refl_params,
          absorption_lengths,
          hit_sensor, refl_lam, rng_keys, t.speed_of_light)

        inside_detector = t.bounds_check(new_positions)
        safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)

        new_survival = state.survival * safe_continuing

        physical_intensities = intensities * state.survival
        # DiCE magic box from the PRE-step log_p: forward value = 1 (charge unchanged),
        # gradient = the accumulated optical score for deposits at this step. The sampling
        # path keeps log_p=0 → dice_dep=1 (no effect), so the data oracle is untouched.
        dice_dep = jnp.exp(state.log_p - jax.lax.stop_gradient(state.log_p))   # (n_rays,)
        detected_factors = detect_probs * reflection_attenuations * dice_dep
        updated_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
        times_ns = hit_times_meters / t.speed_of_light
        total_times = times_ns + state.times[:, None]

        iter_weights = updated_weights
        iter_indices = sensor_indices
        iter_times = total_times.squeeze(-1)

        next_pos, next_dir = _gradient_cut(t, new_positions, new_directions)

        new_log_p = state.log_p + logp_increments   # accumulate score for FUTURE deposits

        new_state = PhotonState(
            positions=next_pos,
            directions=next_dir,
            times=new_times,
            survival=new_survival,
            key=key,
            log_p=new_log_p,
        )
        # step_indirect is a scan OUTPUT, not part of the carry: keeping it
        # out of PhotonState leaves the carry pytree, its remat and the
        # gradient path exactly as they were. A cumulative OR along the
        # iteration axis afterwards gives "had already indirect", which is
        # what the scattering table's direct/indirect split needs.
        outputs = (iter_weights, iter_indices, iter_times, step_indirect)
        return new_state, outputs

    init_state = PhotonState(
        positions=positions,
        directions=directions,
        times=times,
        survival=initial_survival,
        key=key,
        log_p=jnp.zeros(n_rays),
    )
    propagation_step_remat = jax.remat(propagation_step)

    _, (all_weights, all_indices, all_times, all_indirect) = jax.lax.scan(
        propagation_step_remat, init_state, jnp.arange(t.K))

    flat_weights = all_weights.reshape(-1)
    flat_indices = all_indices.reshape(-1)
    flat_times = all_times.reshape(-1)

    # Per-deposit "this photon had already left the straight line". A deposit
    # made at step k belongs to a photon that indirect during steps 0..k-1,
    # so this is the EXCLUSIVE prefix OR along the iteration axis -- inclusive
    # would wrongly tag the step on which a photon both scattered and was
    # detected. Broadcast over the candidate axis to match all_weights'
    # (K, max_candidates, n_rays) layout before the same C-order reshape.
    prior_indirect = jnp.concatenate(
        [jnp.zeros((1,) + all_indirect.shape[1:], dtype=bool),
         jnp.cumsum(all_indirect, axis=0)[:-1] > 0], axis=0)
    flat_indirect = jnp.broadcast_to(
        prior_indirect[:, None, :], all_weights.shape).reshape(-1)

    # Tile per-photon QE to match flat shape.
    # all_weights shape: (K, max_candidates_per_ray, n_rays), C-order reshape
    # → photon index is i % n_rays
    photon_idx = jnp.arange(flat_weights.shape[0]) % n_rays
    flat_qe = qe_per_photon[photon_idx]

    key, qe_key = jax.random.split(key)
    # Per-photon segment id (segment_truth hits), broadcast to flat shape via the
    # same i % n_rays trick as flat_qe. None when the photons carry none → byte-identical.
    flat_segment_idx = (segment_idx[photon_idx] if segment_idx is not None else None)
    return t.hits_fn(
        flat_weights, flat_indices, flat_times, t.num_sensors, qe_key, flat_qe, qe_corrections,
        detector_params.response, flat_segment_idx, flat_indirect)


# ===================================================================
# Source-specific forwards
# ===================================================================
# Each takes its _Forward record first and is jitted per simulator with the record bound.

def _simulation_with_data_impl(f, detector_params, key, photon_data):
    """Given photons (``source='photons'``): from ROOT/PhotonSim files, their times as recorded."""
    photon_origins = photon_data['photon_origins']          # metres, used as given
    photon_directions = photon_data['photon_directions']
    photon_times = photon_data['photon_times']

    n_rays = photon_origins.shape[0]
    mask = jnp.arange(n_rays) < photon_data['N']
    photon_intensities = 1.0 * mask.astype(jnp.float32)

    # Per-photon optical properties — use PhotonSim wavelengths if available
    data_wavelengths = photon_data.get('wavelengths', None)
    scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
        f.optics, n_rays, detector_params, key, wavelengths=data_wavelengths)
    qe_per_photon = _qe_per_photon(qe_weights, detector_params, n_rays)

    # Per-photon segment id (int32, padded with -1): required by the segment_truth production
    # hits, whose output is grouped by it; optional otherwise (None for recon data from
    # pad_photon_data), and the other hits ignore it.
    if f.hits == 'segment_truth':
        segment_idx = photon_data['photon_segment_index']
    else:
        segment_idx = photon_data.get('photon_segment_index', None)

    return f.propagate(
        photon_origins, photon_directions, photon_intensities, photon_times,
        scatter_lengths, mie_scatter_lengths, absorption_lengths,
        qe_per_photon,
        n_rays, detector_params, key, segment_idx=segment_idx)


def _simulation_without_data_impl(f, e, particle_params, detector_params, key, model_params,
                                  dedx_params):
    """SIREN mode: particle_params is ParticleParams (its ``t0`` starts every photon's clock);
    ``e`` is the :class:`_Emitters`.

    ``model_params`` and ``dedx_params`` are the Cherenkov and dE/dx SIREN params (``dedx_params``
    None unless the medium scintillates), jit arguments in every binding. When the medium
    scintillates the per-photon arrays are the Cherenkov+Scintillation concatenation.
    """
    energy = particle_params.energy
    track_origin = particle_params.position
    track_direction = particle_params.direction  # property

    def _pt0(distance_m):
        return jax.vmap(predict_t0, in_axes=(0, None, None, None))(
            distance_m, energy, e.t0, e.c_mm_per_ns)
    optics = f.optics

    if f.has_scintillation:
        # ---- Cherenkov + Scintillation (wbls) / scintillation-only path ----
        n_total = f.n_cher + f.n_scint
        key, cher_key, scint_key, opt_key, wl_key = jax.random.split(key, 5)
        parts_dirs, parts_origins, parts_intens, parts_times, parts_wls = [], [], [], [], []

        if f.has_cherenkov:
            ch_dirs, ch_origins, ch_intens = e.cherenkov_get_rays(
                track_origin, track_direction, energy, f.n_cher, model_params, cher_key)
            ch_t0 = jax.lax.stop_gradient(_pt0(jnp.linalg.norm(ch_origins - track_origin, axis=1)))
            if optics.wavelength_sampling == 'cherenkov_qe':
                ch_wls = optics.qe_sampler(wl_key, f.n_cher)
            else:
                ch_wls = sample_cherenkov_wavelengths(
                    wl_key, f.n_cher, lambda_min=optics.sample_lo, lambda_max=optics.sample_hi)
            parts_dirs.append(ch_dirs); parts_origins.append(ch_origins)
            parts_intens.append(ch_intens); parts_times.append(ch_t0); parts_wls.append(ch_wls)

        # Scintillation surrogate: isotropic rays, Chou-quenched weights (S/kB/C),
        # biexp emission delay (tau_rise/tau_fall) on top of the t0 baseline, Moyal λ.
        s = detector_params.scintillation
        sc_dirs, sc_origins, sc_intens, sc_tdelay, sc_wls = e.scint_ray_fn(
            track_origin, track_direction, energy, f.n_scint,
            dedx_params, scint_key,
            s.S, s.kB, s.C, s.tau_rise, s.tau_fall)
        sc_t0 = jax.lax.stop_gradient(_pt0(jnp.linalg.norm(sc_origins - track_origin, axis=1)))
        parts_dirs.append(sc_dirs); parts_origins.append(sc_origins)
        parts_intens.append(sc_intens); parts_times.append(sc_t0 + sc_tdelay); parts_wls.append(sc_wls)

        photon_directions = jnp.concatenate(parts_dirs)
        photon_origins = jnp.concatenate(parts_origins)
        photon_intensities = jnp.concatenate(parts_intens)
        photon_times = particle_params.t0 + jnp.concatenate(parts_times)
        wavelengths = jnp.concatenate(parts_wls)

        scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            optics, n_total, detector_params, opt_key, wavelengths=wavelengths)
        qe_per_photon = _qe_per_photon(qe_weights, detector_params, n_total)
        return f.propagate(
            photon_origins, photon_directions, photon_intensities, photon_times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_total, detector_params, key)

    # ---- Cherenkov-only path (byte-identical to pre-scint forward) ----
    n_photons = f.n_photons
    key, ray_key, opt_key = jax.random.split(key, 3)
    # New refactor-v2 Cherenkov emitter: intensities already carry the absolute count
    # (pmf × the emitter's n_photons(E)); no separate normalization or mean_topk amplitude.
    photon_directions, photon_origins, photon_intensities = e.cherenkov_get_rays(
        track_origin, track_direction, energy, n_photons, model_params, ray_key)

    distances_to_vertex = jnp.linalg.norm(photon_origins - track_origin, axis=1)
    # Emission-time baseline t0(distance_to_vertex, energy), DETACHED: the TIME term does not carry
    # ENERGY/VERTEX gradient through the emission-time model (those flow via geometry/charge).
    t0 = jax.lax.stop_gradient(_pt0(distances_to_vertex))

    # Per-photon optical properties (Cherenkov spectrum when wavelength_mode)
    scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
        optics, n_photons, detector_params, opt_key)
    qe_per_photon = _qe_per_photon(qe_weights, detector_params, n_photons)

    return f.propagate(
        photon_origins, photon_directions, photon_intensities, particle_params.t0 + t0,
        scatter_lengths, mie_scatter_lengths, absorption_lengths,
        qe_per_photon,
        n_photons, detector_params, key)


def _simulation_sensor_calibration_impl(f, source, detector_params, key):
    """Light source (``source='light_source'``): a callable (IsotropicSource, LaserSource, ...)."""
    n_photons = f.n_photons
    key, source_key, opt_key = jax.random.split(key, 3)
    photon_directions, photon_origins, photon_intensities = source(n_photons, source_key, f.n_medium)
    photon_times = jnp.zeros((n_photons,))

    # Per-photon optical properties
    # Source wavelength can be None (→ Cherenkov), scalar, or (Nphot,) array;
    # _get_optical_arrays normalizes the shape.
    wavelengths = getattr(source, 'wavelength', None)
    scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
        f.optics, n_photons, detector_params, opt_key, wavelengths=wavelengths)
    qe_per_photon = _qe_per_photon(qe_weights, detector_params, n_photons)

    return f.propagate(
        photon_origins, photon_directions, photon_intensities, photon_times,
        scatter_lengths, mie_scatter_lengths, absorption_lengths,
        qe_per_photon,
        n_photons, detector_params, key)


# ---- Track-mode emission-process dispatch (refactor-v2 wbls/scint merge) ----
# The medium's `emission_processes` tuple decides which surrogates run. The
# Cherenkov/Scintillation photon-count split (when both run) is a static,
# material-level knob (`medium.cherenkov_fraction`). Water has only
# ("cherenkov",) → `has_scintillation` is False → the scintillation branch is
# never even traced, so the Cherenkov-only path stays byte-identical.

def _emission_split(medium, n_photons, wavelength_mode):
    """``(has_cherenkov, has_scintillation, n_cher, n_scint)`` for the medium's emission."""
    has_cherenkov = "cherenkov" in medium.emission_processes
    has_scintillation = "scintillation" in medium.emission_processes
    if has_scintillation and not wavelength_mode:
        raise ValueError(
            "Scintillation emission requires wavelength_mode=True (the Moyal "
            "wavelength sampler feeds the per-photon spectral weight).")
    if has_cherenkov and has_scintillation:
        n_cher = int(round(n_photons * float(medium.cherenkov_fraction)))
        n_scint = n_photons - n_cher
    elif has_cherenkov:
        n_cher, n_scint = n_photons, 0
    elif has_scintillation:
        n_cher, n_scint = 0, n_photons
    else:
        raise ValueError(
            f"medium {medium.material!r} has no emission_processes enabled "
            f"({medium.emission_processes!r}).")
    return has_cherenkov, has_scintillation, n_cher, n_scint


def _load_emitters(particle, medium, has_scintillation):
    """``(_Emitters, Cherenkov Emitter, dE/dx Emitter or None)`` for the track mode.

    The emitters, once, from the set the material JSON names (``"siren"``; wbls and ice run
    water's). Each ray function closes over its emitter's net, domain and curves. The scintillation
    surrogate is built only when the medium scintillates (wbls).
    """
    siren_set = medium.model['siren']
    cherenkov = load_emitter(particle, siren_set, 'cherenkov')
    if cherenkov.t0 is None:
        raise ValueError(f"the {siren_set} {particle} Cherenkov emitter has no t0 fit (emission time); "
                         f"install it with lucid-siren-promote --t0")
    cherenkov_get_rays = make_cherenkov_surrogate_fn(cherenkov)

    scint_ray_fn = None
    dedx = None
    if has_scintillation:
        # Moyal sampling params are static (they bake the inverse-CDF lookup at
        # factory time), from the medium's scintillation.spectrum.
        dedx = load_emitter(particle, siren_set, 'dedx')
        scint_ray_fn = make_scintillation_surrogate_fn(
            dedx,
            medium.scintillation_lambda_min,
            medium.scintillation_lambda_max,
            moyal_loc=medium.moyal_loc,
            moyal_scale=medium.moyal_scale,
        )
    return _Emitters(cherenkov_get_rays=cherenkov_get_rays, t0=cherenkov.t0,
                     c_mm_per_ns=cherenkov.c_mm_per_ns, scint_ray_fn=scint_ray_fn), cherenkov, dedx


def _bake_detector_params(impl, default_dp, slot):
    """``impl`` with ``default_dp`` as its positional argument ``slot`` (its ``detector_params``):
    the call signatures without it.

    ``default_dp`` reaches the jitted ``impl`` as an argument, never as a closed-over constant: XLA
    folds constants, which moves rays between grid bins, so a baked and a passed DetectorParams
    would give different events for the same key.
    """
    def baked(*args, **kwargs):
        return impl(*args[:slot], default_dp, *args[slot:], **kwargs)
    return baked


def setup_event_simulator(
        json_filename,
        n_photons=1_000_000,
        temperature=0.2,
        K=7,
        *,
        source,
        transport=None,
        detection=None,
        hits=None,
        hits_config=None,
        particle='muon',
        charge_resolution=None,
        physics_config=None,
        default_detector_params=False,
        wavelength_mode=True,
        wavelength_sampling='cherenkov',
        reflection_model='scalar_mix',
        reflection_wavelength=400.0,
        cherenkov_emission_band=None,
        emission_processes=None,
        **grid_params):
    """
    Set up and return an event simulator using DetectorParams / ParticleParams.

    Parameters
    ----------
    json_filename : str
        Path to detector geometry JSON.
    n_photons : int
        Number of photons per event.
    temperature : float
        Soft-assignment temperature for propagation.
    K : int
        Maximum scattering iterations.
    source : {'particle', 'light_source', 'photons'}
        Where the photons come from, which sets the call signature (see Returns):

        - ``'particle'`` -- a track, its photons drawn from the particle's SIREN emitter.
        - ``'light_source'`` -- a source object passed at call time (laser, isotropic, cascade,
          shotgun).
        - ``'photons'`` -- photons given at call time (e.g. PhotonSim), transported as given.
    transport : {'expected', 'sampled'} or None
        The propagation: ``'expected'`` carries each photon's expected weight through every
        scatter, absorption and reflection (differentiable, no shot noise); ``'sampled'`` draws
        each branch (Monte Carlo, one realisation). None (default): ``'sampled'`` for
        ``source='photons'``, which supports only that, ``'expected'`` otherwise.
    detection : {'expected', 'sampled'} or None
        How QE, the TTS and the charge resolution enter the hits: as expectations or as draws.
        None (default): ``'sampled'`` for ``source='photons'``, ``'expected'`` otherwise.
    hits : str or None
        What the sensors report, with ``detection`` an entry of
        :data:`lucid.simulation.hits.HITS` (an unsupported pair raises, naming the supported
        ones). None (default): ``'photon_list'`` for ``source='particle'``, ``'first_arrival'``
        otherwise.

        - ``'first_arrival'`` -- per-sensor charge and first-arrival time (expected: the mean of
          the predicted first-arrival distribution; sampled: Bernoulli QE, hard first arrival
          with the TTS, ``charge_resolution`` smearing).
        - ``'photon_list'`` -- per-deposit log-weights, times and sensors, and per-sensor charge
          (expected only; the reconstruction's likelihood input).
        - ``'segment_truth'`` -- the sampled per-sensor hits plus the per-deposit arrays and
          segment indices (sampled only; dataset production).
        - ``'waveform'`` -- time-binned per-sensor waveforms.
        - ``'photon_hits'`` -- per-photon first-detection records (sampled only).
    hits_config : dict or None
        The setup-time options of the hits function (e.g. ``window_ns`` for ``'waveform'``); a
        key it does not take raises, naming the ones it does.
    particle : str
        Particle type (e.g., 'muon', 'electron'). Used to load SIREN model,
        t0 parameters, and photon normalization from config.
    charge_resolution : str or None
        Per-sensor charge-resolution model of the sampled hits: None (raw Poisson counts),
        "Abe_2013" (SK fractional Gaussian), or "Bellamy_94" (per-photoelectron SPE
        spectrum, WCSim SK PMT). Timing resolution comes solely from the per-photon
        TTS (``detector_params.response.tts``).
    physics_config : str or None
        Path to physics config JSON (e.g. ``SK_physics_config.json``).
        Required when ``default_detector_params=True``.
    default_detector_params : bool or DetectorParams
        Controls whether ``detector_params`` is baked into the returned function:

        - ``False`` (default) -- returned function **requires** ``detector_params``
          as an explicit argument.
        - ``True`` -- loads ``DetectorParams`` from *physics_config* at setup time
          and bakes it into the returned function.
        - ``DetectorParams`` instance -- bakes that instance directly (no file load).

        Baked or passed, the params reach the jitted forward as an argument, so both give the
        same event for the same key.
    wavelength_sampling : str
        How LUCiD samples λ when it samples (a particle, a light source with
        ``source.wavelength=None``). Ignored silently when the caller supplies
        wavelengths explicitly (scalar/array source, given photons).

        - ``'cherenkov'`` (default) — λ ~ 1/λ², per-photon weight = ``qe_fn(λ)``.
        - ``'cherenkov_qe'`` — λ ~ QE(λ)/λ² via inverse CDF, per-photon weight
          collapses to the scalar ``<QE>_C``. Variance-optimal in expected-value
          mode. In Bernoulli / MC mode the expected waveform matches
          ``'cherenkov'`` but per-shot Binomial fluctuations are suppressed, so
          use only when the output is interpreted as a density estimate.
          Rejected at setup when ``wavelength_mode=False``, no QE curve is
          loaded, or ``source='photons'``.
    reflection_model : str
        Reflection model: ``'scalar_mix'`` (DEFAULT — the scalar wall/sensor
        rates plus a specular/diffuse direction mixture via
        ``wall_fspec/sensor_fspec``, DiCE-scored, no per-photon λ needed),
        ``'scalar'`` (the legacy angle/λ-independent rates with hard direction —
        walls diffuse, sensors specular; byte-identical to the pre-mixture step), or
        ``'angular'`` (Schlick blacksheet wall + multilayer-Fresnel cathode
        sensor, with a specular/diffuse direction mixture). The angular model
        reads the ``DetectorParams.reflection`` fields ``wall_R0/wall_p/
        wall_fspec`` and ``cathode_nr/cathode_nk/sensor_fspec``; params baked from a
        physics config state the first four in its ``"angular_reflection"`` block.
    reflection_wavelength : float
        Wavelength (nm) fed to the reflection model's dispersion (cathode/glass
        Fresnel). Exact for monochromatic-laser calibration; ignored by the
        scalar model. Default 400 nm.
    cherenkov_emission_band : (float, float) or None
        The band (nm) LUCiD samples Cherenkov λ over in wavelength mode. None (default): in track
        mode the Cherenkov emitter's ``emission_band_nm``, the band its N_photons(E) counts (the
        light outside the QE curve then counts nothing); in calibration with
        ``source.wavelength=None``, the medium/QE window. Pass a band only to sample another one.
    emission_processes : tuple of str or None
        The light a particle emits: a non-empty subset of ``('cherenkov', 'scintillation')``.
        None (default) runs the material's own processes (``materials/<material>.json``); a tuple
        overrides them, e.g. to predict one process alone. With both, the photon budget splits by
        the material's ``cherenkov_fraction``.

    Returns
    -------
    Simulator
        Callable. When ``default_detector_params`` is ``False``:

        - ``source='light_source'`` ``(source, detector_params, key)``
        - ``source='particle'``     ``(particle_params, detector_params, key)``
        - ``source='photons'``      ``(detector_params, key, photon_data)``

        When ``default_detector_params`` is truthy (``True`` or a ``DetectorParams``) the
        ``detector_params`` argument is dropped: ``(source, key)``, ``(particle_params, key)``,
        ``(key, photon_data)``.

        The call returns the hits function's NamedTuple: ``FirstArrival(charge, time)``,
        ``PhotonList(log_w, times, indices, charge)``, ``SegmentTruth`` (per-sensor charge and
        true and smeared times, then the per-deposit weights, true and smeared times, sensors and
        segments), ``Waveform(waveform, n_dropped, n_detected)`` or
        ``PhotonHits(detected, sensor_id, hit_time, indirect)``.

        ``.det_geom``, ``.reflection_model`` and ``.default_detector_params`` (None when not
        baked in) record what it was built from; for a particle ``.emitter`` is the Cherenkov
        :class:`~lucid.siren.emitter.Emitter` (``.emitter.n_photons`` is its N_photons(E)), None
        otherwise.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {_SOURCES}, got {source!r}")
    given_photons = source == 'photons'
    if transport is None:
        transport = 'sampled' if given_photons else 'expected'
    if transport not in _TRANSPORTS:
        raise ValueError(f"transport must be one of {_TRANSPORTS}, got {transport!r}")
    if given_photons and transport == 'expected':
        raise ValueError("source='photons' supports only transport='sampled': given photons are "
                         "one realisation, transported as such")
    if detection is None:
        detection = 'sampled' if given_photons else 'expected'
    if hits is None:
        hits = 'photon_list' if source == 'particle' else 'first_arrival'
    hits_fn = bind_hits(hits, detection, hits_config, n_photons, charge_resolution)
    if emission_processes is not None and (
            not emission_processes or not set(emission_processes) <= {'cherenkov', 'scintillation'}):
        raise ValueError(f"emission_processes must be a non-empty subset of "
                         f"('cherenkov', 'scintillation'), got {emission_processes!r}")
    default_dp, medium, qe_curve = _resolve_detector_params(
        default_detector_params, physics_config, reflection_model)

    # ---- Build containers from flat args -----------------------------------
    from lucid.geometry.detector_geometry import DetectorGeometry

    det_geom = DetectorGeometry.from_config(
        json_filename, temperature=temperature, medium=medium,
        **grid_params)

    if emission_processes is not None:
        det_geom = det_geom._replace(medium=det_geom.medium._replace(
            emission_processes=tuple(emission_processes)))

    detector = det_geom.detector
    # String telescopes use the volume (per-DOM, no-reflection) photon step; all
    # surface geometries (cylinder/sphere/box) keep the byte-identical surface step.
    from lucid.geometry.string import StringTelescope
    is_volume = isinstance(detector, StringTelescope)
    default_dp = _broadcast_qe_corrections(default_dp, det_geom.num_sensors)

    # ---- Reflection model (pluggable; default 'scalar_mix'; 'scalar' = byte-identical legacy) -----
    # reflection_fn is captured statically in the differentiable step's closure;
    # build_refl_params packs the model's parameters out of DetectorParams.
    n_medium = float(det_geom.medium.refractive_index)   # sources and the cathode's first interface
    reflection_fn, build_refl_params = get_reflection_model(reflection_model, n_medium=n_medium)

    # ---- Select photon update function --------------------------------------
    # Both paths get the SAME reflection model. The sampling path (given photons / sampled
    # truth) previously hard-coded scalar reflection, which silently desynced it from the
    # differentiable forward whenever reflection_model != 'scalar'; binding reflection_fn here
    # keeps the truth generator and the expected-value model consistent.
    if transport == 'sampled':
        photon_update_fn = partial(photon_iteration_sample, reflection_fn=reflection_fn)
    else:
        photon_update_fn = jax.remat(
            make_photon_iteration_update_factors_safe(reflection_fn))

    has_cherenkov, has_scintillation, n_cher, n_scint = _emission_split(
        det_geom.medium, n_photons, wavelength_mode)
    cherenkov = dedx = None
    if source == 'particle':
        emitters, cherenkov, dedx = _load_emitters(particle, det_geom.medium, has_scintillation)
        if cherenkov_emission_band is None:
            if cherenkov.emission_band is None and wavelength_mode and has_cherenkov:
                raise ValueError(
                    f"the {particle} Cherenkov emitter records no emission_band_nm (the band its "
                    "N_photons(E) counts); install it with lucid-siren-promote --emission-band-nm "
                    "or pass cherenkov_emission_band")
            cherenkov_emission_band = cherenkov.emission_band
    optics = _build_optics(det_geom.medium, wavelength_mode, wavelength_sampling, qe_curve,
                           cherenkov_emission_band, given_photons)
    if default_dp is not None and not is_volume:
        _check_peak_qe(default_dp, reflection_fn, build_refl_params, reflection_wavelength,
                       reflection_model, optics)

    transport = _Transport(
        propagate_fn=det_geom.propagator, photon_update_fn=photon_update_fn,
        hits_fn=hits_fn, reflection_fn=reflection_fn,
        build_refl_params=build_refl_params, reflection_wavelength=reflection_wavelength,
        bounds_check=detector.bounds_check, speed_of_light=det_geom.speed_of_light,
        num_sensors=det_geom.num_sensors, K=K,
        detach_directions=given_photons, is_volume=is_volume)
    forward = _Forward(
        propagate=jax.jit(partial(_common_propagation, transport), static_argnames=('n_rays',)),
        optics=optics, n_photons=n_photons, n_medium=n_medium, hits=hits,
        has_cherenkov=has_cherenkov, has_scintillation=has_scintillation,
        n_cher=n_cher, n_scint=n_scint)

    # ---- Return the right function ------------------------------------------
    if given_photons:
        impl = jax.jit(partial(_simulation_with_data_impl, forward))
    elif source == 'light_source':
        impl = jax.jit(partial(_simulation_sensor_calibration_impl, forward))
    else:
        impl = jax.jit(partial(_simulation_without_data_impl, forward, emitters))
        # The SIREN params are jit arguments, as the DetectorParams are (_bake_detector_params).
        impl = partial(impl, model_params=cherenkov.params,
                       dedx_params=None if dedx is None else dedx.params)
    fn = (impl if default_dp is None
          else _bake_detector_params(impl, default_dp, slot=0 if given_photons else 1))
    return Simulator(fn, det_geom, reflection_model, default_dp, cherenkov)
