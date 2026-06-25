"""Event simulator factory (setup_event_simulator)."""
from lucid.sources.siren_rays import (
    make_cherenkov_surrogate_fn,
    make_scintillation_surrogate_fn,
    predict_t0,
)
from lucid.siren.core import build_cherenkov_context, build_dedx_context
from lucid.propagation.cylinder import create_photon_propagator
from lucid.propagation.box import create_box_photon_propagator, box_bounds_check
from lucid.geometry import generate_detector, get_material_from_config
from lucid.utils import (
    unpack_t0_params, unpack_siren_params,
    get_speed_of_light_in_material,
    spherical_to_cartesian, base_dir_path,
    smear_times, smear_charges_SK_like,
)
from lucid.detector_params import DetectorParams, ParticleParams, load_detector_params, load_physics_config
from lucid.wavelength.medium import make_medium, load_qe_curve, qe_curve_bounds, _MATERIALS_DIR
from lucid.wavelength.optical_model import evaluate_optical_model, OpticalArrays
from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths, build_qe_weighted_cherenkov_sampler,
)

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import os
from functools import partial
from lucid.siren.training.inference import SIRENPredictor

from lucid.simulation.optics import (
    normalize, jax_normalize, jax_rotate_vector,
)
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.reflection import get_reflection_model
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood, make_hits_moments,
    make_hits_per_photon,
    build_make_hits_waveform, build_make_hits_waveform_expected,
    build_make_hits_per_photon_shotgun,
)

# ===================================================================
# Event simulator factory
# ===================================================================

def setup_event_simulator(
        json_filename,
        n_photons=1_000_000,
        temperature=0.2,
        K=7,
        is_data=False,
        is_calibration=False,
        max_candidates_per_ray=4,
        detector_type='Cylinder',
        use_expected_value=True,
        particle='muon',
        apply_smearing=True,
        physics_config=None,
        default_detector_params=False,
        wavelength_mode=True,
        hit_mode=None,
        n_grad_iters=None,
        pos_grad_threshold=None,  # None → use mode default (calib:K, track:0)
        waveform_config=None,
        wavelength_sampling='cherenkov',
        overlap_st_width_frac=0.35,
        overlap_renorm=1.0,
        overlap_mode='interp',
        reflection_model='scalar',
        reflection_wavelength=400.0,
        spectrum=None,
        cherenkov_emission_band=None,
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
    is_data : bool
        ROOT-file data mode.
    is_calibration : bool
        Calibration mode (source passed at call time).
    max_candidates_per_ray : int
        Grid cell sensor limit.
    detector_type : str
        'Cylinder', 'Sphere', or 'Box'.
    use_expected_value : bool
        True -> STE (differentiable), False -> MC sampling.
    particle : str
        Particle type (e.g., 'muon', 'electron'). Used to load SIREN model,
        t0 parameters, and photon normalization from config.
    apply_smearing : bool
        If True, apply SK-like charge and time smearing in data mode.
    physics_config : str or None
        Path to physics config JSON (e.g. ``SK_physics_config.json``).
        Required when ``default_detector_params=True``.
    default_detector_params : bool or DetectorParams
        Controls whether ``detector_params`` is baked into the returned function:

        - ``False`` (default) -- returned function **requires** ``detector_params``
          as an explicit argument.
        - ``True`` -- loads ``DetectorParams`` from *physics_config* at setup time
          and bakes it into the closure.
        - ``DetectorParams`` instance -- bakes that instance directly (no file load).
    hit_mode : str or None
        How sensor hits are aggregated. ``None`` (default) uses the source-appropriate
        choice; pass a value to override (useful for analysis):

        - ``'aggregated'`` -- differentiable per-sensor (charges, times). Default for calibration.
        - ``'per_photon'`` -- per-photon arrays (log_w, times, indices, charges). Default for track.
        - ``'realistic'`` -- Bernoulli QE sampling, hard-min timing, optional smearing. Default for data.
    wavelength_sampling : str
        How LUCiD samples λ when it samples (track mode, calibration with
        ``source.wavelength=None``). Ignored silently when the caller supplies
        wavelengths explicitly (scalar/array source, ROOT-file data-mode photons).

        - ``'cherenkov'`` (default) — λ ~ 1/λ², per-photon weight = ``qe_fn(λ)``.
        - ``'cherenkov_qe'`` — λ ~ QE(λ)/λ² via inverse CDF, per-photon weight
          collapses to the scalar ``<QE>_C``. Variance-optimal in expected-value
          mode. In Bernoulli / MC mode the expected waveform matches
          ``'cherenkov'`` but per-shot Binomial fluctuations are suppressed, so
          use only when the output is interpreted as a density estimate.
          Rejected at setup when ``wavelength_mode=False``, no QE curve is
          loaded, or ``is_data=True``.
    overlap_st_width_frac : float
        Straight-through surrogate width (fraction of sensor radius) for the
        hard-step overlap gradient. Backward-only; default 0.35.
    overlap_renorm : float
        Soft-overlap renormalization constant C = hard_total/soft_total
        (restores the ~1% total/energy lost to inter-sensor gaps without
        changing the gradient direction). Default 1.0 = OFF (byte-identical).
    overlap_mode : str
        Soft-overlap lookup interpolation: ``'interp'`` (default, piecewise
        linear) or ``'cubic'`` (C2 natural spline — correct curvature for the
        autodiff Hessian wrt photon→sensor distance).
    reflection_model : str
        Reflection model: ``'scalar'`` (default, byte-identical — angle/λ-
        independent wall/sensor rates, hard direction) or ``'angular'`` (Schlick
        blacksheet wall + multilayer-Fresnel cathode sensor, with a specular/
        diffuse direction mixture). The angular model reads the
        ``DetectorParams.reflection`` fields ``wall_R0/wall_p/wall_fspec`` and
        ``cathode_nr/cathode_nk/sensor_fspec``.
    reflection_wavelength : float
        Wavelength (nm) fed to the reflection model's dispersion (cathode/glass
        Fresnel). Exact for monochromatic-laser calibration; ignored by the
        scalar model. Default 400 nm.
    spectrum : Spectrum or None
        Optional λ-sampling law (``lucid.wavelength`` Monochromatic / PowerLaw /
        QEWeighted). When given it supersedes ``wavelength_sampling`` for broadband
        sampling and provides the scalar ``<QE>_C`` collapse via ``spectrum.mean_qe``.
        Default ``None`` reproduces the ``wavelength_sampling`` behaviour.

    Returns
    -------
    callable
        When ``default_detector_params`` is ``False``:

        - **Calibration** ``(source, detector_params, key) -> (charges, times)``
        - **Track**       ``(particle_params, detector_params, key) -> (charges, times)``
        - **Data**        ``(particle_params, detector_params, key, photon_data) -> (charges, times)``

        When ``default_detector_params`` is truthy (``True`` or a ``DetectorParams``):

        - **Calibration** ``(source, key) -> (charges, times)``
        - **Track**       ``(particle_params, key) -> (charges, times)``
        - **Data**        ``(particle_params, key, photon_data) -> (charges, times)``

        When detector params are baked in, the returned function also exposes
        a ``.default_detector_params`` attribute for inspection.
    """
    # ---- Resolve default_detector_params ------------------------------------
    _medium_model_path = None
    _qe_curve_path = None

    if default_detector_params is False:
        _default_dp = None
        # Still extract medium/QE paths from physics config if provided
        if physics_config is not None:
            _, _medium_model_path, _qe_curve_path = load_physics_config(physics_config)
    elif default_detector_params is True:
        if physics_config is None:
            raise ValueError("physics_config is required when default_detector_params=True")
        _default_dp, _medium_model_path, _qe_curve_path = load_physics_config(physics_config)
    elif isinstance(default_detector_params, DetectorParams):
        _default_dp = default_detector_params
        # Extract medium/QE paths from physics config if provided alongside
        if physics_config is not None:
            _, _medium_model_path, _qe_curve_path = load_physics_config(physics_config)
    else:
        raise TypeError(
            f"default_detector_params must be bool or DetectorParams, got {type(default_detector_params)}")

    # ---- Build containers from flat args -----------------------------------
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.simulation.config import SimConfig

    det_geom = DetectorGeometry.from_config(
        json_filename, temperature=temperature,
        max_candidates_per_ray=max_candidates_per_ray,
        detector_type=detector_type,
        overlap_st_width_frac=overlap_st_width_frac,
        overlap_renorm=overlap_renorm,
        overlap_mode=overlap_mode,
        **grid_params)

    mode = 'data' if is_data else ('calibration' if is_calibration else 'track')
    sim_config = SimConfig(
        n_photons=n_photons, K=K, mode=mode,
        use_expected_value=use_expected_value,
        apply_smearing=apply_smearing,
        n_grad_iters=n_grad_iters)

    # ---- Extract fields from containers ------------------------------------
    material = det_geom.medium.material
    SPEED_OF_LIGHT_MATERIAL = det_geom.speed_of_light
    detector = det_geom.detector
    # String telescopes use the volume (per-DOM, no-reflection) photon step; all
    # surface geometries (cylinder/sphere/box) keep the byte-identical surface step.
    from lucid.geometry.string import StringTelescope
    _is_volume = isinstance(detector, StringTelescope)
    sensor_points = det_geom.sensor_points
    NUM_SENSORS = det_geom.num_sensors
    Nphot = sim_config.n_photons
    propagate_photons = det_geom.propagator

    # ---- Handle qe_corrections for baked-in detector params -----------------
    if _default_dp is not None:
        qe_corr = _default_dp.per_pmt.qe_corrections
        # If scalar placeholder (from null in JSON), broadcast to NUM_SENSORS
        if qe_corr.ndim == 0:
            _default_dp = _default_dp._replace(
                per_pmt=_default_dp.per_pmt._replace(
                    qe_corrections=jnp.ones(NUM_SENSORS) * qe_corr))
        elif len(qe_corr) != NUM_SENSORS:
            raise ValueError(
                f"qe_corrections has {len(qe_corr)} elements "
                f"but detector has {NUM_SENSORS} sensors")

    # ---- Reflection model (pluggable; default 'scalar' = byte-identical) -----
    # reflection_fn is captured statically in the differentiable step's closure;
    # build_refl_params packs the model's parameters out of DetectorParams.
    reflection_fn, build_refl_params = get_reflection_model(reflection_model)

    # ---- Two-medium (nested) detection ---------------------------------------
    # _IS_NESTED == has_interface (I>0). For single-medium it is False → the interface
    # code (and its extra RNG key) is never traced and the forward stays byte-identical.
    # Nested transport is calibration-only: track/data need SIREN-in-LS + per-mode
    # medium_id init, which are not wired (fail fast at setup, not at call time).
    _IS_NESTED = det_geom.is_nested
    if _IS_NESTED and not sim_config.is_calibration:
        raise NotImplementedError(
            "nested/two-medium transport is calibration-only (point sources); track and "
            "data modes need SIREN-in-LS emission + per-mode medium_id init (not yet wired)")

    # ---- Photon update function (ONE factory step, statically specialized on has_interface) --
    # make_photon_step(..., False) is byte-identical to the legacy single-medium step;
    # (..., True) is the legacy nested step (Snell/Fresnel/TIR + the 8th medium_id output).
    if sim_config.is_data or sim_config.use_expected_value is False:
        photon_update_fn = make_photon_step('sample', _IS_NESTED)
    else:
        photon_update_fn = jax.remat(
            make_photon_step('update_factors', _IS_NESTED, reflection_fn))

    if _IS_NESTED:
        SPEED_OF_LIGHT_OUTER = det_geom.speed_of_light_outer
        N_INNER = float(det_geom.medium.refractive_index)
        N_OUTER = float(det_geom.medium_outer.refractive_index)
    else:
        SPEED_OF_LIGHT_OUTER = None
        N_INNER = N_OUTER = None

    # ---- Geometry bounds check (delegates to detector method) ----------------
    def get_inside_detector_flag(positions):
        return detector.bounds_check(positions)

    # ---- Resolve hit_mode ---------------------------------------------------
    _VALID_HIT_MODES = ('aggregated', 'per_photon', 'realistic', 'moments', 'per_segment',
                        'waveform', 'waveform_expected', 'shotgun_per_photon')
    if hit_mode is None:
        if sim_config.is_data:
            hit_mode = 'realistic'
        elif sim_config.is_calibration:
            hit_mode = 'aggregated'
        else:
            hit_mode = 'per_photon'
    elif hit_mode not in _VALID_HIT_MODES:
        raise ValueError(
            f"hit_mode must be one of {_VALID_HIT_MODES} or None, got {hit_mode!r}")

    # ---- make_hits wrapper selection ----------------------------------------
    # Every wrapper accepts a trailing ``response`` bundle (gain, t0, spe_width, tts)
    # built from DetectorParams at call time; only the moments mode consumes it.
    def _make_hits_aggregated(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
        return make_hits_simulation(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_per_photon(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
        return make_hits_likelihood(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_realistic(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
        tts = 0.0 if response is None else response[3]
        return make_hits_data(flat_weights, flat_indices, flat_times, num_sensors,
                              qe=qe, qe_corrections=qe_corrections,
                              rng_key=qe_key, apply_smearing=sim_config.apply_smearing, tts=tts)

    def _make_hits_moments(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
        gain, t0, spe_width, tts = response
        return make_hits_moments(flat_weights, flat_indices, flat_times, num_sensors,
                                 qe=qe, qe_corrections=qe_corrections,
                                 gain=gain, spe_width=spe_width, t0=t0, tts=tts)

    def _make_hits_per_segment(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
        # Production v3: per-sensor totals + per-photon pass-through arrays (incl the
        # per-photon segment index) for the host-side per-(segment, sensor) groupby.
        tts = 0.0 if response is None else response[3]
        return make_hits_per_photon(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections,
                                    rng_key=qe_key, apply_smearing=sim_config.apply_smearing,
                                    tts=tts, flat_segment_idx=flat_segment_idx)

    # Shotgun hit modes (waveform + per-photon). Defaults match SK-realistic
    # PMT behaviour; override via ``waveform_config``.
    _wf_cfg = dict(window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0,
                   t_min_ns=0.0, smear_time=True, smear_charge=True)
    if waveform_config:
        _wf_cfg.update(waveform_config)

    if hit_mode == 'waveform':
        _wf_fn = build_make_hits_waveform(n_photons=n_photons, **_wf_cfg)
        def _make_hits_waveform(flat_weights, flat_indices, flat_times, num_sensors,
                                qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
            return _wf_fn(flat_weights, flat_indices, flat_times, num_sensors,
                          qe_key, qe, qe_corrections)
    elif hit_mode == 'waveform_expected':
        # Expected-value waveform: no Bernoulli, no gain smear — those do not
        # exist when every slot deposits a continuous weight.
        _wf_exp_cfg = {k: v for k, v in _wf_cfg.items() if k != 'smear_charge'}
        _wf_exp_fn = build_make_hits_waveform_expected(
            n_photons=n_photons, **_wf_exp_cfg)
        def _make_hits_waveform_expected(flat_weights, flat_indices, flat_times, num_sensors,
                                         qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
            return _wf_exp_fn(flat_weights, flat_indices, flat_times, num_sensors,
                              qe_key, qe, qe_corrections)
    elif hit_mode == 'shotgun_per_photon':
        _pp_fn = build_make_hits_per_photon_shotgun(
            n_photons=n_photons,
            tts_sigma_ns=_wf_cfg['tts_sigma_ns'],
            smear_time=_wf_cfg['smear_time'])
        def _make_hits_shotgun_pp(flat_weights, flat_indices, flat_times, num_sensors,
                                  qe_key, qe, qe_corrections, response=None, flat_segment_idx=None):
            return _pp_fn(flat_weights, flat_indices, flat_times, num_sensors,
                          qe_key, qe, qe_corrections)

    _make_hits_fn = {
        'aggregated': _make_hits_aggregated,
        'per_photon': _make_hits_per_photon,
        'realistic': _make_hits_realistic,
        'moments': _make_hits_moments,
        'per_segment': _make_hits_per_segment,
        'waveform': _make_hits_waveform if hit_mode == 'waveform' else None,
        'waveform_expected': _make_hits_waveform_expected if hit_mode == 'waveform_expected' else None,
        'shotgun_per_photon': _make_hits_shotgun_pp if hit_mode == 'shotgun_per_photon' else None,
    }[hit_mode]

    # ---- Wavelength-dependent medium (when wavelength_mode=True) -----
    # Sampling/grid bounds: clamp the QE curve's knot range into the
    # [300, 700] nm floor/ceiling that LUCiD's water medium covers.
    # Without a QE curve, default to the full [300, 700] nm range.
    if wavelength_mode:
        if _qe_curve_path is not None:
            _qe_lo, _qe_hi = qe_curve_bounds(_qe_curve_path)
            _wl_lo = max(300.0, _qe_lo)
            _wl_hi = min(700.0, _qe_hi)
            if _wl_hi <= _wl_lo:
                raise ValueError(
                    f"QE curve range [{_qe_lo:.1f}, {_qe_hi:.1f}] nm does not "
                    f"overlap the water medium range [300, 700] nm. Check "
                    f"the qe_curve path in physics_config: {_qe_curve_path}")
        else:
            _wl_lo, _wl_hi = 300.0, 700.0
        _wl_grid = jnp.linspace(_wl_lo, _wl_hi, 200)
        _medium_wl = make_medium(material, wavelength_grid=_wl_grid,
                                 medium_model_path=_medium_model_path)
        # Outer-medium wavelength curves for the nested two-medium detector (resolved
        # by material name from config/materials/<outer>.json). None for single-medium.
        _medium_wl_outer = (make_medium(det_geom.medium_outer.material,
                                        wavelength_grid=_wl_grid)
                            if _IS_NESTED else None)
        _qe_fn = load_qe_curve(_qe_curve_path) if _qe_curve_path else None
    else:
        _wl_lo, _wl_hi = 300.0, 700.0
        _medium_wl = None
        _medium_wl_outer = None
        _qe_fn = None

    # ---- Cherenkov λ-SAMPLING band (Method A) ----------------------------
    # The medium grid + QE machinery stay clamped to [_wl_lo, _wl_hi] (the
    # [300,700]∩QE window the water medium covers). But the net's nphot(E) is
    # the photon count over the GEANT4 EMISSION band (e.g. [275,674] nm), which
    # extends past the QE/medium window. Sampling only over [_wl_lo,_wl_hi]
    # treats all nphot photons as detectable → over-counts charge by the
    # QE-dead fringe fraction (~17%, drives recon dE ≈ −13%). Sampling over the
    # true emission band instead lets QE(λ)=0 (out-of-knot) and the medium
    # interp-clamp make the fringe photons INERT, so the detected charge uses
    # the correct in-band fraction with NO scalar norm. Default None →
    # byte-identical (sample over [_wl_lo,_wl_hi]); pass the net's GEANT4
    # emission band to fix the normalization.
    if cherenkov_emission_band is not None:
        _sample_lo = float(cherenkov_emission_band[0])
        _sample_hi = float(cherenkov_emission_band[1])
    else:
        _sample_lo, _sample_hi = _wl_lo, _wl_hi

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
        if _qe_fn is None:
            raise ValueError(
                "wavelength_sampling='cherenkov_qe' requires a QE curve — set "
                "qe_curve in the physics_config.")
        if is_data:
            raise ValueError(
                "wavelength_sampling='cherenkov_qe' is incompatible with "
                "is_data=True: PhotonSim ROOT photons carry their own "
                "wavelengths; LUCiD does not sample in data mode.")
        _qe_sampler, _mean_qe_c = build_qe_weighted_cherenkov_sampler(
            _qe_fn, _wl_lo, _wl_hi)
    else:
        _qe_sampler = None
        _mean_qe_c = None

    def _get_optical_arrays(n, detector_params, key, wavelengths=None):
        """Compute per-photon (n,) scatter/absorption arrays and QE weights.

        When wavelength_mode=True: uses medium coefficients at given wavelengths.
        When wavelength_mode=False: broadcasts DetectorParams scalars.

        Returns (scatter_lengths, absorption_lengths, qe_weights, key).
        qe_weights is (n,) or None.
        """
        if wavelength_mode and _medium_wl is None:
            raise RuntimeError(
                "wavelength_mode=True but no medium model was loaded. "
                "Provide a physics_config with 'medium_model' or set wavelength_mode=False.")
        if not wavelength_mode:
            # Monochromatic / scalar mode: the pure optical model broadcasts the
            # DetectorParams scalars (wavelengths=None path).
            oa = evaluate_optical_model(detector_params, None, _medium_wl, n)
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
            if spectrum is not None:
                # Explicit Spectrum object supersedes the wavelength_sampling string.
                wavelengths = spectrum.sample(wl_key, n, _wl_lo, _wl_hi)
                sampled_via_qe_importance = spectrum.mean_qe is not None
            elif wavelength_sampling == 'cherenkov_qe':
                wavelengths = _qe_sampler(wl_key, n)
                sampled_via_qe_importance = True
            else:
                wavelengths = sample_cherenkov_wavelengths(
                    wl_key, n, lambda_min=_sample_lo, lambda_max=_sample_hi)
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
            detector_params, wavelengths, _medium_wl, n,
            qe_fn=None if sampled_via_qe_importance else _qe_fn)
        # The scalar <QE>_C comes from the Spectrum when one was used, else the
        # built-in QE-weighted sampler.
        _collapse_qe = (spectrum.mean_qe
                        if (spectrum is not None and spectrum.mean_qe is not None)
                        else _mean_qe_c)
        qe_weights = jnp.full(n, _collapse_qe) if sampled_via_qe_importance else oa.qe
        return oa.scatter_len, oa.mie_len, oa.abs_len, qe_weights, key

    def _get_optical_arrays_nested(n, detector_params, key, wavelengths=None):
        """Per-photon optics for BOTH media of a nested detector at the SAME wavelengths.

        Pure absorption (no re-emission) ⇒ each photon's λ is fixed for its whole life, so
        we resolve λ once and evaluate the optical model against the inner and the outer
        medium curves. The per-photon QE weight is a PMT property (applied at the outer
        surface) and is shared. Returns
        ``(sl_in, ml_in, al_in, sl_out, ml_out, al_out, qe_weights, key)``.
        """
        if not wavelength_mode:
            # Monochromatic: both media broadcast the DetectorParams scalars (identical
            # optics; only the interface n + speed differ). Physical per-medium optics
            # need wavelength_mode (or per-medium DetectorParams scalars, a later phase).
            oa = evaluate_optical_model(detector_params, None, _medium_wl, n)
            return (oa.scatter_len, oa.mie_len, oa.abs_len,
                    oa.scatter_len, oa.mie_len, oa.abs_len, oa.qe, key)

        # Resolve wavelengths once (sample Cherenkov 1/λ² when None; else broadcast/use).
        if wavelengths is None:
            key, wl_key = jax.random.split(key)
            if spectrum is not None:
                wavelengths = spectrum.sample(wl_key, n, _wl_lo, _wl_hi)
            elif wavelength_sampling == 'cherenkov_qe':
                wavelengths = _qe_sampler(wl_key, n)
            else:
                wavelengths = sample_cherenkov_wavelengths(
                    wl_key, n, lambda_min=_sample_lo, lambda_max=_sample_hi)
        else:
            wavelengths = jnp.asarray(wavelengths)
            if wavelengths.ndim == 0:
                wavelengths = jnp.full(n, wavelengths)

        oa_in = evaluate_optical_model(detector_params, wavelengths, _medium_wl, n, qe_fn=_qe_fn)
        # Outer medium: use the separately-fittable outer optical bundle when present
        # (per-medium calibration); else share the inner bundle's deviation curves.
        dp_outer = detector_params
        if detector_params.outer_optics is not None:
            dp_outer = detector_params._replace(
                scattering=detector_params.outer_optics.scattering,
                absorption=detector_params.outer_optics.absorption)
        oa_out = evaluate_optical_model(dp_outer, wavelengths, _medium_wl_outer, n, qe_fn=None)
        return (oa_in.scatter_len, oa_in.mie_len, oa_in.abs_len,
                oa_out.scatter_len, oa_out.mie_len, oa_out.abs_len, oa_in.qe, key)

    # ================================================================
    # Core propagation (shared by all modes)
    # ================================================================

    @partial(jax.jit, static_argnames=(
        'n_rays', 'K', 'n_grad_iters', 'max_candidates_per_ray', 'num_sensors',
        'propagate_fn', 'photon_update_fn', 'pos_grad_threshold', 'make_hits_fn',
        'is_volume'))
    def _common_propagation(
            positions, directions, intensities, times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key,
            num_sensors, K, n_grad_iters, max_candidates_per_ray,
            propagate_fn, photon_update_fn,
            pos_grad_threshold, make_hits_fn, is_volume=False, segment_idx=None,
            scatter_lengths_outer=None, mie_scatter_lengths_outer=None,
            absorption_lengths_outer=None, initial_medium_id=None):
        """Core photon propagation loop.

        Parameters
        ----------
        scatter_lengths : jnp.ndarray
            Per-photon scattering lengths, shape (n_rays,).
        mie_scatter_lengths : jnp.ndarray
            Per-photon Mie scattering lengths, shape (n_rays,).
        absorption_lengths : jnp.ndarray
            Per-photon absorption lengths, shape (n_rays,).
        qe_per_photon : jnp.ndarray
            Per-photon quantum efficiency, shape (n_rays,).
        pos_grad_threshold : int
            Iteration threshold for position stop_gradient.
        make_hits_fn : callable
            Sensor response aggregation function.
        """

        # Packed reflection params for the pluggable reflection model (default scalar →
        # ScalarReflection(wall_rate, sensor_rate)). build_refl_params is chosen at setup.
        refl_params = build_refl_params(detector_params)
        # Wavelength fed to the reflection model. Scalar reflection ignores it; the angular
        # (Fresnel) model uses it for the cathode/glass dispersion. A scalar reflection
        # wavelength is exact for monochromatic-laser calibration (the validated case); a
        # per-photon λ array can be threaded here later for broadband sources.
        refl_lam = jnp.asarray(reflection_wavelength)
        qe_corrections = detector_params.per_pmt.qe_corrections
        g = detector_params.scattering.g

        from lucid.simulation.types import PhotonState

        initial_survival = jnp.ones(n_rays)

        def propagation_step(carry, i):
            state = carry
            key, prop_key = jax.random.split(state.key)

            prop_results = propagate_fn(state.positions, state.directions)
            depositions = prop_results['sensor_weights']
            sensor_indices = prop_results['sensor_indices']
            inside_sensor = prop_results['inside_sensor']

            if is_volume:
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
                 per_dom_charges, continuing_factors, logp_increments) = jax.vmap(
                    photon_step_volume,
                    in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0, 0, None, None)
                )(state.positions, state.directions, state.times,
                  sensor_distances.squeeze(-1), depositions,
                  scatter_lengths, mie_scatter_lengths, absorption_lengths, seg_lengths,
                  rng_keys, SPEED_OF_LIGHT_MATERIAL, g)

                inside_detector = get_inside_detector_flag(new_positions)
                safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)
                new_survival = state.survival * safe_continuing

                # DiCE magic box from the PRE-step log_p: forward value = 1 (charge unchanged),
                # reverse-mode injects the accumulated optical score so the deposit carries the
                # gradient of the probability the photon's trajectory reached this step.
                dice_dep = jnp.exp(state.log_p - jax.lax.stop_gradient(state.log_p))   # (n_rays,)
                physical_intensities = intensities * state.survival * dice_dep
                updated_weights = per_dom_charges.T * physical_intensities[None, :]
                total_times = sensor_distances / SPEED_OF_LIGHT_MATERIAL + state.times[:, None]

                next_pos = jnp.where(i < pos_grad_threshold, new_positions, jax.lax.stop_gradient(new_positions))
                next_dir = jnp.where(i < n_grad_iters, new_directions, jax.lax.stop_gradient(new_directions))
                new_log_p = state.log_p + logp_increments   # accumulate optical score for FUTURE deposits
                new_state = PhotonState(
                    positions=next_pos, directions=next_dir, times=new_times,
                    survival=new_survival, key=key, log_p=new_log_p)
                outputs = (updated_weights, sensor_indices, total_times.squeeze(-1))
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
            if _IS_NESTED:
                # Two-medium: select per-photon optics + speed by the photon's current
                # medium, hand the interface flag + medium id + the two refractive
                # indices to the nested step (it returns the updated medium id 8th).
                mid = state.medium_id
                in_inner = (mid == 0)
                sl_p = jnp.where(in_inner, scatter_lengths, scatter_lengths_outer)
                ml_p = jnp.where(in_inner, mie_scatter_lengths, mie_scatter_lengths_outer)
                al_p = jnp.where(in_inner, absorption_lengths, absorption_lengths_outer)
                spd_p = jnp.where(in_inner, SPEED_OF_LIGHT_MATERIAL, SPEED_OF_LIGHT_OUTER)
                hit_interface = prop_results['hit_interface']
                (new_positions, new_directions, new_times,
                 detect_probs, reflection_attenuations,
                 continuing_factors, logp_increments, new_medium_id) = jax.vmap(
                    photon_update_fn,   # factory step, has_interface=True (8-tuple incl. new_medium_id)
                    in_axes=(0, 0, 0, 0, 0,
                             0, 0, None, None, 0,
                             0, None, 0, 0,
                             0, 0, None, None)
                )(state.positions, state.directions, state.times,
                  surface_distances, normals,
                  sl_p, ml_p, g, refl_params, al_p,
                  hit_sensor, refl_lam, rng_keys, spd_p,
                  hit_interface, mid, N_INNER, N_OUTER)
            else:
                (new_positions, new_directions, new_times,
                 detect_probs, reflection_attenuations,
                 continuing_factors, logp_increments) = jax.vmap(
                    photon_update_fn,
                    in_axes=(0, 0, 0, 0, 0,
                             0, 0, None, None, 0,
                             0, None, 0, None)
                )(state.positions, state.directions, state.times,
                  surface_distances, normals,
                  scatter_lengths, mie_scatter_lengths, g, refl_params,
                  absorption_lengths,
                  hit_sensor, refl_lam, rng_keys, SPEED_OF_LIGHT_MATERIAL)
                new_medium_id = None

            inside_detector = get_inside_detector_flag(new_positions)
            safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)

            new_survival = state.survival * safe_continuing

            physical_intensities = intensities * state.survival
            # DiCE magic box from the PRE-step log_p: forward value = 1 (charge unchanged),
            # gradient = the accumulated optical score for deposits at this step. The sampling
            # path keeps log_p=0 → dice_dep=1 (no effect), so the data oracle is untouched.
            dice_dep = jnp.exp(state.log_p - jax.lax.stop_gradient(state.log_p))   # (n_rays,)
            detected_factors = detect_probs * reflection_attenuations * dice_dep
            updated_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
            # Final hop to the sensor uses the speed of the photon's current medium. For a
            # sensor hit the photon is always in the outer medium (it crossed the interface
            # in an earlier step); per-photon spd_p captures that, scalar for single-medium.
            # hit_times_meters is (max_cand, n_rays, 1); per-photon speed → (1, n_rays, 1).
            _final_speed = spd_p[None, :, None] if _IS_NESTED else SPEED_OF_LIGHT_MATERIAL
            times_ns = hit_times_meters / _final_speed
            total_times = times_ns + state.times[:, None]

            iter_weights = updated_weights
            iter_indices = sensor_indices
            iter_times = total_times.squeeze(-1)

            # Stop gradient on positions and directions independently:
            # - Position: threshold=K → gradient flows all iterations (standard)
            #             threshold=0 → always stop (likelihood)
            # - Direction: n_grad_iters=0 (reconstruction) → always stop
            #              n_grad_iters=2 (calibration) → gradient flows for first 2
            next_pos = jnp.where(i < pos_grad_threshold, new_positions, jax.lax.stop_gradient(new_positions))
            next_dir = jnp.where(i < n_grad_iters, new_directions, jax.lax.stop_gradient(new_directions))

            new_log_p = state.log_p + logp_increments   # accumulate score for FUTURE deposits

            new_state = PhotonState(
                positions=next_pos,
                directions=next_dir,
                times=new_times,
                survival=new_survival,
                key=key,
                log_p=new_log_p,
                medium_id=new_medium_id,   # None (single-medium) carries through unchanged
            )
            outputs = (iter_weights, iter_indices, iter_times)
            return new_state, outputs

        init_state = PhotonState(
            positions=positions,
            directions=directions,
            times=times,
            survival=initial_survival,
            key=key,
            log_p=jnp.zeros(n_rays),
            medium_id=initial_medium_id,    # None for single-medium; (n_rays,) int for nested
        )
        propagation_step_remat = jax.remat(propagation_step)

        _, (all_weights, all_indices, all_times) = jax.lax.scan(
            propagation_step_remat, init_state, jnp.arange(K))

        flat_weights = all_weights.reshape(-1)
        flat_indices = all_indices.reshape(-1)
        flat_times = all_times.reshape(-1)

        # Tile per-photon QE to match flat shape.
        # all_weights shape: (K, max_candidates_per_ray, n_rays), C-order reshape
        # → photon index is i % n_rays
        photon_idx = jnp.arange(flat_weights.shape[0]) % n_rays
        flat_qe = qe_per_photon[photon_idx]

        key, qe_key = jax.random.split(key)
        # Calibrated PMT response bundle (gain, t0, spe_width, tts) — consumed by the
        # 'moments' mode; ignored by the others. Neutral defaults ⇒ byte-identical.
        response = (detector_params.per_pmt.gain, detector_params.per_pmt.t0,
                    detector_params.response.spe_width, detector_params.response.tts)
        # Per-photon segment id (per_segment mode), broadcast to flat shape via the
        # same i % n_rays trick as flat_qe. None for every other mode → byte-identical.
        flat_segment_idx = (segment_idx[photon_idx] if segment_idx is not None else None)
        return make_hits_fn(
            flat_weights, flat_indices, flat_times, num_sensors, qe_key, flat_qe, qe_corrections,
            response, flat_segment_idx=flat_segment_idx)

    # ================================================================
    # Mode-specific simulation functions
    # ================================================================

    @jax.jit
    def _simulation_with_data_impl(particle_params, detector_params, key, photon_data):
        """Data mode: photons from ROOT/PhotonSim files, particle_params is ParticleParams."""
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        photon_origins = photon_data['photon_origins'] / 100.0  # cm to m
        photon_directions = photon_data['photon_directions']
        photon_times = photon_data['photon_times']

        # Apply rotation if specified (using jax.lax.cond for JIT compatibility)
        def apply_rotation_fn(args):
            origins, directions, rot_axis, rot_angle = args
            rotated_directions = jax.vmap(
                lambda v: jax_rotate_vector(v, rot_axis, rot_angle)
            )(directions)
            rotated_origins = jax.vmap(
                lambda v: jax_rotate_vector(v, rot_axis, rot_angle)
            )(origins)
            return rotated_origins, rotated_directions

        def no_rotation_fn(args):
            origins, directions, _, _ = args
            return origins, directions

        rotation_axis = photon_data['rotation_axis']
        rotation_angle = photon_data['rotation_angle']
        apply_rotation = photon_data['apply_rotation']

        final_origins, final_directions = jax.lax.cond(
            apply_rotation,
            apply_rotation_fn,
            no_rotation_fn,
            (photon_origins, photon_directions, rotation_axis, rotation_angle)
        )

        # Apply translation if specified (AFTER rotation)
        def apply_translation_fn(args):
            origins, translation_vec = args
            return origins + translation_vec[None, :]

        def no_translation_fn(args):
            origins, _ = args
            return origins

        apply_translation = photon_data.get('apply_translation', False)
        translation_vector = photon_data.get('translation_vector', jnp.array([0.0, 0.0, 0.0]))

        final_origins = jax.lax.cond(
            apply_translation,
            apply_translation_fn,
            no_translation_fn,
            (final_origins, translation_vector)
        )

        n_rays = photon_origins.shape[0]
        mask = jnp.arange(n_rays) < photon_data['N']
        photon_intensities = 1.0 * mask.astype(jnp.float32)

        # Per-photon optical properties — use PhotonSim wavelengths if available
        data_wavelengths = photon_data.get('wavelengths', None)
        scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            n_rays, detector_params, key, wavelengths=data_wavelengths)
        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.response.qe
        else:
            qe_per_photon = jnp.full(n_rays, detector_params.response.qe)

        # Per-photon segment id for the per_segment production hit mode (int32, padded
        # with -1). None for recon data (pad_photon_data) → other hit modes ignore it.
        _segment_idx = photon_data.get('photon_segment_index', None)

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            final_origins, final_directions, photon_intensities, photon_times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn, is_volume=_is_volume,
            segment_idx=_segment_idx)

    # SIREN config: model path + ray-sampling knobs. The s/s_max-trained net carries its
    # own count/range model (nphot/smax) in the trained-model metadata, read in the context.
    # Only track mode uses the SIREN surrogate; calibration/data sources supply their own
    # photons, so don't require a data/<material>/<particle>/siren_params.json for them
    # (lets calibration-only materials like a LAB-LS buffer run without a SIREN).
    siren_cfg = unpack_siren_params(particle, material) if mode == 'track' else None

    # ---- Track-mode emission-process dispatch (refactor-v2 wbls/scint merge) ----
    # The medium's `emission_processes` tuple decides which surrogates run. The
    # Cherenkov/Scintillation photon-count split (when both run) is a static,
    # material-level knob (`medium.cherenkov_fraction`). Water has only
    # ("cherenkov",) → `_has_scintillation` is False → the scintillation branch is
    # never even traced, so the Cherenkov-only path below stays byte-identical.
    _medium = det_geom.medium
    _has_cherenkov = "cherenkov" in _medium.emission_processes
    _has_scintillation = "scintillation" in _medium.emission_processes
    if _has_scintillation and not wavelength_mode:
        raise ValueError(
            "Scintillation emission requires wavelength_mode=True (the Moyal "
            "wavelength sampler feeds the per-photon spectral weight).")
    if _has_cherenkov and _has_scintillation:
        _n_cher = int(round(Nphot * float(_medium.cherenkov_fraction)))
        _n_scint = Nphot - _n_cher
    elif _has_cherenkov:
        _n_cher, _n_scint = Nphot, 0
    elif _has_scintillation:
        _n_cher, _n_scint = 0, Nphot
    else:
        raise ValueError(
            f"medium {_medium.material!r} has no emission_processes enabled "
            f"({_medium.emission_processes!r}).")
    _n_total = _n_cher + _n_scint

    @jax.jit
    def _simulation_without_data_impl(particle_params, detector_params, key, model_params):
        """SIREN mode: particle_params is ParticleParams.

        ``model_params`` is the Cherenkov SIREN params (threaded). When the medium
        scintillates the scintillation surrogate (``scint_ray_fn``) and its dE/dx
        net (``dedx_model_params``) are closed over from the build section, and the
        per-photon arrays are the Cherenkov+Scintillation concatenation.
        """
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        a_c, l_c, b_c = t0_params      # cubic stretched_exp_delay coeffs (three length-4 cubics in log10 E)
        _pt0 = jax.vmap(predict_t0, in_axes=(0, None, None, None, None))
        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold

        if _has_scintillation:
            # ---- Cherenkov + Scintillation (wbls) / scintillation-only path ----
            key, cher_key, scint_key, opt_key, wl_key = jax.random.split(key, 5)
            parts_dirs, parts_origins, parts_intens, parts_times, parts_wls = [], [], [], [], []

            if _has_cherenkov:
                ch_dirs, ch_origins, ch_intens = cherenkov_get_rays(
                    track_origin, track_direction, energy, _n_cher, model_params, cher_key)
                ch_dist_mm = jnp.linalg.norm(ch_origins - track_origin, axis=1) * 1000
                ch_t0 = jax.lax.stop_gradient(_pt0(ch_dist_mm, energy,
                                                   jnp.asarray(a_c), jnp.asarray(l_c), jnp.asarray(b_c)))
                if wavelength_sampling == 'cherenkov_qe':
                    ch_wls = _qe_sampler(wl_key, _n_cher)
                else:
                    ch_wls = sample_cherenkov_wavelengths(
                        wl_key, _n_cher, lambda_min=_wl_lo, lambda_max=_wl_hi)
                parts_dirs.append(ch_dirs); parts_origins.append(ch_origins)
                parts_intens.append(ch_intens); parts_times.append(ch_t0); parts_wls.append(ch_wls)

            # Scintillation surrogate: isotropic rays, Chou-quenched weights (S/kB/C),
            # biexp emission delay (tau_rise/tau_fall) on top of the t0 baseline, Moyal λ.
            s = detector_params.scintillation
            sc_dirs, sc_origins, sc_intens, sc_tdelay, sc_wls = scint_ray_fn(
                track_origin, track_direction, energy, _n_scint,
                dedx_model_params, scint_key,
                s.S, s.kB, s.C, s.tau_rise, s.tau_fall)
            sc_dist_mm = jnp.linalg.norm(sc_origins - track_origin, axis=1) * 1000
            sc_t0 = jax.lax.stop_gradient(_pt0(sc_dist_mm, energy,
                                               jnp.asarray(a_c), jnp.asarray(l_c), jnp.asarray(b_c)))
            parts_dirs.append(sc_dirs); parts_origins.append(sc_origins)
            parts_intens.append(sc_intens); parts_times.append(sc_t0 + sc_tdelay); parts_wls.append(sc_wls)

            photon_directions = jnp.concatenate(parts_dirs)
            photon_origins = jnp.concatenate(parts_origins)
            photon_intensities = jnp.concatenate(parts_intens)
            photon_times = jnp.concatenate(parts_times)
            wavelengths = jnp.concatenate(parts_wls)

            scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
                _n_total, detector_params, opt_key, wavelengths=wavelengths)
            if qe_weights is not None:
                qe_per_photon = qe_weights * detector_params.response.qe
            else:
                qe_per_photon = jnp.full(_n_total, detector_params.response.qe)
            return _common_propagation(
                photon_origins, photon_directions, photon_intensities, photon_times,
                scatter_lengths, mie_scatter_lengths, absorption_lengths,
                qe_per_photon,
                _n_total, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
                propagate_photons, photon_update_fn,
                pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn, is_volume=_is_volume)

        # ---- Cherenkov-only path (byte-identical to pre-scint forward) ----
        key, ray_key, opt_key = jax.random.split(key, 3)
        # New refactor-v2 Cherenkov emitter: intensities already carry the absolute count
        # (pmf × n_photons_fn(E)); no separate normalization or mean_topk amplitude.
        photon_directions, photon_origins, photon_intensities = cherenkov_get_rays(
            track_origin, track_direction, energy, Nphot, model_params, ray_key)
        photon_times = jnp.zeros((Nphot,))

        distances_to_vertex = jnp.linalg.norm(photon_origins - track_origin, axis=1) * 1000
        # Emission-time baseline t0(distance_to_vertex, energy), DETACHED: the TIME term does not carry
        # ENERGY/VERTEX gradient through the emission-time model (those flow via geometry/charge).
        t0 = jax.lax.stop_gradient(_pt0(distances_to_vertex, energy,
                                        jnp.asarray(a_c), jnp.asarray(l_c), jnp.asarray(b_c)))

        # Per-photon optical properties (Cherenkov spectrum when wavelength_mode)
        scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.response.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.response.qe)

        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times + t0,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn, is_volume=_is_volume)

    @jax.jit
    def _simulation_sensor_calibration_impl(source, detector_params, key):
        """Calibration mode: source is a callable (IsotropicSource or LaserSource).

        Point sources are the supported two-medium source (the spectrum is a pluggable
        knob: ``source.wavelength=None`` → Cherenkov 1/λ², scalar → monochromatic laser,
        ``(Nphot,)`` → an explicit spectrum). For nested detectors each photon's
        ``medium_id`` is initialised from its emission radius and BOTH media's per-photon
        optics are evaluated at the same wavelengths.
        """
        key, source_key, opt_key = jax.random.split(key, 3)
        photon_directions, photon_origins, photon_intensities = source(Nphot, source_key)
        photon_times = jnp.zeros((Nphot,))

        # Source wavelength can be None (→ Cherenkov), scalar, or (Nphot,) array;
        # the optical-array helpers normalize the shape.
        wavelengths = getattr(source, 'wavelength', None)
        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold

        if _IS_NESTED:
            (sl_in, ml_in, al_in, sl_out, ml_out, al_out,
             qe_weights, key) = _get_optical_arrays_nested(
                Nphot, detector_params, opt_key, wavelengths=wavelengths)
            if qe_weights is not None:
                qe_per_photon = qe_weights * detector_params.response.qe
            else:
                qe_per_photon = jnp.full(Nphot, detector_params.response.qe)
            # Per-photon initial medium from the emission point (0 inner / 1 outer).
            medium_id0 = detector.region_of(photon_origins)
            return _common_propagation(
                photon_origins, photon_directions, photon_intensities, photon_times,
                sl_in, ml_in, al_in,
                qe_per_photon,
                Nphot, detector_params, key, NUM_SENSORS, sim_config.K,
                sim_config.effective_n_grad_iters, max_candidates_per_ray,
                propagate_photons, photon_update_fn,
                pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn, is_volume=_is_volume,
                scatter_lengths_outer=sl_out, mie_scatter_lengths_outer=ml_out,
                absorption_lengths_outer=al_out, initial_medium_id=medium_id0)

        scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key, wavelengths=wavelengths)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.response.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.response.qe)

        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn, is_volume=_is_volume)

    # ---- Return the right function ------------------------------------------
    if sim_config.is_data:
        if _default_dp is not None:
            @jax.jit
            def _sim_data_default(particle_params, key, photon_data):
                return _simulation_with_data_impl(particle_params, _default_dp, key, photon_data)
            _sim_data_default.default_detector_params = _default_dp
            _sim_data_default.medium = det_geom.medium      # production introspection
            _sim_data_default.det_geom = det_geom           # (event_generation reads these)
            return _sim_data_default
        else:
            return _simulation_with_data_impl
    elif sim_config.is_calibration:
        if _default_dp is not None:
            @jax.jit
            def _sim_calibration_default(source, key):
                return _simulation_sensor_calibration_impl(source, _default_dp, key)
            _sim_calibration_default.default_detector_params = _default_dp
            _sim_calibration_default.medium = det_geom.medium
            _sim_calibration_default.det_geom = det_geom
            return _sim_calibration_default
        else:
            return _simulation_sensor_calibration_impl
    else:
        model_base_path = siren_cfg['siren_model_path']
        photonsim_predictor = SIRENPredictor(model_base_path)
        model_params = photonsim_predictor.params
        # Build the new Cherenkov emitter once (closes over the SIREN context: net + domain
        # ranges + smax/nphot from the trained-model metadata). Referenced as a closure var
        # by _simulation_without_data_impl.
        _cher_ctx = build_cherenkov_context(photonsim_predictor, siren_cfg['ray_sampling'])
        cherenkov_get_rays = make_cherenkov_surrogate_fn(_cher_ctx)
        t0_params = unpack_t0_params(particle, material)   # (a_coeffs, l_coeffs, b_coeffs) cubic

        # Scintillation surrogate — built only when the medium scintillates (wbls).
        # Closed over by _simulation_without_data_impl's scintillation branch.
        if _has_scintillation:
            if siren_cfg['dedx_model_path'] is None:
                raise ValueError(
                    f"medium {_medium.material!r} enables scintillation but "
                    f"data/{material}/{particle}/siren_params.json has no 'dedx_model' "
                    f"block — add one pointing to the dE/dx SIREN.")
            # Moyal sampling params are static (they bake the inverse-CDF lookup at
            # factory time). Resolve from the material JSON's scintillation.spectrum.
            from lucid.detector_params import _scintillation_defaults_from_medium
            scint_material_path = os.path.join(_MATERIALS_DIR, f"{material}.json")
            scint_defaults = _scintillation_defaults_from_medium(scint_material_path)
            if 'moyal_loc' not in scint_defaults or 'moyal_scale' not in scint_defaults:
                raise ValueError(
                    f"material {_medium.material!r} enables scintillation but "
                    f"{scint_material_path!r} has no scintillation.spectrum.moyal_loc / "
                    f"moyal_scale — the surrogate cannot build its λ inverse-CDF sampler.")
            dedx_predictor = SIRENPredictor(siren_cfg['dedx_model_path'])
            dedx_model_params = dedx_predictor.params
            scint_ray_fn = make_scintillation_surrogate_fn(
                build_dedx_context(dedx_predictor, siren_cfg['dedx_sampling']),
                _medium.scintillation_lambda_min,
                _medium.scintillation_lambda_max,
                moyal_loc=scint_defaults['moyal_loc'],
                moyal_scale=scint_defaults['moyal_scale'],
            )
        if _default_dp is not None:
            @jax.jit
            def _sim_track_default(particle_params, key):
                return _simulation_without_data_impl(particle_params, _default_dp, key,
                                                     model_params=model_params)
            _sim_track_default.default_detector_params = _default_dp
            _sim_track_default.medium = det_geom.medium
            _sim_track_default.det_geom = det_geom
            return _sim_track_default
        else:
            return partial(_simulation_without_data_impl, model_params=model_params)


