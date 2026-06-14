"""Event simulator factory (setup_event_simulator)."""
from lucid.sources.siren_rays import (  # noqa: F401
    predict_t0_cubic,
    photonsim_differentiable_get_rays,
    predict_t0,
)
from lucid.propagation.cylinder import create_photon_propagator
from lucid.propagation.sphere import create_sphere_photon_propagator
from lucid.propagation.box import create_box_photon_propagator, box_bounds_check
from lucid.geometry import generate_detector, get_material_from_config
from lucid.utils import (
    unpack_t0_params, unpack_photonsim_params,
    get_speed_of_light_in_material,
    spherical_to_cartesian, base_dir_path,
    smear_times, smear_charges_SK_like,
)
from lucid.detector_params import DetectorParams, ParticleParams, load_detector_params, load_physics_config
from lucid.wavelength.medium import make_medium, load_qe_curve, qe_curve_bounds
from lucid.wavelength.optical_model import evaluate_optical_model, OpticalArrays
from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths, build_qe_weighted_cherenkov_sampler,
)

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import os
from lucid.siren.core import create_photonsim_siren_grid
from functools import partial
from lucid.siren.training.inference import SIRENPredictor

from lucid.simulation.optics import (
    normalize, jax_normalize, jax_rotate_vector,
)
from lucid.simulation.photon_step import (
    photon_iteration_sample, make_photon_iteration_update_factors_safe,
)
from lucid.simulation.reflection import get_reflection_model
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood, make_hits_moments,
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

    # ---- Select photon update function --------------------------------------
    if sim_config.is_data:
        photon_update_fn = photon_iteration_sample
    elif sim_config.use_expected_value is False:
        photon_update_fn = photon_iteration_sample
    else:
        photon_update_fn = jax.remat(
            make_photon_iteration_update_factors_safe(reflection_fn))

    # ---- Geometry bounds check (delegates to detector method) ----------------
    def get_inside_detector_flag(positions):
        return detector.bounds_check(positions)

    # ---- Resolve hit_mode ---------------------------------------------------
    _VALID_HIT_MODES = ('aggregated', 'per_photon', 'realistic', 'moments',
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
    def _make_hits_aggregated(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None):
        return make_hits_simulation(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_per_photon(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None):
        return make_hits_likelihood(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_realistic(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None):
        tts = 0.0 if response is None else response[3]
        return make_hits_data(flat_weights, flat_indices, flat_times, num_sensors,
                              qe=qe, qe_corrections=qe_corrections,
                              rng_key=qe_key, apply_smearing=sim_config.apply_smearing, tts=tts)

    def _make_hits_moments(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections, response=None):
        gain, t0, spe_width, tts = response
        return make_hits_moments(flat_weights, flat_indices, flat_times, num_sensors,
                                 qe=qe, qe_corrections=qe_corrections,
                                 gain=gain, spe_width=spe_width, t0=t0, tts=tts)

    # Shotgun hit modes (waveform + per-photon). Defaults match SK-realistic
    # PMT behaviour; override via ``waveform_config``.
    _wf_cfg = dict(window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0,
                   t_min_ns=0.0, smear_time=True, smear_charge=True)
    if waveform_config:
        _wf_cfg.update(waveform_config)

    if hit_mode == 'waveform':
        _wf_fn = build_make_hits_waveform(n_photons=n_photons, **_wf_cfg)
        def _make_hits_waveform(flat_weights, flat_indices, flat_times, num_sensors,
                                qe_key, qe, qe_corrections, response=None):
            return _wf_fn(flat_weights, flat_indices, flat_times, num_sensors,
                          qe_key, qe, qe_corrections)
    elif hit_mode == 'waveform_expected':
        # Expected-value waveform: no Bernoulli, no gain smear — those do not
        # exist when every slot deposits a continuous weight.
        _wf_exp_cfg = {k: v for k, v in _wf_cfg.items() if k != 'smear_charge'}
        _wf_exp_fn = build_make_hits_waveform_expected(
            n_photons=n_photons, **_wf_exp_cfg)
        def _make_hits_waveform_expected(flat_weights, flat_indices, flat_times, num_sensors,
                                         qe_key, qe, qe_corrections, response=None):
            return _wf_exp_fn(flat_weights, flat_indices, flat_times, num_sensors,
                              qe_key, qe, qe_corrections)
    elif hit_mode == 'shotgun_per_photon':
        _pp_fn = build_make_hits_per_photon_shotgun(
            n_photons=n_photons,
            tts_sigma_ns=_wf_cfg['tts_sigma_ns'],
            smear_time=_wf_cfg['smear_time'])
        def _make_hits_shotgun_pp(flat_weights, flat_indices, flat_times, num_sensors,
                                  qe_key, qe, qe_corrections, response=None):
            return _pp_fn(flat_weights, flat_indices, flat_times, num_sensors,
                          qe_key, qe, qe_corrections)

    _make_hits_fn = {
        'aggregated': _make_hits_aggregated,
        'per_photon': _make_hits_per_photon,
        'realistic': _make_hits_realistic,
        'moments': _make_hits_moments,
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
        _qe_fn = load_qe_curve(_qe_curve_path) if _qe_curve_path else None
    else:
        _wl_lo, _wl_hi = 300.0, 700.0
        _medium_wl = None
        _qe_fn = None

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
                    wl_key, n, lambda_min=_wl_lo, lambda_max=_wl_hi)
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

    # ================================================================
    # Core propagation (shared by all modes)
    # ================================================================

    @partial(jax.jit, static_argnames=(
        'n_rays', 'K', 'n_grad_iters', 'max_candidates_per_ray', 'num_sensors',
        'propagate_fn', 'photon_update_fn', 'pos_grad_threshold', 'make_hits_fn'))
    def _common_propagation(
            positions, directions, intensities, times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key,
            num_sensors, K, n_grad_iters, max_candidates_per_ray,
            propagate_fn, photon_update_fn,
            pos_grad_threshold, make_hits_fn):
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
            hit_times_meters = prop_results['times']
            hit_positions = prop_results['positions']
            normals = prop_results['normals']
            inside_sensor = prop_results['inside_sensor']

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
            times_ns = hit_times_meters / SPEED_OF_LIGHT_MATERIAL
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
        return make_hits_fn(
            flat_weights, flat_indices, flat_times, num_sensors, qe_key, flat_qe, qe_corrections,
            response)

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

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            final_origins, final_directions, photon_intensities, photon_times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn)

    # Load photonsim parameters from configuration (power-law normalization, SIREN path)
    photonsim_params = unpack_photonsim_params(particle, material)
    tot_n_photons_a, tot_n_photons_b, tot_n_photons_c = photonsim_params['tot_n_photons_normalization']
    num_seeds_a, num_seeds_b, num_seeds_c = photonsim_params['num_seeds']

    @jax.jit
    def tot_n_photons_normalization(x):
        """Power law: a * energy^b + c. Parameters loaded from config."""
        return tot_n_photons_a * jnp.power(x, tot_n_photons_b) + tot_n_photons_c

    @jax.jit
    def _simulation_without_data_impl(particle_params, detector_params, key, grid_data, model_params):
        """SIREN mode: particle_params is ParticleParams."""
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        key, ray_key, opt_key = jax.random.split(key, 3)
        photon_directions, photon_origins, photon_weights = photonsim_differentiable_get_rays(
            track_origin, track_direction, energy, Nphot, grid_data, model_params, ray_key,
            num_seeds_a, num_seeds_b, num_seeds_c
        )

        total_photons_norm = tot_n_photons_normalization(energy)
        photon_intensities = (total_photons_norm * photon_weights) / Nphot
        photon_times = jnp.zeros((Nphot,))

        distances_to_vertex = jnp.linalg.norm(photon_origins - track_origin, axis=1) * 1000
        # Emission-time baseline t0(distance_to_vertex, energy), DETACHED: the TIME term does not carry
        # ENERGY/VERTEX gradient through the emission-time model (those flow via geometry/charge). _T0_FORM
        # selects refactor-v2's CUBIC stretched-exp ('cubic') vs the legacy LINEAR ('linear') — a Python
        # branch on the static setup-time tag, traced out at jit.
        if _T0_FORM == 'cubic':
            a_c, l_c, b_c, c_mm = _T0_PAYLOAD
            _pt0 = jax.vmap(predict_t0_cubic, in_axes=(0, None, None, None, None, None))
            t0 = jax.lax.stop_gradient(_pt0(distances_to_vertex, energy,
                                            jnp.asarray(a_c), jnp.asarray(l_c), jnp.asarray(b_c), c_mm))
        else:
            bs, bi, As, Ai, Bs, Bi, off = _T0_PAYLOAD
            _pt0 = jax.vmap(predict_t0, in_axes=(0, None, None, None, None, None, None, None, None))
            t0 = jax.lax.stop_gradient(_pt0(distances_to_vertex, energy, bs, bi, As, Ai, Bs, Bi, off))

        # Per-photon optical properties (Cherenkov spectrum when wavelength_mode)
        scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.response.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.response.qe)

        # Track-mode position-gradient default: was 0 (always stop) as a workaround
        # for the reflection-normal curvature explosion. The normal-fix now lives
        # inside photon_iteration_update_factors, so position gradient can flow
        # all K bounces.
        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times + t0,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn)

    @jax.jit
    def _simulation_sensor_calibration_impl(source, detector_params, key):
        """Calibration mode: source is a callable (IsotropicSource or LaserSource)."""
        key, source_key, opt_key = jax.random.split(key, 3)
        photon_directions, photon_origins, photon_intensities = source(Nphot, source_key)
        photon_times = jnp.zeros((Nphot,))

        # Per-photon optical properties
        # Source wavelength can be None (→ Cherenkov), scalar, or (Nphot,) array;
        # _get_optical_arrays normalizes the shape.
        wavelengths = getattr(source, 'wavelength', None)
        scatter_lengths, mie_scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key, wavelengths=wavelengths)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.response.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.response.qe)

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            scatter_lengths, mie_scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn)

    # ---- Return the right function ------------------------------------------
    if sim_config.is_data:
        if _default_dp is not None:
            @jax.jit
            def _sim_data_default(particle_params, key, photon_data):
                return _simulation_with_data_impl(particle_params, _default_dp, key, photon_data)
            _sim_data_default.default_detector_params = _default_dp
            return _sim_data_default
        else:
            return _simulation_with_data_impl
    elif sim_config.is_calibration:
        if _default_dp is not None:
            @jax.jit
            def _sim_calibration_default(source, key):
                return _simulation_sensor_calibration_impl(source, _default_dp, key)
            _sim_calibration_default.default_detector_params = _default_dp
            return _sim_calibration_default
        else:
            return _simulation_sensor_calibration_impl
    else:
        model_base_path = photonsim_params['siren_model_path']
        photonsim_predictor = SIRENPredictor(model_base_path)
        grid_data = create_photonsim_siren_grid(photonsim_predictor)
        model_params = photonsim_predictor.params
        _T0_FORM, _T0_PAYLOAD = unpack_t0_params(particle, material)   # ('cubic'|'linear', coeffs)
        if _default_dp is not None:
            @jax.jit
            def _sim_track_default(particle_params, key):
                return _simulation_without_data_impl(particle_params, _default_dp, key,
                                                     grid_data=grid_data, model_params=model_params)
            _sim_track_default.default_detector_params = _default_dp
            return _sim_track_default
        else:
            return partial(_simulation_without_data_impl,
                           grid_data=grid_data,
                           model_params=model_params)


