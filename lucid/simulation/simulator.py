"""Event simulator factory (setup_event_simulator)."""
from __future__ import annotations

from lucid.sources.siren_rays import (
    make_photonsim_ray_fn,
    predict_t0,
)
from lucid.propagation.cylinder import create_photon_propagator
from lucid.propagation.sphere import create_sphere_photon_propagator
from lucid.propagation.box import create_box_photon_propagator, box_bounds_check
from lucid.geometry import generate_detector, get_material_from_config
from lucid.utils import (
    unpack_t0_params, unpack_siren_params,
    get_speed_of_light_in_material,
    spherical_to_cartesian, base_dir_path,
    smear_times, smear_charges_SK_like,
)
from lucid.detector_params import DetectorParams, ParticleParams, load_detector_params, load_physics_config
from lucid.wavelength.medium import make_medium, load_qe_curve, qe_curve_bounds
from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths, build_qe_weighted_cherenkov_sampler,
)

import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Tuple, Union
import os
from lucid.siren.core import build_photonsim_context
from functools import partial
from lucid.siren.training.inference import SIRENPredictor

from lucid.simulation.optics import (
    normalize, jax_normalize, jax_rotate_vector,
)
from lucid.simulation.photon_step import (
    photon_iteration_sample, photon_iteration_update_factors_safe,
)
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood,
    make_hits_per_photon,
    build_make_hits_waveform, build_make_hits_waveform_expected,
    build_make_hits_per_photon_shotgun,
)

# ===================================================================
# Event simulator factory
# ===================================================================

def setup_event_simulator(
        json_filename: str,
        n_photons: int = 1_000_000,
        temperature: Optional[float] = 0.2,
        K: int = 7,
        is_data: bool = False,
        is_calibration: bool = False,
        max_candidates_per_ray: int = 4,
        detector_type: str = 'Cylinder',
        use_expected_value: bool = True,
        particle: str = 'muon',
        apply_smearing: bool = True,
        physics_config: Optional[str] = None,
        default_detector_params: Union[bool, DetectorParams] = False,
        wavelength_mode: bool = True,
        hit_mode: Optional[str] = None,
        n_grad_iters: Optional[int] = None,
        pos_grad_threshold: Optional[int] = None,  # None → use mode default (calib:K, track:0)
        waveform_config: Optional[dict[str, Any]] = None,
        wavelength_sampling: str = 'cherenkov',
        **grid_params: Any) -> Callable:
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

        - ``'aggregated'`` -- differentiable soft-min per-sensor. Returns 2:
          ``(charges, times)``. Default for calibration.
        - ``'per_photon'`` -- per-photon arrays for likelihood losses. Returns 4:
          ``(log_w, times, indices, charges)``. Default for track.
        - ``'realistic'`` -- Bernoulli QE + per-photon TTS. Returns 3:
          ``(charges, times_true, times_reco)``. Default for data.
        - ``'per_segment'`` -- realistic + per-(segment, sensor) decomposition.
          Returns 6: ``(charges, times_true, times_reco, pe_per_seg,
          t_per_seg_true, t_per_seg_reco)``. Requires ``photon_segment_index``
          in photon_data.
        - ``'waveform'`` -- dense waveform histogram with Bernoulli QE + TTS +
          gain smearing. Returns 3: ``(waveform, n_dropped, n_detected)``.
        - ``'waveform_expected'`` -- continuous expected-value waveform. Returns 3:
          ``(waveform, n_dropped, n_detected)``.
        - ``'shotgun_per_photon'`` -- per-photon first-detection records. Returns 3:
          ``(detected, sensor_id, hit_time)``.
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

    Returns
    -------
    callable
        The return arity depends on ``hit_mode`` (see above). The calling
        convention depends on ``default_detector_params``:

        When ``default_detector_params`` is ``False``:

        - **Calibration** ``(source, detector_params, key) -> result``
        - **Track**       ``(particle_params, detector_params, key) -> result``
        - **Data**        ``(particle_params, detector_params, key, photon_data) -> result``

        When ``default_detector_params`` is truthy (``True`` or a ``DetectorParams``):

        - **Calibration** ``(source, key) -> result``
        - **Track**       ``(particle_params, key) -> result``
        - **Data**        ``(particle_params, key, photon_data) -> result``

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

    from lucid.geometry.string import StringTelescope
    _is_volume = isinstance(detector, StringTelescope)

    # ---- Handle qe_corrections for baked-in detector params -----------------
    if _default_dp is not None:
        qe_corr = _default_dp.qe_corrections
        # If scalar placeholder (from null in JSON), broadcast to NUM_SENSORS
        if qe_corr.ndim == 0:
            _default_dp = _default_dp._replace(qe_corrections=jnp.ones(NUM_SENSORS) * qe_corr)
        elif len(qe_corr) != NUM_SENSORS:
            raise ValueError(
                f"qe_corrections has {len(qe_corr)} elements "
                f"but detector has {NUM_SENSORS} sensors")

    # ---- Select photon update function --------------------------------------
    if sim_config.is_data:
        photon_update_fn = photon_iteration_sample
    elif sim_config.use_expected_value is False:
        photon_update_fn = photon_iteration_sample
    else:
        photon_update_fn = jax.remat(photon_iteration_update_factors_safe)

    # ---- Geometry bounds check (delegates to detector method) ----------------
    def get_inside_detector_flag(positions):
        return detector.bounds_check(positions)

    # ---- Resolve hit_mode ---------------------------------------------------
    _VALID_HIT_MODES = ('aggregated', 'per_photon', 'realistic', 'per_segment',
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
    # All wrappers accept ``flat_segment_idx`` for signature uniformity;
    # only the per_segment wrapper consumes it.
    def _make_hits_aggregated(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                              flat_segment_idx=None):
        return make_hits_simulation(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_per_photon(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                              flat_segment_idx=None):
        return make_hits_likelihood(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_realistic(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                             flat_segment_idx=None):
        return make_hits_data(flat_weights, flat_indices, flat_times, num_sensors,
                              qe=qe, qe_corrections=qe_corrections,
                              rng_key=qe_key, apply_smearing=sim_config.apply_smearing,
                              tts_sigma_ns=sim_config.tts_sigma_ns)

    def _make_hits_per_segment_fn(flat_weights, flat_indices, flat_times, num_sensors,
                                  qe_key, qe, qe_corrections,
                                  flat_segment_idx=None):
        return make_hits_per_photon(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections,
                                    rng_key=qe_key, apply_smearing=sim_config.apply_smearing,
                                    tts_sigma_ns=sim_config.tts_sigma_ns,
                                    flat_segment_idx=flat_segment_idx)

    # Shotgun hit modes (waveform + per-photon). Defaults match SK-realistic
    # PMT behaviour; override via ``waveform_config``.
    _wf_cfg = dict(window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0,
                   t_min_ns=0.0, smear_time=True, smear_charge=True)
    if waveform_config:
        _wf_cfg.update(waveform_config)

    if hit_mode == 'waveform':
        _wf_fn = build_make_hits_waveform(n_photons=n_photons, **_wf_cfg)
        def _make_hits_waveform(flat_weights, flat_indices, flat_times, num_sensors,
                                qe_key, qe, qe_corrections,
                                flat_segment_idx=None):
            return _wf_fn(flat_weights, flat_indices, flat_times, num_sensors,
                          qe_key, qe, qe_corrections)
    elif hit_mode == 'waveform_expected':
        _wf_exp_cfg = {k: v for k, v in _wf_cfg.items() if k != 'smear_charge'}
        _wf_exp_fn = build_make_hits_waveform_expected(
            n_photons=n_photons, **_wf_exp_cfg)
        def _make_hits_waveform_expected(flat_weights, flat_indices, flat_times, num_sensors,
                                         qe_key, qe, qe_corrections,
                                         flat_segment_idx=None):
            return _wf_exp_fn(flat_weights, flat_indices, flat_times, num_sensors,
                              qe_key, qe, qe_corrections)
    elif hit_mode == 'shotgun_per_photon':
        _pp_fn = build_make_hits_per_photon_shotgun(
            n_photons=n_photons,
            tts_sigma_ns=_wf_cfg['tts_sigma_ns'],
            smear_time=_wf_cfg['smear_time'])
        def _make_hits_shotgun_pp(flat_weights, flat_indices, flat_times, num_sensors,
                                  qe_key, qe, qe_corrections,
                                  flat_segment_idx=None):
            return _pp_fn(flat_weights, flat_indices, flat_times, num_sensors,
                          qe_key, qe, qe_corrections)

    _make_hits_fn = {
        'aggregated': _make_hits_aggregated,
        'per_photon': _make_hits_per_photon,
        'realistic': _make_hits_realistic,
        'per_segment': _make_hits_per_segment_fn,
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
            return (jnp.full(n, detector_params.scatter_length),
                    jnp.full(n, detector_params.absorption_length),
                    None, key)

        # Normalize wavelength input to per-photon array:
        #   None   → sample Cherenkov spectrum (Method A or B depending on
        #            wavelength_sampling)
        #   scalar → broadcast to (n,)
        #   (n,)   → use as-is
        sampled_via_qe_importance = False
        if wavelengths is None:
            key, wl_key = jax.random.split(key)
            if wavelength_sampling == 'cherenkov_qe':
                wavelengths = _qe_sampler(wl_key, n)
                sampled_via_qe_importance = True
            else:
                wavelengths = sample_cherenkov_wavelengths(
                    wl_key, n, lambda_min=_wl_lo, lambda_max=_wl_hi)
        else:
            wavelengths = jnp.asarray(wavelengths)
            if wavelengths.ndim == 0:
                wavelengths = jnp.full(n, wavelengths)

        wavelengths = jnp.clip(wavelengths,
                               _medium_wl.wavelength_grid[0],
                               _medium_wl.wavelength_grid[-1])
        sc = jnp.interp(wavelengths, _medium_wl.wavelength_grid, _medium_wl.scatter_coeff)
        ac = jnp.interp(wavelengths, _medium_wl.wavelength_grid, _medium_wl.absorption_coeff)
        scatter_lengths = 1.0 / (sc + 1e-30)
        absorption_lengths = 1.0 / (ac + 1e-30)

        # QE-weight convention:
        #   • Method B sampled here  → the λ-dependence of QE is already in
        #     the sampling distribution, so the per-photon weight collapses
        #     to the scalar <QE>_C.
        #   • Otherwise (explicit wavelengths, PhotonSim data, or Method A):
        #     the per-photon weight must include qe_fn(λ).
        if sampled_via_qe_importance:
            qe_weights = jnp.full(n, _mean_qe_c)
        elif _qe_fn is not None:
            qe_weights = _qe_fn(wavelengths)
        else:
            qe_weights = None
        return scatter_lengths, absorption_lengths, qe_weights, key

    # ================================================================
    # Core propagation (shared by all modes)
    # ================================================================

    @partial(jax.jit, static_argnames=(
        'n_rays', 'K', 'n_grad_iters', 'max_candidates_per_ray', 'num_sensors',
        'propagate_fn', 'photon_update_fn', 'pos_grad_threshold', 'make_hits_fn',
        'is_volume'))
    def _common_propagation(
            positions, directions, intensities, times,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key,
            num_sensors, K, n_grad_iters, max_candidates_per_ray,
            propagate_fn, photon_update_fn,
            pos_grad_threshold, make_hits_fn,
            segment_idx=None,
            is_volume=False):
        """Core photon propagation loop.

        Parameters
        ----------
        scatter_lengths : jnp.ndarray
            Per-photon scattering lengths, shape (n_rays,).
        absorption_lengths : jnp.ndarray
            Per-photon absorption lengths, shape (n_rays,).
        qe_per_photon : jnp.ndarray
            Per-photon quantum efficiency, shape (n_rays,).
        pos_grad_threshold : int
            Iteration threshold for position stop_gradient.
        make_hits_fn : callable
            Sensor response aggregation function.
        segment_idx : jnp.ndarray or None
            Per-photon segment index (n_rays,). Only used by per_segment mode.
        """

        wall_reflection_rate = detector_params.wall_reflection_rate
        sensor_reflection_rate = detector_params.sensor_reflection_rate
        qe_corrections = detector_params.qe_corrections
        if qe_corrections.ndim == 0:
            qe_corrections = jnp.ones(num_sensors) * qe_corrections

        from lucid.simulation.types import PhotonState

        # Photons originating outside the detector must not contribute
        # weight at step 0. The propagation loop zeroes continuing_factors
        # for outside photons, but only after iteration 0's weights have
        # been emitted — so mirror the inside-detector mask here.
        initial_survival = get_inside_detector_flag(positions).astype(jnp.float32)

        def propagation_step(carry, i):
            state = carry
            key, prop_key = jax.random.split(state.key)

            # Block per-photon gradient through propagate_fn for photons whose
            # origin is OUTSIDE the cylinder. The cylinder ray-trace's auto-
            # derived backward has undefined Jacobian for "outside" rays
            # (sqrt+where pattern in the barrel intersection produces inf
            # intermediates that combine with zero downstream cotangents to
            # give 0×inf=NaN at the track parameters).
            #
            # jnp.where backward *selects* between cotangents (no multiplication),
            # so an outside photon's NaN cotangent from propagate_fn is replaced
            # with 0 rather than multiplied with it — avoiding 0×NaN=NaN. The
            # forward value is unchanged: both branches numerically equal
            # state.positions/state.directions.
            inside_flag = jax.lax.stop_gradient(get_inside_detector_flag(state.positions))
            safe_positions  = jnp.where(inside_flag[:, None],
                                        state.positions,
                                        jax.lax.stop_gradient(state.positions))
            safe_directions = jnp.where(inside_flag[:, None],
                                        state.directions,
                                        jax.lax.stop_gradient(state.directions))

            prop_results = propagate_fn(safe_positions, safe_directions)
            depositions = prop_results['sensor_weights']
            sensor_indices = prop_results['sensor_indices']
            sensor_distances = prop_results['sensor_distances']
            hit_positions = prop_results['positions']
            normals = prop_results['normals']
            inside_sensor = prop_results['inside_sensor']

            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            if is_volume:
                # ── Volume model: per-DOM survival, no reflection ──
                from lucid.simulation.photon_step_volume import photon_step_volume
                seg_lengths = prop_results.get(
                    'envelope_exit_t',
                    jnp.linalg.norm(hit_positions - state.positions, axis=1))
                seg_lengths = jnp.maximum(seg_lengths, 1.0)

                (new_positions, new_directions, new_times,
                 per_dom_charges, continuing_factors) = jax.vmap(
                    photon_step_volume,
                    in_axes=(0, 0, 0, 1, 1, 0, 0, 0, 0, None)
                )(state.positions, state.directions, state.times,
                  sensor_distances.squeeze(-1), depositions,
                  scatter_lengths, absorption_lengths, seg_lengths,
                  rng_keys, SPEED_OF_LIGHT_MATERIAL)

                inside_detector = get_inside_detector_flag(new_positions)
                safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)
                new_survival = state.survival * safe_continuing

                physical_intensities = intensities * state.survival
                updated_weights = per_dom_charges.T * physical_intensities[None, :]
            else:
                # ── Surface model: shared surface_distance, reflection ──
                hit_sensor = jnp.max(inside_sensor, axis=0)
                surface_distances = jnp.linalg.norm(hit_positions - state.positions, axis=1) - 1e-6

                (new_positions, new_directions, new_times,
                 detect_probs, reflection_attenuations,
                 continuing_factors) = jax.vmap(
                    photon_update_fn,
                    in_axes=(0, 0, 0, 0, 0,
                             0, None, None, 0,
                             0, 0, None)
                )(state.positions, state.directions, state.times,
                  surface_distances, normals,
                  scatter_lengths, wall_reflection_rate, sensor_reflection_rate,
                  absorption_lengths,
                  hit_sensor, rng_keys, SPEED_OF_LIGHT_MATERIAL)

                inside_detector = get_inside_detector_flag(new_positions)
                safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)
                new_survival = state.survival * safe_continuing

                physical_intensities = intensities * state.survival
                detected_factors = detect_probs * reflection_attenuations
                updated_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
            sensor_times_ns = sensor_distances / SPEED_OF_LIGHT_MATERIAL
            total_times = sensor_times_ns + state.times[:, None]

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

            new_state = PhotonState(
                positions=next_pos,
                directions=next_dir,
                times=new_times,
                survival=new_survival,
                key=key,
            )
            outputs = (iter_weights, iter_indices, iter_times)
            return new_state, outputs

        init_state = PhotonState(
            positions=positions,
            directions=directions,
            times=times,
            survival=initial_survival,
            key=key,
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

        # Per-photon segment id, broadcast to flat shape via the same trick.
        # None when no segment decomposition is requested (every mode except
        # 'per_segment'); make_hits_fn dispatches accordingly.
        flat_segment_idx = (segment_idx[photon_idx]
                             if segment_idx is not None else None)

        key, qe_key = jax.random.split(key)
        return make_hits_fn(
            flat_weights, flat_indices, flat_times, num_sensors,
            qe_key, flat_qe, qe_corrections,
            flat_segment_idx=flat_segment_idx)

    # ================================================================
    # Mode-specific simulation functions
    # ================================================================

    @jax.jit
    def _simulation_with_data_impl(particle_params, detector_params, key, photon_data):
        """Data mode: photons from ROOT/PhotonSim files, particle_params is ParticleParams.

        When ``photon_data`` carries ``'photon_segment_index'`` (per-photon
        int32 array, padded with -1 sentinels), the per_segment hit-mode
        wrapper passes the per-photon flat lists back to the host so it
        can build the per-(segment, sensor) decomposition there.
        """
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        photon_origins = photon_data['photon_origins']  # already in m
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
        scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            n_rays, detector_params, key, wavelengths=data_wavelengths)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.qe
        else:
            qe_per_photon = jnp.full(n_rays, detector_params.qe)

        # Optional per-photon segment id (per_segment mode). Other modes
        # don't pass this — segment_idx stays None and the per_segment
        # wrapper's pass-through is unused.
        segment_idx = photon_data.get('photon_segment_index', None)

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            final_origins, final_directions, photon_intensities, photon_times,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn,
            segment_idx=segment_idx, is_volume=_is_volume)

    # Load SIREN inference parameters (model path + ray-sampling knobs).
    siren_params = unpack_siren_params(particle, material)

    @jax.jit
    def _simulation_without_data_impl(particle_params, detector_params, key):
        """SIREN track mode: particle_params is ParticleParams.

        Closes over `ray_fn`, `model_params` and `t0_params`, which are
        assigned in the track-mode branch below (late binding — this impl is
        only ever called through that branch's return)."""
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        key, ray_key, opt_key = jax.random.split(key, 3)
        # ray_fn returns photon_intensities already normalised so that
        # sum(intensities) == N_photons(energy) — no separate rescaling.
        photon_directions, photon_origins, photon_intensities = ray_fn(
            track_origin, track_direction, energy, Nphot, model_params, ray_key)
        photon_times = jnp.zeros((Nphot,))

        distances_to_vertex = jnp.linalg.norm(photon_origins - track_origin, axis=1) * 1000
        predict_t0_vec = jax.vmap(predict_t0, in_axes=(0, None, None, None, None, None, None, None, None))
        baseline_slope, baseline_intercept, A_slope, A_intercept, B_slope, B_intercept, offset = t0_params
        t0 = jax.lax.stop_gradient(
            predict_t0_vec(distances_to_vertex, energy,
                           baseline_slope, baseline_intercept,
                           A_slope, A_intercept,
                           B_slope, B_intercept, offset))

        # Per-photon optical properties (Cherenkov spectrum when wavelength_mode)
        scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.qe)

        # Track-mode position-gradient default: was 0 (always stop) as a workaround
        # for the reflection-normal curvature explosion. The normal-fix now lives
        # inside photon_iteration_update_factors, so position gradient can flow
        # all K bounces.
        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times + t0,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn,
            is_volume=_is_volume)

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
        scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key, wavelengths=wavelengths)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.qe)

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_candidates_per_ray,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn,
            is_volume=_is_volume)

    # ---- Return the right function ------------------------------------------
    if sim_config.is_data:
        if _default_dp is not None:
            @jax.jit
            def _sim_data_default(particle_params, key, photon_data):
                return _simulation_with_data_impl(
                    particle_params, _default_dp, key, photon_data)
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
        model_base_path = siren_params['siren_model_path']
        photonsim_predictor = SIRENPredictor(model_base_path)
        ctx = build_photonsim_context(photonsim_predictor, siren_params['ray_sampling'])
        ray_fn = make_photonsim_ray_fn(ctx)
        model_params = photonsim_predictor.params
        t0_params = unpack_t0_params(particle, material)
        if _default_dp is not None:
            @jax.jit
            def _sim_track_default(particle_params, key):
                return _simulation_without_data_impl(particle_params, _default_dp, key)
            _sim_track_default.default_detector_params = _default_dp
            return _sim_track_default
        else:
            return _simulation_without_data_impl
