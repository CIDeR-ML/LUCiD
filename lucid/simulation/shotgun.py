"""Photon-shotgun setup — thin wrapper over ``setup_event_simulator``.

Supplies shotgun-appropriate defaults (hard sensor kernel, MC sampling, K=12)
and attaches a ``.batch`` helper for N-case ``vmap`` execution. All real work
happens inside ``setup_event_simulator`` via the ``hit_mode='waveform'`` and
``hit_mode='shotgun_per_photon'`` paths.
"""
from typing import Optional

import jax

from lucid.simulation.simulator import setup_event_simulator


def setup_shotgun_simulator(
    json_filename: str,
    *,
    physics_config: str,
    n_photons: int,
    output_mode: str = 'waveform',
    K: int = 12,
    temperature: Optional[float] = None,
    max_sensors_per_cell: int = 4,
    detector_type: str = 'Cylinder',
    window_ns: float = 500.0,
    bin_width_ns: float = 1.0,
    tts_sigma_ns: float = 1.0,
    t_min_ns: float = 0.0,
    smear_time: bool = True,
    smear_charge: bool = True,
    default_detector_params: bool = True,
    **grid_params,
):
    """Build a photon-shotgun simulator.

    Parameters
    ----------
    json_filename, physics_config : str
        Detector geometry and physics config JSONs.
    n_photons : int
        Photons per case (baked into the JIT cache key).
    output_mode : {'waveform', 'per_photon'}
        ``'waveform'`` → dense ``(num_sensors, n_time_bins)`` + drop/detected counters.
        ``'per_photon'`` → ``(detected, sensor_id, hit_time)`` arrays length n_photons.
    K : int
        Max scattering iterations; default 12 (covers tail for all detectors).
    temperature : float or None
        Propagator sensor kernel. ``None`` → step (hard edges); shotgun default.
    window_ns, bin_width_ns, tts_sigma_ns, t_min_ns : float
        Waveform binning + TTS smearing. Also applies to per-photon (tts only).
    smear_time, smear_charge : bool
        Toggle per-photon Gaussian TTS and SK-like gain smearing.
    default_detector_params : bool
        If True, bake DetectorParams from physics_config into the closure —
        returned callable is ``sim(source, key)``. Otherwise ``sim(source, dp, key)``.

    Returns
    -------
    callable
        Jitted simulator. In addition to being callable per-case, it exposes
        a ``.batch`` attribute that vmaps over a leading case axis.
    """
    _HIT_MODES = {'waveform': 'waveform', 'per_photon': 'shotgun_per_photon'}
    if output_mode not in _HIT_MODES:
        raise ValueError(
            f"output_mode must be 'waveform' or 'per_photon', got {output_mode!r}")
    hit_mode = _HIT_MODES[output_mode]

    waveform_config = dict(
        window_ns=window_ns, bin_width_ns=bin_width_ns,
        tts_sigma_ns=tts_sigma_ns, t_min_ns=t_min_ns,
        smear_time=smear_time, smear_charge=smear_charge,
    )

    sim = setup_event_simulator(
        json_filename,
        n_photons=n_photons,
        temperature=temperature,
        K=K,
        is_calibration=True,
        max_sensors_per_cell=max_sensors_per_cell,
        detector_type=detector_type,
        use_expected_value=False,   # MC sampling (binary detection)
        physics_config=physics_config,
        default_detector_params=default_detector_params,
        wavelength_mode=True,
        hit_mode=hit_mode,
        waveform_config=waveform_config,
        **grid_params,
    )

    # Attach geometry metadata so callers/notebooks can introspect without
    # rebuilding the detector. Cheap: generate_detector just parses JSON.
    from lucid.geometry import generate_detector
    _detector = generate_detector(json_filename)
    sim.detector = _detector
    sim.num_sensors = len(_detector.all_points)
    sim.sensor_points = _detector.all_points

    # Metadata for callers (e.g. IO helpers).
    sim.output_mode = output_mode
    sim.n_photons = n_photons
    sim.waveform_config = waveform_config
    sim.K = K

    # Batched variant: vmap over a leading case axis. Signature matches the
    # underlying sim (with or without baked-in detector_params).
    if default_detector_params:
        sim.batch = jax.jit(jax.vmap(sim, in_axes=(0, 0)))
    else:
        sim.batch = jax.jit(jax.vmap(sim, in_axes=(0, None, 0)))

    return sim
