"""Photon-shotgun setup — thin wrapper over ``setup_event_simulator``.

Supplies shotgun-appropriate defaults (hard sensor kernel, MC sampling, K=12)
and returns it with a ``.batch`` helper for N-case ``vmap`` execution. All real work
happens inside ``setup_event_simulator`` with ``hits='waveform'`` or ``hits='photon_hits'``.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import jax

from lucid.simulation.simulator import Simulator, setup_event_simulator


@dataclass(frozen=True, eq=False)
class ShotgunSimulator:
    """A shotgun ``Simulator`` with its batched variant and readout layout. Calling it calls ``sim``."""
    sim: Simulator
    batch: Callable             # jit(vmap(sim)) over a leading case axis
    num_sensors: int
    waveform_config: dict

    def __call__(self, *args, **kwargs):
        return self.sim(*args, **kwargs)


def setup_shotgun_simulator(
    json_filename: str,
    *,
    physics_config: str,
    n_photons: int,
    hits: str = 'waveform',
    transport: str = 'sampled',
    detection: str = 'sampled',
    K: int = 12,
    temperature: Optional[float] = None,
    window_ns: float = 500.0,
    bin_width_ns: float = 1.0,
    tts_sigma_ns: float = 1.0,
    t_min_ns: float = 0.0,
    smear_time: bool = True,
    smear_charge: bool = True,
    default_detector_params: bool = True,
    wavelength_sampling: str = 'cherenkov',
    **grid_params,
):
    """Build a photon-shotgun simulator.

    Parameters
    ----------
    json_filename, physics_config : str
        Detector geometry and physics config JSONs.
    n_photons : int
        Photons per case (baked into the JIT cache key).
    hits : {'waveform', 'photon_hits'}
        ``'waveform'`` → dense ``(num_sensors, n_time_bins)`` + drop/detected counters.
        ``'photon_hits'`` → ``(detected, sensor_id, hit_time, indirect)`` arrays length n_photons.
    transport, detection : {'expected', 'sampled'}
        ``'sampled'`` (default) runs the Bernoulli MC propagator and Bernoulli QE, producing
        literal per-shot realizations. ``'expected'`` runs the expected-value propagator and a
        continuous QE weight: the expected waveform (same shape) with substantially lower MC
        noise at fixed photon count. ``'photon_hits'`` supports only ``detection='sampled'``
        (each photon has multiple continuous slots in the expected propagation).
    K : int
        Max scattering iterations; default 12 (covers tail for all detectors).
    temperature : float or None
        Propagator sensor kernel. ``None`` → step (hard edges); shotgun default.
    window_ns, bin_width_ns, tts_sigma_ns, t_min_ns : float
        Waveform binning + TTS smearing. Also applies to per-photon (tts only).
    smear_time, smear_charge : bool
        Toggle per-photon Gaussian TTS and SK-like gain smearing.
    wavelength_sampling : {'cherenkov', 'cherenkov_qe'}
        ``'cherenkov'`` (default) samples λ from 1/λ² and applies QE(λ)
        per-photon. ``'cherenkov_qe'`` importance-samples λ ∝ QE(λ)/λ²
        so the per-photon QE weight collapses to the scalar <QE>_C —
        strictly lower variance at fixed photon count, but the output
        becomes a density estimate rather than a literal per-shot
        realization (Binomial fluctuations get suppressed).
    default_detector_params : bool
        If True, bake DetectorParams from physics_config into the closure —
        returned callable is ``sim(source, key)``. Otherwise ``sim(source, dp, key)``.

    Returns
    -------
    ShotgunSimulator
        Callable per case; ``.batch`` vmaps over a leading case axis.
    """
    if hits not in ('waveform', 'photon_hits'):
        raise ValueError(f"hits must be 'waveform' or 'photon_hits', got {hits!r}")

    waveform_config = dict(
        window_ns=window_ns, bin_width_ns=bin_width_ns,
        tts_sigma_ns=tts_sigma_ns, t_min_ns=t_min_ns,
        smear_time=smear_time, smear_charge=smear_charge,
    )
    # Each hits function takes only its own options: the per-photon records the TTS alone, the
    # expected waveform no charge smearing.
    if hits == 'photon_hits':
        hits_config = dict(tts_sigma_ns=tts_sigma_ns, smear_time=smear_time)
    elif detection == 'expected':
        hits_config = {k: v for k, v in waveform_config.items() if k != 'smear_charge'}
    else:
        hits_config = waveform_config

    sim = setup_event_simulator(
        json_filename,
        n_photons=n_photons,
        temperature=temperature,
        K=K,
        source='light_source',
        transport=transport,
        detection=detection,
        hits=hits,
        hits_config=hits_config,
        physics_config=physics_config,
        default_detector_params=default_detector_params,
        wavelength_mode=True,
        wavelength_sampling=wavelength_sampling,
        **grid_params,
    )

    # Batched variant: vmap over a leading case axis. Signature matches the
    # underlying sim (with or without baked-in detector_params).
    if default_detector_params:
        batch = jax.jit(jax.vmap(sim, in_axes=(0, 0)))
    else:
        batch = jax.jit(jax.vmap(sim, in_axes=(0, None, 0)))

    return ShotgunSimulator(sim, batch, sim.det_geom.num_sensors, waveform_config)
