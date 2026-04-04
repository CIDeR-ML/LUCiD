"""LUCiD — Light-based Unified Calibration and trackIng Differentiable simulation."""

from lucid.simulation import setup_event_simulator
from lucid.detector_params import DetectorParams, ParticleParams
from lucid.geometry import generate_detector

__all__ = [
    'setup_event_simulator',
    'DetectorParams',
    'ParticleParams',
    'generate_detector',
]
