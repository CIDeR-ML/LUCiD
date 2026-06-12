"""LUCiD — Light-based Unified Calibration and trackIng Differentiable simulation."""

from lucid.simulation import setup_event_simulator, make_sim_pair
from lucid.detector_params import DetectorParams, ParticleParams, JointParams
from lucid.geometry import generate_detector

__all__ = [
    'setup_event_simulator',
    'make_sim_pair',
    'DetectorParams',
    'ParticleParams',
    'JointParams',
    'generate_detector',
]
