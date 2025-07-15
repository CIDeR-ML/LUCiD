"""
SIREN (Sinusoidal Representation Networks) module for photon simulation.

This module provides:
- Core SIREN model implementation
- Training utilities for PhotonSim data
- Validation tools for trained models
"""

from .core import SIREN, SineLayer

__all__ = ['SIREN', 'SineLayer']