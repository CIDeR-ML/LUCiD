"""
SIREN (Sinusoidal Representation Networks) module for photon simulation.

This module provides:
- Core SIREN model implementation
- Training utilities for PhotonSim data
- Validation tools for trained models
"""

from .core import (
    SIREN, SineLayer, SirenContext,
    build_cherenkov_context, build_dedx_context,
)

__all__ = [
    'SIREN', 'SineLayer', 'SirenContext',
    'build_cherenkov_context', 'build_dedx_context',
]