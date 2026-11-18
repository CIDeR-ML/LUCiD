"""LUCiD propagation: per-geometry ray tracing.

The surface geometries (cylinder, sphere, box) share one propagator, built by
`lucid.propagation.shared.create_propagator`; string telescopes have their own,
`lucid.propagation.string.string_propagator.create_string_propagator`.
"""
from . import cylinder, sphere, box

__all__ = ['cylinder', 'sphere', 'box']
