"""Pipeline NamedTuples for the simulation loop.

These replace raw dicts, positional tuples, and ad-hoc returns with
structured, named containers. All are registered as JAX pytrees
(NamedTuples are pytrees by default in JAX).
"""
from typing import NamedTuple
import jax.numpy as jnp


class PhotonState(NamedTuple):
    """Carry state for the jax.lax.scan propagation loop.

    6-tuple: (positions, directions, times, survival, key, log_p).

    ``log_p`` is the per-photon accumulated DiCE score (sum of ``lf + la`` over
    prior steps); the implicit-capture deposit reads the PRE-step ``log_p``.
    """
    positions: jnp.ndarray          # (n_rays, 3)
    directions: jnp.ndarray         # (n_rays, 3)
    times: jnp.ndarray              # (n_rays,)
    survival: jnp.ndarray           # (n_rays,)
    key: jnp.ndarray                # PRNGKey
    log_p: jnp.ndarray              # (n_rays,)
