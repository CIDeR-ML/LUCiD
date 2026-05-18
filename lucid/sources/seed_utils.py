"""RNG seed derivation utilities for deterministic event generation.

Provides ``derive_event_keys`` and ``derive_subprocess_seeds`` which
fold master_seed, job_id, event_idx, and interaction_idx into
independent JAX PRNG streams.  Also exports the ``T0_HALF_WINDOW_NS``
constant consumed by event generation drivers.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp

__all__ = [
    "T0_HALF_WINDOW_NS",
    "derive_event_keys",
    "derive_subprocess_seeds",
]


# Tag constants used with jax.random.fold_in so each subprocess stream in
# the seed hierarchy gets a distinct derivation. Value is arbitrary — it
# just has to be stable and distinct across tags.
_SUBPROC_PHOTONSIM_TAG = 0xB107
_SUBPROC_GENIE_TAG     = 0x6E1E

# t0 draw half-window (ns). Applied symmetrically per interaction:
# t0 ~ Uniform(-T0_HALF_WINDOW_NS, +T0_HALF_WINDOW_NS). Wide enough to
# (a) randomize absolute event time so downstream models can't assume
# t=0 is the true start, and (b) cover a ±250 ns pile-up window.
T0_HALF_WINDOW_NS = 250.0


def _resolve_master_seed(master_seed):
    """Return a deterministic int seed, drawing from time if master_seed is None."""
    if master_seed is None:
        return int(time.time() * 1_000_000) % (2 ** 31 - 1)
    return int(master_seed) % (2 ** 31 - 1)


def derive_event_keys(master_seed, job_id, event_idx, interaction_idx=0):
    """Derive independent RNG keys for one (job, event, interaction) step.

    Combines ``master_seed``, ``job_id``, ``event_idx`` and
    ``interaction_idx`` via ``jax.random.fold_in`` so every dimension is
    independent — reusing a CLI seed across jobs no longer collides, and
    pile-up interactions within one event get distinct draws.

    Returns a dict with ``vertex_seed`` / ``t0_seed`` (concrete ints for
    ``np.random.default_rng``) and ``sim_key`` / ``smear_key`` (JAX keys
    to be consumed directly by ``jax.random.*``).
    """
    master_seed = _resolve_master_seed(master_seed)
    base = jax.random.PRNGKey(master_seed)
    job_key = jax.random.fold_in(base, int(job_id))
    event_key = jax.random.fold_in(job_key, int(event_idx))
    interaction_key = jax.random.fold_in(event_key, int(interaction_idx))
    vertex_key, t0_key, sim_key, smear_key = jax.random.split(interaction_key, 4)
    return {
        'vertex_seed': int(jax.random.randint(vertex_key, (), 1, 2**31 - 1)),
        't0_seed':     int(jax.random.randint(t0_key,     (), 1, 2**31 - 1)),
        'sim_key':     sim_key,
        'smear_key':   smear_key,
    }


def derive_subprocess_seeds(master_seed, job_id, vertex_idx=0):
    """Derive deterministic seeds for the per-job subprocesses (GENIE, PhotonSim).

    Subprocess seeds are folded at the (master_seed, job_id, vertex_idx)
    level — not per-event — because each subprocess produces all
    ``n_events`` internally and drives its own per-event RNG. The
    ``vertex_idx`` axis exists so pile-up configurations with N
    PhotonSim/GENIE streams per event get independent seeds per stream.

    PhotonSim's Geant4/CLHEP engine needs two seeds (`/random/setSeeds
    s1 s2`); GENIE's gevgen takes one.
    """
    master_seed = _resolve_master_seed(master_seed)
    base = jax.random.PRNGKey(master_seed)
    job_key = jax.random.fold_in(base, int(job_id))
    vertex_key = jax.random.fold_in(job_key, int(vertex_idx))
    genie_key = jax.random.fold_in(vertex_key, _SUBPROC_GENIE_TAG)
    ps_root = jax.random.fold_in(vertex_key, _SUBPROC_PHOTONSIM_TAG)
    ps_key1, ps_key2 = jax.random.split(ps_root, 2)
    return {
        'genie_seed':      int(jax.random.randint(genie_key, (), 1, 2**31 - 1)),
        'photonsim_seed1': int(jax.random.randint(ps_key1,   (), 1, 2**31 - 1)),
        'photonsim_seed2': int(jax.random.randint(ps_key2,   (), 1, 2**31 - 1)),
    }
