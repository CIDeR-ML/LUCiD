"""Seeds of one production job: the master seed and the PhotonSim / GENIE subprocess seeds."""
from __future__ import annotations

import time

import jax

__all__ = ["resolve_master_seed", "derive_subprocess_seeds"]

# fold_in tags of the subprocess streams; arbitrary, stable and distinct.
_SUBPROC_PHOTONSIM_TAG = 0xB107
_SUBPROC_GENIE_TAG     = 0x6E1E


def resolve_master_seed(master_seed):
    """The job's master seed as an int below 2**31 - 1; drawn from the clock when ``None``."""
    if master_seed is None:
        return int(time.time() * 1_000_000) % (2 ** 31 - 1)
    return int(master_seed) % (2 ** 31 - 1)


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
    master_seed = resolve_master_seed(master_seed)
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
