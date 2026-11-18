"""RNG seed derivation for deterministic event generation.

``derive_event_keys`` folds (master_seed, job_id, event_idx, interaction_idx) into independent
per-event streams.
"""
from __future__ import annotations

import jax

__all__ = [
    "derive_event_keys",
]


# Tag constants used with jax.random.fold_in so each stream in the seed hierarchy gets a
# distinct derivation. Values are arbitrary; they have to be stable and distinct across tags.
_SUBPROC_SCINT_TAG     = 0x5C17
_SUBPROC_ROT_TAG       = 0x4074
_SUBPROC_SEED_TAG      = 0x5EED
_SUBPROC_FIT_TAG       = 0xF17


def derive_event_keys(master_seed, job_id, event_idx, interaction_idx=0):
    """Derive independent RNG keys for one (job, event, interaction) step.

    Combines ``master_seed``, ``job_id``, ``event_idx`` and
    ``interaction_idx`` via ``jax.random.fold_in`` so every dimension is
    independent — reusing a CLI seed across jobs no longer collides, and
    pile-up interactions within one event get distinct draws.

    Returns a dict with ``vertex_seed`` / ``t0_seed`` / ``scint_seed`` /
    ``rot_seed`` (concrete ints for ``np.random.default_rng``) and
    ``sim_key`` / ``seed_key`` / ``fit_key`` (JAX keys to be consumed directly by
    ``jax.random.*``). ``seed_key`` is the reconstruction seeder's
    (``lucid.fitting.seed_track``), ``fit_key`` the track fit's (``lucid.fitting.fit_track``).
    ``scint_seed`` drives the scintillation segment-expander RNG when the
    data-mode medium has ``scintillation`` in its ``emission_processes``;
    it's derived deterministically from the same hierarchy so reruns at
    the same ``master_seed`` are byte-equivalent.
    """
    base = jax.random.PRNGKey(int(master_seed) % (2 ** 31 - 1))
    job_key = jax.random.fold_in(base, int(job_id))
    event_key = jax.random.fold_in(job_key, int(event_idx))
    interaction_key = jax.random.fold_in(event_key, int(interaction_idx))
    # A 4-way split: its fourth key has no user, but dropping it would change the other three.
    vertex_key, t0_key, sim_key, _ = jax.random.split(interaction_key, 4)
    # scint_seed is derived via a separate fold_in tag (not appended to the
    # split above) so the existing 4-way split's keys stay bit-identical for
    # Cherenkov-only datasets at the same master_seed.
    scint_key = jax.random.fold_in(interaction_key, _SUBPROC_SCINT_TAG)
    # rot_seed drives the optional LUCiD-side random rotation. Folded off its
    # own tag (not appended to the split above) so the existing keys stay
    # bit-identical at a given master_seed whether or not rotation is used.
    rot_key = jax.random.fold_in(interaction_key, _SUBPROC_ROT_TAG)
    # seed_key drives the reconstruction seeder; its own tag, for the same reason.
    seed_key = jax.random.fold_in(interaction_key, _SUBPROC_SEED_TAG)
    fit_key = jax.random.fold_in(interaction_key, _SUBPROC_FIT_TAG)
    return {
        'vertex_seed': int(jax.random.randint(vertex_key, (), 1, 2**31 - 1)),
        't0_seed':     int(jax.random.randint(t0_key,     (), 1, 2**31 - 1)),
        'scint_seed':  int(jax.random.randint(scint_key,  (), 1, 2**31 - 1)),
        'rot_seed':    int(jax.random.randint(rot_key,    (), 1, 2**31 - 1)),
        'sim_key':     sim_key,
        'seed_key':    seed_key,
        'fit_key':     fit_key,
    }
