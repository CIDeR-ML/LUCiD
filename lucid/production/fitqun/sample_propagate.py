"""Propagate the reference's electron-bomb photons through the detector.

``scattab_nuPRISM_mPMT.mac`` fires 3 MeV electrons one per event, uniformly in
the detector volume and isotropically in direction, and the tables are built
from the Cherenkov photons those electrons make. This module is the LUCiD half
of that: PhotonSim produces the photon list, and this propagates it.

Why not let the shotgun sample its own photons: it emits one direction per
case, so a case lands on a single sensor and contributes one (source, sensor)
geometry. An electron emits a Cherenkov cone across a ring of sensors, which is
where the angular response gets its independent samples. The marginal
distributions agree either way; the correlation structure does not, and it is
the correlation structure the angular response is made of.

The vertex is applied here rather than in PhotonSim: PhotonSim has no
volume-sampling command, so the gun sits at the origin and
:func:`isotropic_sample.translate_uniform` moves each *event* to its own
uniformly drawn point -- the same distribution ``/gps/pos/type Volume`` gives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from . import isotropic_sample


def _chunks(origins: np.ndarray, directions: np.ndarray,
            n_photons: int) -> Iterator[tuple]:
    """Fixed-size photon groups, since the kernel is built for one size.

    An event makes ~550 photons and the count varies, so events are streamed
    into fixed chunks rather than propagated one per call. A chunk is only a
    batching unit -- every photon keeps its own origin and direction, which is
    what makes this different from a per-case pencil beam.
    """
    n = origins.shape[0]
    for lo in range(0, n - n_photons + 1, n_photons):
        sl = slice(lo, lo + n_photons)
        yield origins[sl], directions[sl]


def propagate_and_reduce(
        photonsim_root, shard_out, *, detector_config: str, physics_config: str,
        geometry, shell_radii_cm, n_photons: int = 20000, K: int = 12,
        seed: int = 0, batch: int = 8,
        detector_type: str = "Cylinder", tts_sigma_ns: float = 1.0,
        wavelength_sampling: str = "cherenkov") -> Path:
    """PhotonSim photons in, reduced shard out -- nothing in between.

    The per-photon arrays for one job are several GB. Writing them out only to
    read them back costs that much I/O per job and, on a shared filesystem with
    a few hundred jobs doing it at once, fails: EOS returned ``errno 121``
    mid-write for ~4% of a 250-job run. Folding each propagated group into the
    shard as it is produced means the big arrays never leave memory, and the
    only thing written is the shard itself.
    """
    import jax
    from lucid.simulation.shotgun import setup_shotgun_simulator
    from lucid.sources.shotgun_source import shotgun_source, stack_shotgun_sources
    from .sample_reduce import ShardBuilder

    g = np.load(geometry)
    builder = ShardBuilder(
        pmt_positions_m=g["positions_mm"] / 1000.0,
        pmt_dir_z=g["directions"][:, 2],
        det_radius_cm=float(g["radius"]) * 100.0,
        det_halfheight_cm=float(g["height"]) * 100.0 / 2.0,
        pmt_radius_cm=float(g["sensor_radius"]) * 100.0,
        shell_radii_cm=shell_radii_cm)

    sim = setup_shotgun_simulator(
        detector_config, physics_config=physics_config, n_photons=n_photons,
        output_mode="per_photon", K=K, detector_type=detector_type,
        tts_sigma_ns=tts_sigma_ns, wavelength_sampling=wavelength_sampling)

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    n_groups = 0
    pending: list = []

    def _flush(pending, key):
        if not pending:
            return key
        key, sub = jax.random.split(key)
        batched = stack_shotgun_sources(pending)
        keys = jax.random.split(sub, len(pending))
        det, sid, ht, ind = sim.batch(batched, keys)
        builder.add(detected=np.asarray(det), sensor_id=np.asarray(sid),
                    indirect=np.asarray(ind),
                    emission_pos_m=np.asarray(batched.origins),
                    emission_dir=np.asarray(batched.directions))
        return key

    carry_o = np.zeros((0, 3), dtype=np.float32)
    carry_d = np.zeros((0, 3), dtype=np.float32)
    for origins_m, directions, event_id in isotropic_sample.load_photons(
            photonsim_root):
        # Each event gets its own vertex, as /gps/pos/type Volume does.
        origins_m = isotropic_sample.translate_uniform(
            origins_m, rng, det_radius_m=float(g["radius"]),
            det_halfheight_m=float(g["height"]) / 2.0,
            pmt_radius_m=float(g["sensor_radius"]), event_id=event_id)
        carry_o = np.concatenate([carry_o, origins_m])
        carry_d = np.concatenate([carry_d, directions])
        used = 0
        for o, d in _chunks(carry_o, carry_d, n_photons):
            pending.append(shotgun_source(o, d, n_photons=n_photons))
            used += n_photons
            n_groups += 1
            if len(pending) == batch:
                key = _flush(pending, key)
                pending = []
        carry_o, carry_d = carry_o[used:], carry_d[used:]
    _flush(pending, key)

    shard = builder.result()
    out = shard.save(shard_out)
    occ = max(c.occupancy for c in shard.scattered.values())
    print(f"{out}: {n_groups} groups x {n_photons:,} photons, "
          f"{shard.n_detected:,} detected of {shard.n_photons:,} "
          f"({shard.n_indirect:,} indirect), peak occupancy {100 * occ:.3f}%")
    return Path(out)
