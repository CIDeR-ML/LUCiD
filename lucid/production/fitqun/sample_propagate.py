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

from . import attenlength, isotropic_sample, scattable3d


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
        seed: int = 0, batch: int = 8, atten_out=None, scat3d_out=None,
        detector_type: str = "Cylinder", tts_sigma_ns: float = 1.0,
        wavelength_sampling: str = "cherenkov",
        shell_placement_cm=None, table_source: str = "parent") -> Path:
    """PhotonSim photons in, reduced shard out -- nothing in between.

    The per-photon arrays for one job are several GB. Writing them out only to
    read them back costs that much I/O per job and, on a shared filesystem with
    a few hundred jobs doing it at once, fails: EOS returned ``errno 121``
    mid-write for ~4% of a 250-job run. Folding each propagated group into the
    shard as it is produced means the big arrays never leave memory, and the
    only thing written is the shard itself.
    """
    """``table_source="parent"`` bins the scattering tables on the emitting track's start
    vertex and direction (sttree's srcpos/srcdir, what fiTQun looks the table up with);
    ``"photon"`` bins them on each photon's emission point and direction, as before.
    Propagation always uses the photon's own origin and direction. Shell-placement runs
    (whose tables are not used) always use ``"photon"``.
    """
    """``atten_out`` additionally writes the water attenuation-length histograms.

    They are kept in a side file rather than in the shard because the shard
    format is already consumed by the verified scattering-table build, and the
    per-photon distances these need exist only inside this loop.
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

    # The tune describes the light PRODUCTION writes, so this must propagate
    # photons the same way: read the one production switch rather than inherit
    # the library default, which is off.
    from lucid.production.run_job import DEPOSIT_LEG_BOUND

    sim = setup_shotgun_simulator(
        detector_config, physics_config=physics_config, n_photons=n_photons,
        output_mode="per_photon", K=K, detector_type=detector_type,
        deposit_leg_bound=DEPOSIT_LEG_BOUND,
        tts_sigma_ns=tts_sigma_ns, wavelength_sampling=wavelength_sampling)

    atten = {"all": np.zeros(attenlength.N_R_BINS),
             "direct": np.zeros(attenlength.N_R_BINS)}
    # A dict, not two names: `_flush` is nested, and an augmented assignment
    # to an enclosing name would rebind it as a local (UnboundLocalError).
    _i, _d = scattable3d.empty()
    scat3d = {"indirect": _i, "direct": _d}
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    n_groups = 0
    pending: list = []
    if table_source not in ("parent", "photon"):
        raise ValueError(f"table_source must be 'parent' or 'photon', not {table_source!r}")
    use_parent = table_source == "parent" and shell_placement_cm is None
    pending_src: list = []         # parent (pos, dir) per pending group, same order

    def _flush(pending, key):
        if not pending:
            return key
        key, sub = jax.random.split(key)
        batched = stack_shotgun_sources(pending)
        keys = jax.random.split(sub, len(pending))
        det, sid, ht, ind = sim.batch(batched, keys)
        det_np, sid_np, ind_np = np.asarray(det), np.asarray(sid), np.asarray(ind)
        origins_np = np.asarray(batched.origins)
        src_kw = {}
        if use_parent:
            src_kw = dict(source_pos_m=np.concatenate([p for p, _ in pending_src]),
                          source_dir=np.concatenate([d for _, d in pending_src]))
        builder.add(detected=det_np, sensor_id=sid_np, indirect=ind_np,
                    emission_pos_m=origins_np,
                    emission_dir=np.asarray(batched.directions), **src_kw)
        # Both side outputs need the same per-photon geometry, which exists
        # only here, so compute it once and fan out.
        if atten_out is not None or scat3d_out is not None:
            d = det_np.reshape(-1).astype(bool)
            if d.any():
                src = origins_np.reshape(-1, 3)[d] * 100.0          # cm
                sdir = np.asarray(batched.directions).reshape(-1, 3)[d]
                pmt = builder.pmt_pos_cm[sid_np.reshape(-1)[d]]
                ind = ind_np.reshape(-1)[d].astype(bool)
                R, costh, dwall = scattable3d.observables(
                    src, sdir, pmt, det_radius_cm=builder.det_radius_cm,
                    det_halfheight_cm=builder.det_halfheight_cm)
                if atten_out is not None:
                    _, ha, hd = attenlength.histograms(R, ind, dwall)
                    atten["all"] += ha
                    atten["direct"] += hd
                if scat3d_out is not None:
                    h3, h2 = scattable3d.histograms(R, costh, dwall, ind)
                    scat3d["indirect"] += h3
                    scat3d["direct"] += h2
        return key

    carry_o = np.zeros((0, 3), dtype=np.float32)
    carry_d = np.zeros((0, 3), dtype=np.float32)
    carry_po = np.zeros((0, 3), dtype=np.float32)
    carry_pd = np.zeros((0, 3), dtype=np.float32)
    # Shell placement concentrates the vertices where one angular-response shell
    # can see them, which is the only thing such a run is for: the scattering
    # tables in the same shard are then meaningless, so keep those outputs apart
    # from a volume-uniform production's.
    axes_m = None
    if shell_placement_cm is not None:
        from .angular_driver import sensor_axes
        axes_m = sensor_axes(
            builder.pmt_pos_cm,
            det_radius_cm=builder.det_radius_cm,
            det_halfheight_cm=builder.det_halfheight_cm)

    for photons in isotropic_sample.load_photons(photonsim_root, with_parent=use_parent):
        origins_m, directions, event_id = photons[:3]
        if shell_placement_cm is not None:
            origins_m = isotropic_sample.translate_shell(
                origins_m, rng, pmt_pos_m=builder.pmt_pos_cm / 100.0,
                pmt_axes=axes_m,
                shell_r_cm=float(shell_placement_cm[0]),
                shell_dr_cm=float(shell_placement_cm[1]),
                det_radius_m=float(g["radius"]),
                det_halfheight_m=float(g["height"]) / 2.0,
                pmt_radius_m=float(g["sensor_radius"]), event_id=event_id)
        else:
            # Each event gets its own vertex, as /gps/pos/type Volume does. The shift is
            # drawn once (same draws as shifting the photons directly) and applied to the
            # photons and to their parents' vertices alike.
            shift = isotropic_sample.translate_uniform(
                np.zeros_like(origins_m), rng, det_radius_m=float(g["radius"]),
                det_halfheight_m=float(g["height"]) / 2.0,
                pmt_radius_m=float(g["sensor_radius"]), event_id=event_id)
            origins_m = origins_m + shift
            if use_parent:
                carry_po = np.concatenate([carry_po, photons[3] + shift])
                carry_pd = np.concatenate([carry_pd, photons[4]])
        carry_o = np.concatenate([carry_o, origins_m])
        carry_d = np.concatenate([carry_d, directions])
        used = 0
        for o, d in _chunks(carry_o, carry_d, n_photons):
            pending.append(shotgun_source(o, d, n_photons=n_photons))
            if use_parent:
                pending_src.append((carry_po[used:used + n_photons],
                                    carry_pd[used:used + n_photons]))
            used += n_photons
            n_groups += 1
            if len(pending) == batch:
                key = _flush(pending, key)
                pending = []
                pending_src = []
        carry_o, carry_d = carry_o[used:], carry_d[used:]
        if use_parent:
            carry_po, carry_pd = carry_po[used:], carry_pd[used:]
    _flush(pending, key)

    if atten_out is not None:
        edges = np.linspace(0.0, attenlength.R_MAX_CM, attenlength.N_R_BINS + 1)
        np.savez_compressed(atten_out, edges=edges,
                            h_all=atten["all"], h_direct=atten["direct"])
        print(f"{atten_out}: attenuation histograms, "
              f"{atten['all'].sum():,.0f} photons ({atten['direct'].sum():,.0f} direct)")

    if scat3d_out is not None:
        er, ew, ec = scattable3d.edges()
        np.savez_compressed(scat3d_out, r_edges=er, wall_edges=ew, costh_edges=ec,
                            hsct3d=scat3d["indirect"], hdir2d=scat3d["direct"])
        print(f"{scat3d_out}: 3D scattering table, "
              f"{scat3d['indirect'].sum():,.0f} indirect / "
              f"{scat3d['direct'].sum():,.0f} direct")

    shard = builder.result()
    out = shard.save(shard_out)
    occ = max(c.occupancy for c in shard.scattered.values())
    print(f"{out}: {n_groups} groups x {n_photons:,} photons, "
          f"{shard.n_detected:,} detected of {shard.n_photons:,} "
          f"({shard.n_indirect:,} indirect), peak occupancy {100 * occ:.3f}%")
    return Path(out)
