"""Per-step bench of the production data-mode pipeline.

Mirrors the per-event loop in lucid/sources/event_io.py:952-1357 and
times each substep in isolation with warmup + block_until_ready, so the
numbers reflect the real cost of each step rather than where Python
happens to demand the bytes.

Usage (inside the lucid:latest container):

    apptainer/docker run ... lucid:latest \\
        python3 /opt/LUCiD/ci_tests/production_pipeline_perf.py \\
            --root-file /out/output_job_000001.root \\
            --event-idx 0 --warmup 3 --runs 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import uproot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
from lucid.sources.event_io import (
    read_particle_data_from_photonsim,
    get_max_photons_per_particle,
)
from lucid.sources.v3_writer import (
    build_interaction_metadata,
    _compute_contained,
)
from lucid.utils import smear_charges_SK_like, smear_times


def _block(out):
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        out,
    )


def bench(fn, *args, warmup, runs, block=False):
    for _ in range(warmup):
        out = fn(*args)
        if block:
            _block(out)
    times = []
    for _ in range(runs):
        s = time.perf_counter()
        out = fn(*args)
        if block:
            _block(out)
        times.append(time.perf_counter() - s)
    return out, np.array(times)


def fmt(times: np.ndarray) -> str:
    a = 1000.0 * times
    return (f"mean={a.mean():8.2f} ms   std={a.std():7.2f}   "
            f"min={a.min():8.2f}   max={a.max():8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-file", required=True, type=Path)
    # Match production defaults from lucid/production/run_job.py:189-190.
    ap.add_argument("--detector-json", default=str(REPO / "config" / "SK_like_geom_config.json"))
    ap.add_argument("--physics-json",  default=str(REPO / "config" / "SK_like_physics_config.json"))
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--event-idx", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs",   type=int, default=10)
    ap.add_argument("--pad-size", type=int, default=None,
                    help="Override PAD_SIZE (the kernel-input photon "
                         "axis length). Default: max photons per particle "
                         "across the whole ROOT file (production behavior). "
                         "If smaller than the chosen event's actual photon "
                         "count, the per-particle photon arrays are "
                         "truncated to fit (a warning is printed).")
    args = ap.parse_args()

    print(f"jax devices:   {jax.devices()}")
    print(f"root file:     {args.root_file}")
    print(f"event_idx:     {args.event_idx}")
    print(f"K:             {args.K}")
    print(f"warmup x runs: {args.warmup} x {args.runs}")
    print()

    # === PAD_SIZE scan === (matches event_io.py:879-884 production logic)
    with uproot.open(str(args.root_file)) as f:
        n_events_in_file = int(f["OpticalPhotons"].num_entries)
    max_photons_per_particle = get_max_photons_per_particle(str(args.root_file))
    auto_pad = max_photons_per_particle + 1
    PAD_SIZE = args.pad_size if args.pad_size is not None else auto_pad
    print(f"n_events:                 {n_events_in_file}")
    print(f"max photons / particle:   {max_photons_per_particle:,}")
    print(f"PAD_SIZE (auto):          {auto_pad:,}")
    if args.pad_size is not None:
        print(f"PAD_SIZE (override):      {PAD_SIZE:,}  <-- using this")
    print()

    # === Simulator setup ===
    t0 = time.perf_counter()
    event_simulator = setup_event_simulator(
        args.detector_json, 0, K=args.K,
        is_data=True, temperature=0.0,
        apply_smearing=False,
        physics_config=args.physics_json,
        default_detector_params=True,
    )
    print(f"setup_event_simulator (Python only, no JIT yet): "
          f"{time.perf_counter() - t0:.3f}s")
    print()

    EVENT_IDX = args.event_idx
    WARMUP, RUNS = args.warmup, args.runs

    # === Step 1 — root_read+categorize ===
    # Post-Stage-5a/Stage-6, this step covers ROOT decompression + Python
    # categorization (categorize_event + derive_meaningful_tracks +
    # filter_segments_to_meaningful + assign_group_ids + photon→particle
    # bucketing). The categorization piece is pure-Python over per-event
    # arrays, so its cost scales with n_tracks and n_segments.
    def step_root_read():
        return read_particle_data_from_photonsim(
            str(args.root_file), EVENT_IDX,
        )
    particle_data, t_root = bench(step_root_read, warmup=WARMUP, runs=RUNS)
    n_particles = particle_data["n_particles"]
    total_photons = len(particle_data["photon_origins"])
    print(f"event {EVENT_IDX}: n_particles={n_particles}  total_photons={total_photons:,}")
    print(f"root_read+cat : {fmt(t_root)}")

    # === Step 2 — preprocess (NumPy scatter/pad) ===
    default_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def step_preprocess():
        particles = particle_data["particles"]
        n_p = particle_data["n_particles"]
        all_o = particle_data["photon_origins"].astype(np.float32, copy=False)
        all_d = particle_data["photon_directions"].astype(np.float32, copy=False)
        all_t = particle_data["photon_times"].astype(np.float32, copy=False)
        all_w = particle_data["photon_wavelengths"].astype(np.float32, copy=False)

        bo = np.zeros((n_p, PAD_SIZE, 3), dtype=np.float32)
        bd = np.tile(default_direction, (n_p, PAD_SIZE, 1))
        bt = np.zeros((n_p, PAD_SIZE), dtype=np.float32)
        bw = np.zeros((n_p, PAD_SIZE), dtype=np.float32)
        Np = np.zeros(n_p, dtype=np.int32)
        te = np.zeros(n_p, dtype=np.float32)
        tp = np.zeros((n_p, 3), dtype=np.float32)
        td = np.zeros((n_p, 3), dtype=np.float32)
        for i, p in enumerate(particles):
            idx = p["photon_indices"]; N = len(idx)
            # Truncate if PAD_SIZE was overridden to something smaller than
            # the actual photon count. The kernel only uses the first Np[i]
            # slots, so this lets us bench at any PAD_SIZE we want.
            N_use = min(N, PAD_SIZE)
            Np[i] = N_use
            ti = p["track_info"]
            if ti is not None:
                te[i] = ti["energy"]; tp[i] = ti["position"]; td[i] = ti["direction"]
            else:
                te[i] = particle_data["primary_energy"]; td[i] = [0.0, 0.0, 1.0]
            if N_use > 0:
                bo[i, :N_use] = all_o[idx[:N_use]]
                bd[i, :N_use] = all_d[idx[:N_use]]
                bt[i, :N_use] = all_t[idx[:N_use]]
                bw[i, :N_use] = all_w[idx[:N_use]]
        return bo, bd, bt, bw, Np, te, tp, td

    pre_out, t_pre = bench(step_preprocess, warmup=WARMUP, runs=RUNS)
    bo, bd, bt, bw, Np, te, tp, td = pre_out
    print(f"preprocess    : {fmt(t_pre)}  (per-particle photons: {Np.tolist()})")

    # === Step 3 — host->device ===
    def step_device_put():
        return (
            jax.device_put(bo), jax.device_put(bd), jax.device_put(bt), jax.device_put(bw),
            jax.device_put(Np), jax.device_put(te), jax.device_put(tp), jax.device_put(td),
        )
    arrays_d, t_dput = bench(step_device_put, warmup=WARMUP, runs=RUNS, block=True)
    bo_d, bd_d, bt_d, bw_d, Np_d, te_d, tp_d, td_d = arrays_d
    bytes_pushed = bo.nbytes + bd.nbytes + bt.nbytes + bw.nbytes + Np.nbytes + te.nbytes + tp.nbytes + td.nbytes
    print(f"device_put    : {fmt(t_dput)}  ({bytes_pushed:,} bytes / call)")

    # === Step 4 — kernel-only (vmap + block) ===
    def simulate_single_particle(track_E, track_pos, track_dir,
                                 ph_o, ph_d, ph_t, ph_w, N, key):
        track_params = ParticleParams.from_cartesian(
            energy=track_E, position=track_pos, direction=track_dir, t0=0.0)
        photonsim_data = {
            "photon_origins": ph_o, "photon_directions": ph_d,
            "photon_times": ph_t, "wavelengths": ph_w, "N": N,
            "apply_rotation": False,
            "rotation_axis": jnp.array([1.0, 0.0, 0.0]),
            "rotation_angle": 0.0,
            "apply_translation": False,
            "translation_vector": jnp.zeros(3),
        }
        return event_simulator(track_params, key, photonsim_data)

    simulate_all_particles = jax.vmap(
        simulate_single_particle,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    master_key = jax.random.PRNGKey(42)
    particle_keys = jax.random.split(master_key, n_particles)

    def step_kernel():
        return simulate_all_particles(
            te_d, tp_d, td_d, bo_d, bd_d, bt_d, bw_d, Np_d, particle_keys)

    print(f"kernel (n_particles={n_particles}, PAD_SIZE={PAD_SIZE:,}, K={args.K}):")
    (PE_pp, T_pp), t_kernel = bench(step_kernel, warmup=WARMUP, runs=RUNS, block=True)
    print(f"              : {fmt(t_kernel)}")

    # === Step 5 — to_host (np.asarray on warm buffers) ===
    PE_pp_warm, T_pp_warm = step_kernel()
    _block((PE_pp_warm, T_pp_warm))

    def step_to_host():
        return np.asarray(PE_pp_warm, dtype=np.float32), np.asarray(T_pp_warm, dtype=np.float32)
    _, t_host = bench(step_to_host, warmup=WARMUP, runs=RUNS)
    print(f"to_host       : {fmt(t_host)}")

    # === Step 6 — aggregate (jnp.sum/jnp.min) + smearing ===
    def step_agg_smear():
        PE_true = jnp.sum(PE_pp_warm, axis=0)
        T_true  = jnp.min(jnp.where(T_pp_warm > 0, T_pp_warm, jnp.inf), axis=0)
        T_true  = jnp.where(jnp.isfinite(T_true), T_true, 0.0)
        sk1, sk2 = jax.random.split(jax.random.PRNGKey(0))
        PE_reco = smear_charges_SK_like(PE_true, key=sk1)
        T_reco  = smear_times(T_true, key=sk2)
        return PE_reco, T_reco
    _, t_smear = bench(step_agg_smear, warmup=WARMUP, runs=RUNS, block=True)
    print(f"agg+smear     : {fmt(t_smear)}")

    # === Step 7 — t0 shift (numpy where) ===
    T_pp_np = np.asarray(T_pp_warm)
    def step_t0():
        t0v = np.float32(1.5)
        return np.where(T_pp_np > 0, T_pp_np + t0v, T_pp_np)
    _, t_t0 = bench(step_t0, warmup=WARMUP, runs=RUNS)
    print(f"t0_shift      : {fmt(t_t0)}")

    # === Step 8 — build_interaction_metadata ===
    def step_meta():
        return build_interaction_metadata(
            particle_data, t0=1.5,
            vertex_xyz=np.zeros(3, dtype=np.float32),
            source_type_code=0)
    interaction_meta, t_meta = bench(step_meta, warmup=WARMUP, runs=RUNS)
    print(f"meta          : {fmt(t_meta)}")

    # === Step 9 — _compute_contained ===
    PE_pp_np = np.asarray(PE_pp_warm)
    extended_info = {
        "n_particles": n_particles,
        "particles": particle_data["particles"],
        "track_info_dict": particle_data["track_info_dict"],
        "primary_to_interaction": {
            tid: 0 for tid in interaction_meta["primary_track_ids"]},
        "interaction_metadata": [interaction_meta],
        "PE_per_particle": PE_pp_np,
        "T_per_particle": T_pp_np,
        "PE_reco": PE_pp_np[0],
        "T_reco":  T_pp_np[0],
        "source": "PhotonSim_Particles_VMAP",
    }
    def step_contain():
        return _compute_contained(extended_info, None)
    _, t_cont = bench(step_contain, warmup=WARMUP, runs=RUNS)
    print(f"contain       : {fmt(t_cont)}")

    # === Summary ===
    rows = [
        ("root_read+cat", t_root),
        ("preprocess", t_pre),
        ("device_put", t_dput),
        ("kernel",     t_kernel),
        ("to_host",    t_host),
        ("agg+smear",  t_smear),
        ("t0_shift",   t_t0),
        ("meta",       t_meta),
        ("contain",    t_cont),
    ]
    total_mean_s = sum(t.mean() for _, t in rows)
    print()
    print(f'{"stage":<12}  {"mean ms":>10}  {"std ms":>9}  {"min ms":>9}  {"max ms":>9}  {"frac":>6}')
    print("-" * 64)
    for name, t in rows:
        m = 1000 * t.mean()
        print(f'{name:<12}  {m:>10.2f}  {1000*t.std():>9.2f}  '
              f'{1000*t.min():>9.2f}  {1000*t.max():>9.2f}  '
              f'{m/(1000*total_mean_s):>6.1%}')
    print("-" * 64)
    print(f'{"SUM":<12}  {1000*total_mean_s:>10.2f} ms  ~ {total_mean_s:.2f} s')
    print()
    print(f"PAD_SIZE: {PAD_SIZE:,}    n_particles: {n_particles}    "
          f"actual photons: {total_photons:,}")
    print(f"kernel per-event mean: {1000*t_kernel.mean():.2f} ms")
    print(f"kernel per-photon equivalent (over PAD_SIZE x n_particles): "
          f"{t_kernel.mean() / (PAD_SIZE * n_particles) * 1e9:.2f} ns/photon")


if __name__ == "__main__":
    main()
