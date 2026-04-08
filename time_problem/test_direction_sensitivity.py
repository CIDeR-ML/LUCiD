"""
Test whether the predicted timing pattern changes with track direction.

If timing is truly physical (Cherenkov), changing direction should change the pattern.
If timing is always isotropic from vertex regardless of direction, something is wrong.
"""

import sys
import os
import numpy as np

def measure_timing(charges_np, times_np, detector_points, vertex, label):
    nonzero = charges_np > 0
    n_hit = np.sum(nonzero)
    if n_hit < 10:
        print(f"  [{label}] Only {n_hit} sensors hit — skipping")
        return None

    hit_times = times_np[nonzero]
    hit_positions = detector_points[nonzero]
    distances = np.linalg.norm(hit_positions - vertex, axis=1)

    corr = np.corrcoef(hit_times, distances)[0, 1]

    # Also check: do the top-10 earliest-time sensors change with direction?
    all_times_full = np.full(len(detector_points), np.inf)
    all_times_full[nonzero] = hit_times
    earliest_10 = np.argsort(all_times_full)[:10]

    print(f"  [{label}]")
    print(f"    Sensors hit: {n_hit} / {len(detector_points)}")
    print(f"    Time range:   {hit_times.min():.2f} – {hit_times.max():.2f} ns")
    print(f"    Corr(time, dist_from_vertex): {corr:.4f}  {'<-- ISOTROPIC' if corr > 0.9 else ''}")
    print(f"    Earliest 10 sensor indices: {earliest_10}")
    print(f"    Earliest 10 times: {all_times_full[earliest_10]}")
    print()
    return earliest_10


# ── Directions to test ────────────────────────────────────────────────
directions = {
    '+x': [1., 0., 0.],
    '+y': [0., 1., 0.],
    '+z': [0., 0., 1.],
    'diagonal': [1., 1., 1.],
}

vertex = np.array([-10., 0., 0.])

import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(719007)

JSON_NEW = '../config/SK_geom_config.json'
JSON_OLD = '../../clean_run/LUCiD/config/SK_geom_config.json'
PHYSICS_CONFIG = '../config/SK_physics_config.json'
Nphot = 1_000_000

# ── NEW simulator ─────────────────────────────────────────────────────
print("=" * 70)
print("NEW SIMULATOR — varying direction")
print("=" * 70)

sys.path.insert(0, '..')
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.detector_params import ParticleParams, load_detector_params

detector = generate_detector(JSON_NEW)
detector_points = np.array(detector.all_points)

sim_new = setup_event_simulator(
    JSON_NEW, Nphot, temperature=0.1, K=7, is_data=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True,
)

new_earliest = {}
for dname, dvec in directions.items():
    track = ParticleParams.from_cartesian(
        energy=jnp.array(1050.0),
        position=jnp.array(vertex, dtype=jnp.float32),
        direction=jnp.array(dvec, dtype=jnp.float32),
        t0=0.0,
    )
    charges, times = jax.lax.stop_gradient(sim_new(track, key))
    e10 = measure_timing(np.array(charges), np.array(times), detector_points, vertex, f"NEW dir={dname}")
    new_earliest[dname] = e10

# Check if earliest sensors change across directions
print("--- NEW: Do earliest sensors change with direction? ---")
for d1 in directions:
    for d2 in directions:
        if d1 >= d2:
            continue
        overlap = len(set(new_earliest[d1]) & set(new_earliest[d2]))
        print(f"  {d1} vs {d2}: {overlap}/10 overlap in earliest sensors")
print()


# ── OLD simulator ─────────────────────────────────────────────────────
print("=" * 70)
print("OLD SIMULATOR — varying direction")
print("=" * 70)

# Clear new modules
mods_to_remove = [k for k in sys.modules if k.startswith('tools')]
for m in mods_to_remove:
    del sys.modules[m]
sys.path.remove('..')

sys.path.insert(0, '../../clean_run/LUCiD')
from tools.simulation import setup_event_simulator as setup_old
from tools.geometry import generate_detector as gen_det_old

sim_old = setup_old(JSON_OLD, Nphot, temperature=0.1, K=7, is_data=False)

old_detector_params = (
    jnp.array(50.0),
    jnp.array(0.2),
    jnp.array(50.0),
    jnp.array(0.001),
)

old_earliest = {}
for dname, dvec in directions.items():
    dvec_jnp = jnp.array(dvec, dtype=jnp.float32)
    dvec_jnp = dvec_jnp / jnp.linalg.norm(dvec_jnp)
    theta = jnp.arccos(jnp.clip(dvec_jnp[2], -1.0, 1.0))
    phi = jnp.arctan2(dvec_jnp[1], dvec_jnp[0])
    old_track = (jnp.array(1050.0), jnp.array(vertex, dtype=jnp.float32), jnp.array([theta, phi]))

    charges, times = jax.lax.stop_gradient(sim_old(old_track, old_detector_params, key))
    e10 = measure_timing(np.array(charges), np.array(times), detector_points, vertex, f"OLD dir={dname}")
    old_earliest[dname] = e10

print("--- OLD: Do earliest sensors change with direction? ---")
for d1 in directions:
    for d2 in directions:
        if d1 >= d2:
            continue
        overlap = len(set(old_earliest[d1]) & set(old_earliest[d2]))
        print(f"  {d1} vs {d2}: {overlap}/10 overlap in earliest sensors")
