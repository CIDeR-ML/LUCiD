"""
Compare OLD vs NEW simulator outputs directly.
Runs the old simulation code from clean_run and the new code side by side.
"""

import sys
import os
import numpy as np

# ── Helper ─────────────────────────────────────────────────────────────
def measure_isotropic(charges_np, times_np, detector_points, vertex, label):
    nonzero = charges_np > 0
    n_hit = np.sum(nonzero)
    if n_hit < 10:
        print(f"  [{label}] Only {n_hit} sensors hit — skipping")
        return

    hit_times = times_np[nonzero]
    hit_positions = detector_points[nonzero]
    distances = np.linalg.norm(hit_positions - vertex, axis=1)
    c_medium = 0.299792 / 1.33

    corr = np.corrcoef(hit_times, distances)[0, 1]
    print(f"  [{label}]")
    print(f"    Sensors hit: {n_hit} / {len(detector_points)}")
    print(f"    Charge range: {charges_np[nonzero].min():.4f} – {charges_np[nonzero].max():.4f}")
    print(f"    Time range:   {hit_times.min():.2f} – {hit_times.max():.2f} ns")
    print(f"    Corr(time, distance): {corr:.4f}  {'<-- ISOTROPIC' if corr > 0.9 else ''}")
    print()


# ── Run NEW simulator ─────────────────────────────────────────────────
print("=" * 70)
print("Running NEW simulator...")
print("=" * 70)

sys.path.insert(0, '..')
import jax
import jax.numpy as jnp
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.detector_params import ParticleParams, load_detector_params

JSON = '../config/SK_geom_config.json'
PHYSICS_CONFIG = '../config/SK_physics_config.json'
Nphot = 1_000_000
key = jax.random.PRNGKey(719007)
vertex = np.array([-10., 0., 0.])

detector = generate_detector(JSON)
detector_points = np.array(detector.all_points)

true_track = ParticleParams.from_cartesian(
    energy=jnp.array(1050.0),
    position=jnp.array(vertex),
    direction=jnp.array([1., 0., 0.]),
    t0=0.0,
)

sim_new = setup_event_simulator(
    JSON, Nphot, temperature=0.1, K=7, is_data=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True,
)
charges_new, times_new = jax.lax.stop_gradient(sim_new(true_track, key))
measure_isotropic(np.array(charges_new), np.array(times_new), detector_points, vertex, "NEW")

# Clear everything related to the new module
del sim_new
# Remove new tools from sys.modules so old import works cleanly
mods_to_remove = [k for k in sys.modules if k.startswith('tools')]
for m in mods_to_remove:
    del sys.modules[m]
sys.path.remove('..')

# ── Run OLD simulator ─────────────────────────────────────────────────
print("=" * 70)
print("Running OLD simulator...")
print("=" * 70)

sys.path.insert(0, '../../clean_run/LUCiD')

# Re-import with old code
from tools.simulation import setup_event_simulator as setup_old
from tools.geometry import generate_detector as gen_det_old

detector_old = gen_det_old(os.path.join('../../clean_run/LUCiD', 'config/SK_geom_config.json'))

# Old simulator: no physics_config, no default_detector_params
sim_old = setup_old(
    os.path.join('../../clean_run/LUCiD', 'config/SK_geom_config.json'),
    Nphot, temperature=0.1, K=7, is_data=False,
)

# Old API: (energy, position, [theta, phi]) tuple + detector_params tuple
true_direction = jnp.array([1., 0., 0.])
theta = jnp.arccos(jnp.clip(true_direction[2], -1.0, 1.0))
phi = jnp.arctan2(true_direction[1], true_direction[0])
old_track = (jnp.array(1050.0), jnp.array(vertex, dtype=jnp.float32), jnp.array([theta, phi]))

old_detector_params = (
    jnp.array(50.0),   # scatter_length
    jnp.array(0.2),    # reflection_rate
    jnp.array(50.0),   # absorption_length
    jnp.array(0.001),  # tau_gs
)

charges_old, times_old = jax.lax.stop_gradient(sim_old(old_track, old_detector_params, key))
measure_isotropic(np.array(charges_old), np.array(times_old), detector_points, vertex, "OLD")
