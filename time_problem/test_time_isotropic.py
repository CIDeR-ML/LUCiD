"""
Minimal reproduction of isotropic timing bug in prediction simulator.

Tests each forward-pass physics change in isolation to find the cause.
"""

import sys
sys.path.insert(0, '..')

import jax
import jax.numpy as jnp
import numpy as np
from tools.geometry import generate_detector
from tools.simulation import (
    setup_event_simulator,
    make_hits_simulation,
    photon_iteration_update_factors,
    normalize,
    compute_reflection_direction,
    compute_scatter_direction,
    sample_scatter_distance,
    sample_cosine_hemisphere,
)
from tools.detector_params import ParticleParams, load_detector_params

# ── Setup ──────────────────────────────────────────────────────────────
JSON = '../config/SK_geom_config.json'
PHYSICS_CONFIG = '../config/SK_physics_config.json'
Nphot = 1_000_000

detector = generate_detector(JSON)
detector_points = np.array(detector.all_points)
NUM_SENSORS = len(detector_points)
vertex = np.array([-10., 0., 0.])

true_track = ParticleParams.from_cartesian(
    energy=jnp.array(1050.0),
    position=jnp.array(vertex),
    direction=jnp.array([1., 0., 0.]),
    t0=0.0,
)

key = jax.random.PRNGKey(719007)


def measure_isotropic(charges, times, label):
    """Measure how isotropic the timing pattern is."""
    charges_np = np.array(charges)
    times_np = np.array(times)

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
    expected_range = (distances.min() / c_medium, distances.max() / c_medium)

    print(f"  [{label}]")
    print(f"    Sensors hit: {n_hit} / {NUM_SENSORS}")
    print(f"    Charge range: {charges_np[nonzero].min():.4f} – {charges_np[nonzero].max():.4f}")
    print(f"    Time range:   {hit_times.min():.2f} – {hit_times.max():.2f} ns")
    print(f"    Expected isotropic range: {expected_range[0]:.2f} – {expected_range[1]:.2f} ns")
    print(f"    Corr(time, distance): {corr:.4f}  {'<-- ISOTROPIC' if corr > 0.9 else ''}")
    print()


# ── Test 0: Current (new) simulator ───────────────────────────────────
print("=" * 70)
print("TEST 0: Current (new) simulator — baseline")
print("=" * 70)

sim_new = setup_event_simulator(
    JSON, Nphot, temperature=0.1, K=7, is_data=False,
    physics_config=PHYSICS_CONFIG, default_detector_params=True,
)
charges_new, times_new = jax.lax.stop_gradient(sim_new(true_track, key))
measure_isotropic(charges_new, times_new, "NEW simulator")


# ── Test D: make_hits_simulation threshold 1e-5 ──────────────────────
print("=" * 70)
print("TEST D: make_hits_simulation with threshold=1e-5 (old default)")
print("=" * 70)

# We need the raw flat_weights/indices/times from propagation.
# Easiest: just re-threshold the existing output.
# If charge min >> 1e-5, this won't change anything (confirming threshold is not the cause).
charges_d = jnp.where(charges_new > 1e-5, charges_new, 0.0)
times_d = jnp.where(charges_new > 1e-5, times_new, 0.0)
measure_isotropic(charges_d, times_d, "threshold=1e-5")


# ── Tests A, B, C: Need modified photon_update_fn ────────────────────
# We can't easily swap just the photon_update_fn inside the compiled simulator.
# Instead, we'll create modified versions of setup_event_simulator by
# monkey-patching the simulation module.

import tools.simulation as sim_module

# Save the original
_original_update_fn = sim_module.photon_iteration_update_factors


def make_old_style_update_fn(change_ste=False, change_reflection_dir=False, change_single_rate=False):
    """Create a photon_update_fn with selected old behaviors restored."""

    def patched_update(position, direction, time, surface_distance,
                       normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
                       absorption_length, hit_sensor, rng_key, speed_of_light):

        k1, k2, k3 = jax.random.split(rng_key, 3)

        # Change C: single reflection rate
        if change_single_rate:
            reflection_rate = wall_reflection_rate  # use wall rate for everything (old: single 0.2)
        else:
            reflection_rate = jnp.where(hit_sensor, sensor_reflection_rate, wall_reflection_rate)

        scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

        ratio = surface_distance / scatter_length
        reach_surface_prob = jnp.exp(-ratio)
        scatter_prob = -jnp.expm1(-ratio)

        reflect_prob = reach_surface_prob * reflection_rate
        detect_prob = reach_surface_prob * (1 - reflection_rate)

        reflection_attenuation = jnp.exp(-surface_distance / absorption_length)
        scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)

        # Change A: STE probabilities
        if change_ste:
            probs = jnp.array([reflect_prob, scatter_prob])  # OLD
        else:
            probs = jnp.array([reach_surface_prob, scatter_prob])  # NEW

        probs_normalized = probs / (jnp.sum(probs) + 1e-10)
        u = jax.random.uniform(k1)
        hard_choice = (u < probs_normalized[0]).astype(jnp.float32)
        hard_weights = jnp.array([hard_choice, 1.0 - hard_choice])
        action_weights = hard_weights - jax.lax.stop_gradient(probs_normalized) + probs_normalized

        surface_weight = action_weights[0]
        scatter_weight = action_weights[1]

        epsilon = 1e-4
        surface_pos = position + surface_distance * normalize(direction) + epsilon * normalize(normal)
        scatter_pos = position + scatter_distance * normalize(direction)

        # Change B: reflection direction
        specular_dir = compute_reflection_direction(direction, normal)
        if change_reflection_dir:
            reflection_dir = specular_dir  # OLD: always specular
        else:
            diffuse_dir = sample_cosine_hemisphere(normal, k3)  # NEW: diffuse for walls
            reflection_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)

        scatter_dir = compute_scatter_direction(direction, k3)

        new_pos = surface_weight * surface_pos + scatter_weight * scatter_pos
        new_dir = normalize(surface_weight * reflection_dir + scatter_weight * scatter_dir)

        continuing_factor = reflect_prob * reflection_attenuation + scatter_prob * scatter_attenuation

        distance_traveled = surface_weight * surface_distance + scatter_weight * scatter_distance
        new_time = time + distance_traveled / speed_of_light

        return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor

    return patched_update


def run_with_patched_update(label, **kwargs):
    """Run simulator with a patched photon_update_fn and measure isotropic correlation."""
    patched_fn = make_old_style_update_fn(**kwargs)

    # Monkey-patch the module so setup_event_simulator picks it up
    sim_module.photon_iteration_update_factors = patched_fn

    # Force fresh compilation by creating a new simulator
    sim_patched = setup_event_simulator(
        JSON, Nphot, temperature=0.1, K=7, is_data=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True,
    )
    charges, times = jax.lax.stop_gradient(sim_patched(true_track, key))
    measure_isotropic(charges, times, label)

    # Restore original
    sim_module.photon_iteration_update_factors = _original_update_fn


# ── Test A: Old STE probs [reflect_prob, scatter_prob] ────────────────
print("=" * 70)
print("TEST A: Old STE probs [reflect_prob, scatter_prob]")
print("=" * 70)
run_with_patched_update("STE=old", change_ste=True)


# ── Test B: Old reflection direction (always specular) ────────────────
print("=" * 70)
print("TEST B: Old reflection direction (always specular, no diffuse)")
print("=" * 70)
run_with_patched_update("ReflDir=specular", change_reflection_dir=True)


# ── Test C: Single reflection rate (old style) ────────────────────────
print("=" * 70)
print("TEST C: Single reflection rate (wall_rate for everything)")
print("=" * 70)
run_with_patched_update("SingleRate", change_single_rate=True)


# ── Test ABC: All old behaviors combined ──────────────────────────────
print("=" * 70)
print("TEST ABC: All old photon_update behaviors combined")
print("=" * 70)
run_with_patched_update("ALL_OLD", change_ste=True, change_reflection_dir=True, change_single_rate=True)
