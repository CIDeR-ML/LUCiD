"""
Check if Cherenkov ring is visible in time residuals after subtracting
the isotropic component (distance/c from vertex).

Also check: is the timing pattern EXACTLY distance/c, or is it offset?
If there's a constant offset, maybe the t0 prediction is dominating.
"""

import sys
sys.path.insert(0, '..')

import jax
import jax.numpy as jnp
import numpy as np
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.detector_params import ParticleParams

JSON = '../config/SK_geom_config.json'
PHYSICS = '../config/SK_physics_config.json'
Nphot = 1_000_000
key = jax.random.PRNGKey(719007)
vertex = np.array([-10., 0., 0.])

detector = generate_detector(JSON)
pts = np.array(detector.all_points)
NUM = len(pts)

sim = setup_event_simulator(
    JSON, Nphot, temperature=0.1, K=7, is_data=False,
    physics_config=PHYSICS, default_detector_params=True,
)

track = ParticleParams.from_cartesian(
    energy=jnp.array(1050.0),
    position=jnp.array(vertex, dtype=jnp.float32),
    direction=jnp.array([1., 0., 0.], dtype=jnp.float32),
    t0=0.0,
)

charges, times = jax.lax.stop_gradient(sim(track, key))
charges_np = np.array(charges)
times_np = np.array(times)

c_med = 0.299792 / 1.33  # m/ns
distances = np.linalg.norm(pts - vertex, axis=1)
expected_isotropic = distances / c_med

# Residuals: positive means later than isotropic, negative means earlier
residuals = times_np - expected_isotropic

print("=== TIME ANALYSIS ===")
print(f"Sensors hit: {np.sum(charges_np > 0)} / {NUM}")
print()

# Linear fit: time = a * distance + b
mask = charges_np > 0
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(distances[mask], times_np[mask], 1)
print(f"Linear fit: time = {coeffs[0]:.4f} * distance + {coeffs[1]:.4f}")
print(f"Expected slope (1/c_medium): {1/c_med:.4f}")
print(f"Slope ratio: {coeffs[0] * c_med:.4f} (should be ~1.0 if isotropic)")
print(f"Intercept (offset): {coeffs[1]:.4f} ns")
print()

# Check if residuals show Cherenkov ring structure
# Ring sensors should have the highest charge AND the most negative residuals
# Sort by charge (descending) and check their residuals
charge_order = np.argsort(charges_np)[::-1]
top_100 = charge_order[:100]  # top 100 by charge (ring sensors)
bottom_100 = charge_order[-100:]  # bottom 100 (far from ring)

print("=== RESIDUALS (time - distance/c) ===")
print(f"All sensors:    mean={residuals[mask].mean():.4f}, std={residuals[mask].std():.4f}")
print(f"Top 100 charge: mean={residuals[top_100].mean():.4f}, std={residuals[top_100].std():.4f}")
print(f"Bot 100 charge: mean={residuals[bottom_100].mean():.4f}, std={residuals[bottom_100].std():.4f}")
print()

# Check top-10 earliest times vs top-10 charge
top10_charge = charge_order[:10]
top10_time = np.argsort(times_np)[:10]
print(f"Top 10 by charge: indices = {top10_charge}")
print(f"Top 10 by time:   indices = {top10_time}")
print(f"Overlap: {len(set(top10_charge) & set(top10_time))}/10")
print()

# What are the times at the ring sensors?
print("=== TOP 10 CHARGE SENSORS ===")
for i in top10_charge:
    print(f"  Sensor {i:5d}: charge={charges_np[i]:.3f}, time={times_np[i]:.2f}, "
          f"dist={distances[i]:.2f}m, expected_t={distances[i]/c_med:.2f}, "
          f"residual={residuals[i]:.2f}")

print()
print("=== TOP 10 EARLIEST TIME SENSORS ===")
for i in top10_time:
    print(f"  Sensor {i:5d}: charge={charges_np[i]:.3f}, time={times_np[i]:.2f}, "
          f"dist={distances[i]:.2f}m, expected_t={distances[i]/c_med:.2f}, "
          f"residual={residuals[i]:.2f}")

# Final check: correlation of residuals with charge
corr_res_charge = np.corrcoef(residuals[mask], charges_np[mask])[0, 1]
print(f"\nCorr(residual, charge): {corr_res_charge:.4f}")
print("  (negative = high charge sensors have earlier-than-isotropic timing = Cherenkov signal)")
