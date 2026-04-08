#!/usr/bin/env python3
"""
TEST 9b: Verify that monkey-patching actually changes the function being called.
Add a side-effect print to confirm.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import tools.simulation as sim_module

_orig = sim_module.photon_iteration_update_factors

# Simple test: does patching the function change what _safe calls?
print("=== Test: does patching work? ===")

# Check: calling _safe should call _orig
print(f"Original function id: {id(_orig)}")
print(f"Module function id:   {id(sim_module.photon_iteration_update_factors)}")

# Create a wrapper that marks calls
call_log = []

def logging_wrapper(*args, **kwargs):
    call_log.append('PATCHED')
    return _orig(*args, **kwargs)

# Patch it
sim_module.photon_iteration_update_factors = logging_wrapper
print(f"After patch, module function id: {id(sim_module.photon_iteration_update_factors)}")

# Now call _safe with dummy data
pos = jnp.array([0., 0., 0.])
dir_ = jnp.array([0., 0., 1.])
time = jnp.array(0.)
surface_dist = jnp.array(10.)
normal = jnp.array([0., 0., -1.])
scatter_len = jnp.array(50.)
wall_refl = jnp.array(0.2)
sensor_refl = jnp.array(0.1)
abs_len = jnp.array(50.)
hit_sensor = jnp.array(0.)
key = jax.random.PRNGKey(0)
sol = jnp.array(0.2255)

print("\nCalling photon_iteration_update_factors_safe (should use patched fn)...")
try:
    result = sim_module.photon_iteration_update_factors_safe(
        pos, dir_, time, surface_dist, normal,
        scatter_len, wall_refl, sensor_refl, abs_len,
        hit_sensor, key, sol)
    print(f"  Result returned OK, new_time={float(result[2]):.4f}")
except Exception as e:
    print(f"  Error: {e}")

print(f"  Call log: {call_log}")
if call_log:
    print("  >>> PATCH IS WORKING - _safe calls the patched function")
else:
    print("  >>> PATCH IS NOT WORKING - _safe bypasses the patch")

# Restore
sim_module.photon_iteration_update_factors = _orig
