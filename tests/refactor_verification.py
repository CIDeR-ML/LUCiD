#!/usr/bin/env python3
"""
Refactor verification tests for LUCiD.

Captures reference outputs from the current code and verifies them after
structural changes. Run BEFORE any refactoring to establish baselines,
then AFTER each phase to confirm nothing broke.

Usage:
    python tests/refactor_verification.py [--capture]

    --capture: Record reference values to tests/baselines.npz
    (default): Compare current outputs against saved baselines
"""

import sys
import os
import argparse
import numpy as np

# Add project root to path (will be removed after pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

# Force CPU for deterministic results
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')


BASELINE_PATH = os.path.join(os.path.dirname(__file__), 'baselines.npz')

# ============================================================================
# Test definitions
# ============================================================================

def run_all_tests():
    """Run all tests, return dict of {test_name: output_array}."""
    results = {}

    # ------------------------------------------------------------------
    # DETERMINISTIC TESTS (pure functions, no RNG)
    # ------------------------------------------------------------------

    from tools.simulation import (
        normalize, compute_reflection_direction, solve_rayleigh_inverse_cdf,
        create_local_frame, make_hits_simulation, make_hits_data,
        make_hits_likelihood,
    )

    # 1. normalize
    v = jnp.array([3.0, 4.0, 0.0])
    results['normalize'] = np.asarray(normalize(v))

    # 2. compute_reflection_direction
    incident = jnp.array([1.0, -1.0, 0.0])
    normal = jnp.array([0.0, 1.0, 0.0])
    results['reflect'] = np.asarray(compute_reflection_direction(incident, normal))

    # 3. solve_rayleigh_inverse_cdf
    results['rayleigh_cdf'] = np.asarray(jnp.array([
        solve_rayleigh_inverse_cdf(jnp.array(0.0)),
        solve_rayleigh_inverse_cdf(jnp.array(0.25)),
        solve_rayleigh_inverse_cdf(jnp.array(0.5)),
        solve_rayleigh_inverse_cdf(jnp.array(0.75)),
        solve_rayleigh_inverse_cdf(jnp.array(1.0)),
    ]))

    # 4. create_local_frame
    z = jnp.array([0.0, 0.0, 1.0])
    results['local_frame_z'] = np.asarray(create_local_frame(z))
    z2 = jnp.array([1.0, 0.0, 0.0])
    results['local_frame_x'] = np.asarray(create_local_frame(z2))

    # 5. spherical_to_cartesian (all 3 implementations)
    from tools.utils import spherical_to_cartesian as stc_utils
    from tools.optimization.utils.functions import spherical_to_cartesian as stc_opt

    theta, phi = jnp.array(0.5), jnp.array(1.2)
    results['stc_utils'] = np.asarray(stc_utils(theta, phi))
    results['stc_opt'] = np.asarray(stc_opt(theta, phi))

    # Also test the one on detector_params
    from tools.detector_params import ParticleParams
    pp = ParticleParams(
        energy=jnp.array(500.0),
        position=jnp.array([1.0, 2.0, 3.0]),
        theta=theta, phi=phi,
        t0=jnp.array(0.0)
    )
    results['stc_particle_direction'] = np.asarray(pp.direction)

    # 6-8. bounds_check functions
    from tools.propagation.cylinder import cylinder_bounds_check
    from tools.propagation.sphere import sphere_bounds_check
    from tools.propagation.box import box_bounds_check

    test_positions = jnp.array([
        [0.0, 0.0, 0.0],      # inside all
        [100.0, 0.0, 0.0],    # outside all
        [4.9, 0.0, 0.0],      # near boundary
        [0.0, 0.0, 4.9],      # near top
        [3.0, 3.0, 3.0],      # corner region
    ])
    results['bounds_cylinder'] = np.asarray(cylinder_bounds_check(test_positions, 5.0, 10.0))
    results['bounds_sphere'] = np.asarray(sphere_bounds_check(test_positions, 5.0))
    results['bounds_box'] = np.asarray(box_bounds_check(test_positions, 10.0, 10.0, 10.0))

    # 9-10. Loss functions
    from tools.losses import poisson_nll as poisson_nll_tools
    from tools.optimization.losses import (
        counts_loss, energy_loss, origin_time_loss, first_arrival_nll,
        segment_logsumexp,
    )

    true_q = jnp.array([0.0, 5.0, 10.0, 0.5, 3.0])
    pred_q = jnp.array([0.1, 4.5, 11.0, 0.3, 3.5])
    results['poisson_nll_tools'] = np.asarray(poisson_nll_tools(true_q, pred_q))
    results['counts_loss'] = np.asarray(counts_loss(true_q, pred_q))
    results['energy_loss'] = np.asarray(energy_loss(pred_q, true_q))

    # 11. origin_time_loss
    origins = jnp.array([0.0, 0.0, 0.0])
    det_pos = jnp.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-5.0, 0.0, 0.0]])
    true_times = jnp.array([25.0, 25.0, 25.0])
    true_charges = jnp.array([1.0, 1.0, 1.0])
    t0 = jnp.array(0.0)
    results['origin_time_loss'] = np.asarray(origin_time_loss(
        origins, det_pos, true_times, true_charges, t0))

    # 12. segment_logsumexp
    data = jnp.array([1.0, 2.0, 3.0, 0.5, 1.5])
    indices = jnp.array([0, 0, 1, 1, 1])
    results['segment_logsumexp'] = np.asarray(segment_logsumexp(data, indices, 2))

    # 13. first_arrival_nll
    log_w = jnp.array([-1.0, -2.0, -1.5, -3.0, -0.5])
    flat_times = jnp.array([10.0, 12.0, 15.0, 11.0, 14.0])
    flat_indices = jnp.array([0, 0, 1, 1, 1])
    t_obs = jnp.array([11.0, 13.0])
    results['first_arrival_nll'] = np.asarray(first_arrival_nll(
        log_w, flat_times, flat_indices, t_obs, 0.15, 2))

    # 14-15. DetectorParams, ParticleParams
    from tools.detector_params import DetectorParams
    dp = DetectorParams(
        scatter_length=jnp.array(50.0),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        absorption_length=jnp.array(50.0),
        qe=jnp.array(0.065),
        qe_corrections=jnp.ones(10),
    )
    results['dp_scatter'] = np.asarray(dp.scatter_length)
    results['dp_qe_shape'] = np.array(dp.qe_corrections.shape)
    results['pp_energy'] = np.asarray(pp.energy)
    results['pp_position'] = np.asarray(pp.position)

    # 16. Detector construction
    from tools.geometry import generate_detector

    # Use SK_like (algorithmic cylinder) since SK_geom requires ConnectionTable ROOT file
    det_cyl = generate_detector('config/SK_like_geom_config.json')
    results['det_cyl_n_sensors'] = np.array(det_cyl.n_sensors)
    results['det_cyl_radius'] = np.array(det_cyl.r)
    results['det_cyl_first5'] = np.asarray(det_cyl.all_points[:5])
    results['det_cyl_last5'] = np.asarray(det_cyl.all_points[-5:])

    # 17-19. make_hits functions
    n_photons = 20
    n_sensors = 5
    flat_weights = jnp.array([0.1, 0.0, 0.3, 0.0, 0.2,
                               0.0, 0.5, 0.0, 0.1, 0.0,
                               0.4, 0.0, 0.0, 0.3, 0.0,
                               0.0, 0.0, 0.2, 0.0, 0.1])
    flat_indices = jnp.array([0, 1, 2, 3, 4,
                               0, 1, 2, 3, 4,
                               0, 1, 2, 3, 4,
                               0, 1, 2, 3, 4])
    flat_times = jnp.array([10.0, 0.0, 12.0, 0.0, 11.0,
                             0.0, 13.0, 0.0, 14.0, 0.0,
                             15.0, 0.0, 0.0, 16.0, 0.0,
                             0.0, 0.0, 17.0, 0.0, 18.0])
    qe_corr = jnp.ones(n_sensors)

    mh_charge, mh_time = make_hits_simulation(
        flat_weights, flat_indices, flat_times, n_sensors,
        qe=0.5, qe_corrections=qe_corr)
    results['mh_sim_charge'] = np.asarray(mh_charge)
    results['mh_sim_time'] = np.asarray(mh_time)

    log_w_mh, safe_times_mh, idx_mh, total_charge_mh = make_hits_likelihood(
        flat_weights, flat_indices, flat_times, n_sensors,
        qe=0.5, qe_corrections=qe_corr)
    results['mh_like_logw'] = np.asarray(log_w_mh)
    results['mh_like_times'] = np.asarray(safe_times_mh)
    results['mh_like_charge'] = np.asarray(total_charge_mh)

    # ------------------------------------------------------------------
    # STOCHASTIC TESTS (fixed seed → deterministic output)
    # ------------------------------------------------------------------

    from tools.simulation import (
        sample_scatter_distance, compute_scatter_direction,
        sample_cosine_hemisphere, photon_iteration_sample,
        photon_iteration_update_factors,
    )

    # 20. sample_scatter_distance
    key = jax.random.PRNGKey(42)
    results['scatter_dist'] = np.asarray(sample_scatter_distance(
        jnp.array(5.0), jnp.array(50.0), key))

    # 21. compute_scatter_direction
    key = jax.random.PRNGKey(42)
    incident_dir = jnp.array([1.0, 0.0, 0.0])
    results['scatter_dir'] = np.asarray(compute_scatter_direction(incident_dir, key))

    # 22. sample_cosine_hemisphere
    key = jax.random.PRNGKey(42)
    normal_vec = jnp.array([0.0, 0.0, 1.0])
    results['cosine_hemi'] = np.asarray(sample_cosine_hemisphere(normal_vec, key))

    # 23. photon_iteration_sample
    key = jax.random.PRNGKey(42)
    pi_result = photon_iteration_sample(
        jnp.array([0.0, 0.0, 0.0]),     # position
        jnp.array([1.0, 0.0, 0.0]),     # direction
        jnp.array(0.0),                  # time
        jnp.array(5.0),                  # surface_distance
        jnp.array([-1.0, 0.0, 0.0]),    # normal
        jnp.array(50.0),                 # scatter_length
        jnp.array(0.2),                  # wall_reflection_rate
        jnp.array(0.2),                  # sensor_reflection_rate
        jnp.array(50.0),                 # absorption_length
        jnp.array(False),                # hit_sensor
        key,                              # rng_key
        jnp.array(0.2253),              # speed_of_light
    )
    for i, name in enumerate(['pos', 'dir', 'time', 'detect', 'refl', 'cont']):
        results[f'pi_sample_{name}'] = np.asarray(pi_result[i])

    # 24. photon_iteration_update_factors
    key = jax.random.PRNGKey(42)
    uf_result = photon_iteration_update_factors(
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array(0.0),
        jnp.array(5.0),
        jnp.array([-1.0, 0.0, 0.0]),
        jnp.array(50.0),
        jnp.array(0.2),
        jnp.array(0.2),
        jnp.array(50.0),
        jnp.array(False),
        key,
        jnp.array(0.2253),
    )
    for i, name in enumerate(['pos', 'dir', 'time', 'detect', 'refl', 'cont']):
        results[f'uf_{name}'] = np.asarray(uf_result[i])

    # 25. get_isotropic_rays
    from tools.generate import get_isotropic_rays
    key = jax.random.PRNGKey(42)
    rays_dir, rays_orig, rays_w = get_isotropic_rays(
        jnp.array([0.0, 0.0, 0.0]), jnp.array(1e6), 100, key)
    results['iso_rays_dir'] = np.asarray(rays_dir)
    results['iso_rays_orig'] = np.asarray(rays_orig)
    results['iso_rays_w'] = np.asarray(rays_w)

    # 26. Small propagator test
    from tools.propagation.cylinder import create_photon_propagator

    # Small detector for fast test
    small_det = generate_detector('config/WCTE_geom_config.json')
    small_sensors = jnp.array(small_det.all_points)
    propagator = create_photon_propagator(
        small_sensors, small_det.S_radius,
        r=small_det.r, h=small_det.H,
        temperature=0.2, max_sensors_per_cell=4,
    )

    # Fixed rays
    key = jax.random.PRNGKey(42)
    n_test_rays = 50
    ray_origins = jnp.zeros((n_test_rays, 3))
    # Spread rays in different directions
    angles = jnp.linspace(0, 2 * jnp.pi, n_test_rays, endpoint=False)
    ray_directions = jnp.stack([
        jnp.cos(angles), jnp.sin(angles), jnp.zeros(n_test_rays)
    ], axis=1)

    prop_result = propagator(ray_origins, ray_directions)
    results['prop_weights'] = np.asarray(prop_result['sensor_weights'])
    results['prop_indices'] = np.asarray(prop_result['sensor_indices'])
    results['prop_positions'] = np.asarray(prop_result['positions'])
    results['prop_normals'] = np.asarray(prop_result['normals'])
    results['prop_inside'] = np.asarray(prop_result['inside_sensor'])

    return results


def capture_baselines(results):
    """Save reference values to disk."""
    np.savez(BASELINE_PATH, **results)
    print(f"Baselines saved to {BASELINE_PATH}")
    print(f"  {len(results)} tests captured")


def verify_against_baselines(results):
    """Compare current outputs against saved baselines."""
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: No baselines found at {BASELINE_PATH}")
        print("Run with --capture first to establish baselines.")
        return False

    baselines = dict(np.load(BASELINE_PATH, allow_pickle=True))

    all_passed = True
    n_pass = 0
    n_fail = 0
    n_missing = 0

    for name in sorted(set(list(results.keys()) + list(baselines.keys()))):
        if name not in results:
            print(f"  MISSING (not in current): {name}")
            n_missing += 1
            continue
        if name not in baselines:
            print(f"  NEW (not in baseline): {name}")
            n_missing += 1
            continue

        current = results[name]
        baseline = baselines[name]

        try:
            if current.dtype.kind == 'b':
                # Boolean arrays — exact match
                match = np.array_equal(current, baseline)
            elif current.dtype.kind in ('i', 'u'):
                # Integer arrays — exact match
                match = np.array_equal(current, baseline)
            else:
                # Float arrays — very tight tolerance (should be exact for same computation)
                match = np.allclose(current, baseline, rtol=1e-6, atol=1e-7)

            if match:
                n_pass += 1
            else:
                print(f"  FAIL: {name}")
                print(f"    baseline: {baseline.flatten()[:5]}...")
                print(f"    current:  {current.flatten()[:5]}...")
                if current.dtype.kind == 'f':
                    diff = np.abs(current - baseline)
                    print(f"    max_diff: {np.max(diff):.2e}")
                n_fail += 1
                all_passed = False
        except Exception as e:
            print(f"  ERROR comparing {name}: {e}")
            n_fail += 1
            all_passed = False

    print(f"\nResults: {n_pass} passed, {n_fail} failed, {n_missing} missing")

    if all_passed and n_missing == 0:
        print("ALL TESTS PASSED")
    elif all_passed:
        print("ALL MATCHED TESTS PASSED (some missing)")
    else:
        print("SOME TESTS FAILED")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='LUCiD refactor verification')
    parser.add_argument('--capture', action='store_true',
                        help='Capture baseline values (run before refactoring)')
    args = parser.parse_args()

    print("Running verification tests...")
    print(f"JAX devices: {jax.devices()}")
    print()

    results = run_all_tests()
    print(f"\n{len(results)} tests completed.\n")

    if args.capture:
        capture_baselines(results)
    else:
        verify_against_baselines(results)


if __name__ == '__main__':
    main()
