"""Level 1 Baseline Capture — Pure function-level values.

Captures reference values from the current code for comparison.
Works with both old (tools.*) and new (lucid.*) imports via
LUCID_BACKEND environment variable.

No SIREN model or ROOT files needed — uses synthetic inputs only.

Usage:
  # Capture from refactored code
  LUCID_BACKEND=lucid JAX_PLATFORM_NAME=cpu python baseline_scripts/L1_capture.py

  # Capture from fixed baseline
  cd /tmp/lucid-baseline && LUCID_BACKEND=tools JAX_PLATFORM_NAME=cpu python /path/to/L1_capture.py
"""
import os
import sys
import json
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

BACKEND = os.environ.get("LUCID_BACKEND", "lucid")

if BACKEND == "tools":
    sys.path.insert(0, os.getcwd())
    from tools.simulation import (
        normalize, compute_reflection_direction, sample_cosine_hemisphere,
        create_local_frame, sample_scatter_distance, solve_rayleigh_inverse_cdf,
        compute_scatter_direction, jax_normalize, jax_rotate_vector,
        photon_iteration_sample, photon_iteration_update_factors,
        photon_iteration_update_factors_safe,
        make_hits_simulation, make_hits_data, make_hits_likelihood,
    )
    from tools.losses import WC_loss, WC_smooth_loss, poisson_nll
    from tools.optimization.losses import (
        counts_loss, energy_loss, origin_time_loss, cone_time_loss,
        segment_logsumexp, first_arrival_nll,
    )
    from tools.detector_params import DetectorParams, ParticleParams
    from tools.geometry import generate_detector
    from tools.propagation.cylinder import create_photon_propagator
    GEOM_CONFIG = "config/WCTE_geom_config.json"
else:
    sys.path.insert(0, os.getcwd())
    from lucid.simulation.optics import (
        normalize, compute_reflection_direction, sample_cosine_hemisphere,
        create_local_frame, sample_scatter_distance, solve_rayleigh_inverse_cdf,
        compute_scatter_direction, jax_normalize,
    )
    from lucid.utils import jax_rotate_vector
    from lucid.simulation.photon_step import (
        photon_iteration_sample, photon_iteration_update_factors,
        photon_iteration_update_factors_safe,
    )
    from lucid.simulation.sensor_response import (
        make_hits_simulation, make_hits_data, make_hits_likelihood,
    )
    from lucid.losses import (
        WC_loss, WC_smooth_loss, poisson_nll, counts_loss, energy_loss,
        origin_time_loss, cone_time_loss, segment_logsumexp, first_arrival_nll,
    )
    from lucid.detector_params import DetectorParams, ParticleParams
    from lucid.geometry import generate_detector
    from lucid.propagation.cylinder import create_photon_propagator
    GEOM_CONFIG = "config/WCTE_geom_config.json"

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit

results = {}


def to_list(x):
    """Convert JAX/numpy array to JSON-serializable list."""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return float(x)


# ── L1-1: Optics functions ──────────────────────────────────────────

key = jax.random.PRNGKey(42)

results["normalize_3_4_0"] = to_list(normalize(jnp.array([3., 4., 0.])))
results["normalize_zero"] = to_list(normalize(jnp.array([0., 0., 0.])))
results["reflection"] = to_list(compute_reflection_direction(
    jnp.array([1., -1., 0.]) / jnp.sqrt(2.), jnp.array([0., 1., 0.])))
results["local_frame_z"] = to_list(create_local_frame(jnp.array([0., 0., 1.])))
results["rayleigh_0.5"] = to_list(solve_rayleigh_inverse_cdf(0.5))
results["rotate_90_z"] = to_list(jax_rotate_vector(
    jnp.array([1., 0., 0.]), jnp.array([0., 0., 1.]), jnp.pi / 2))
results["scatter_dist_D5_S2"] = to_list(sample_scatter_distance(5.0, 2.0, key))
results["scatter_dir"] = to_list(compute_scatter_direction(jnp.array([0., 0., 1.]), key))
results["cosine_hemi"] = to_list(sample_cosine_hemisphere(jnp.array([0., 0., 1.]), key))


# ── L1-2: Photon step functions ─────────────────────────────────────

pi_key = jax.random.PRNGKey(42)
pi_args = dict(
    position=jnp.array([0.5, 0.5, 0.5]),
    direction=jnp.array([0.0, 0.0, 1.0]),
    time=0.0,
    surface_distance=1.5,
    normal=jnp.array([0.0, 0.0, 1.0]),  # outward convention
    scatter_length=50.0,
    wall_reflection_rate=0.5,
    sensor_reflection_rate=0.3,
    absorption_length=100.0,
    hit_sensor=False,
    rng_key=pi_key,
    speed_of_light=0.2253,
)

sample_out = photon_iteration_sample(**pi_args)
results["pi_sample"] = [to_list(x) for x in sample_out]

update_out = photon_iteration_update_factors(**pi_args)
results["pi_update"] = [to_list(x) for x in update_out]

# With sensor hit
pi_args_sensor = {**pi_args, 'hit_sensor': True}
update_sensor = photon_iteration_update_factors(**pi_args_sensor)
results["pi_update_sensor"] = [to_list(x) for x in update_sensor]


# ── L1-3: Custom VJP gradient ───────────────────────────────────────

def loss_pi(pos):
    args = {**pi_args, 'position': pos}
    _, _, _, dp, _, _ = photon_iteration_update_factors_safe(**args)
    return dp

grad_pi = jax.grad(loss_pi)(jnp.array([0.5, 0.5, 0.5]))
results["vjp_grad_wrt_position"] = to_list(grad_pi)

def loss_scatter(scatter_length):
    args = {**pi_args, 'scatter_length': scatter_length}
    _, _, _, dp, _, _ = photon_iteration_update_factors_safe(**args)
    return dp

grad_scatter = jax.grad(loss_scatter)(50.0)
results["vjp_grad_wrt_scatter_length"] = to_list(grad_scatter)

def loss_absorption(abs_length):
    args = {**pi_args, 'absorption_length': abs_length}
    _, _, _, _, refl, _ = photon_iteration_update_factors_safe(**args)
    return refl

grad_abs = jax.grad(loss_absorption)(100.0)
results["vjp_grad_wrt_absorption_length"] = to_list(grad_abs)


# ── L1-4: Sensor response ───────────────────────────────────────────

flat_w = jnp.array([0.5, 0.3, 0.8, 0.1, 0.6])
flat_i = jnp.array([0, 2, 5, 5, 10])
flat_t = jnp.array([10.0, 15.0, 12.0, 20.0, 8.0])
n_det = 20
qe_corr = jnp.ones(n_det)

q_sim, t_sim = make_hits_simulation(flat_w, flat_i, flat_t, n_det,
                                     qe=0.2, qe_corrections=qe_corr, temperature=0.01)
results["hits_sim_charge"] = to_list(q_sim)
results["hits_sim_time"] = to_list(t_sim)

log_w, safe_t, fi, total_q = make_hits_likelihood(flat_w, flat_i, flat_t, n_det,
                                                    qe=0.2, qe_corrections=qe_corr)
results["hits_like_log_w"] = to_list(log_w)
results["hits_like_total_q"] = to_list(total_q)

q_data, t_data_true, t_data = make_hits_data(flat_w, flat_i, flat_t, n_det,
                                 qe=0.2, rng_key=key)
results["hits_data_charge"] = to_list(q_data)
results["hits_data_time"] = to_list(t_data)


# ── L1-5: Loss functions ────────────────────────────────────────────

true_q = jnp.array([10.0, 5.0, 0.0, 3.0, 8.0])
pred_q = jnp.array([9.0, 6.0, 0.1, 2.5, 7.5])

results["poisson_nll"] = to_list(poisson_nll(true_q, pred_q))
results["counts_loss"] = to_list(counts_loss(true_q, pred_q))
results["energy_loss"] = to_list(energy_loss(pred_q, true_q))

seg_data = jnp.array([1.0, 2.0, 3.0, 0.5, 1.5])
seg_idx = jnp.array([0, 0, 1, 1, 2])
results["segment_logsumexp"] = to_list(segment_logsumexp(seg_data, seg_idx, 3))


# ── L1-6: Loss gradients ────────────────────────────────────────────

def loss_counts(pred):
    return counts_loss(true_q, pred)

grad_counts = jax.grad(loss_counts)(pred_q)
results["grad_counts_loss"] = to_list(grad_counts)

def loss_sim_charge(weights):
    qc = jnp.ones(10)
    q, t = make_hits_simulation(weights, jnp.array([0, 0, 3]), jnp.array([10., 12., 8.]),
                                 10, qe=0.2, qe_corrections=qc)
    return jnp.sum(q)

grad_sim = jax.grad(loss_sim_charge)(jnp.array([0.8, 0.6, 0.9]))
results["grad_make_hits_wrt_weights"] = to_list(grad_sim)


# ── L1-7: Propagator output ─────────────────────────────────────────

if os.path.exists(GEOM_CONFIG):
    det = generate_detector(GEOM_CONFIG)
    sp = jnp.array(det.all_points)
    prop = create_photon_propagator(sp, det.S_radius, r=det.r, h=det.H,
                                    temperature=0.2, max_candidates_per_ray=4)
    origins = jnp.zeros((3, 3))
    dirs = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    prop_result = prop(origins, dirs)

    results["prop_positions"] = to_list(prop_result['positions'])
    results["prop_normals"] = to_list(prop_result['normals'])
    results["prop_weights_shape"] = list(prop_result['sensor_weights'].shape)
    results["prop_weights_sum"] = to_list(jnp.sum(prop_result['sensor_weights']))

    # Propagator gradient
    def loss_prop(origin):
        r = prop(origin[None, :], jnp.array([[1., 0., 0.]]))
        return jnp.sum(r['sensor_weights'])

    grad_prop = jax.grad(loss_prop)(jnp.array([0., 0., 0.]))
    results["prop_grad_weights_wrt_origin"] = to_list(grad_prop)
else:
    print(f"WARNING: {GEOM_CONFIG} not found, skipping propagator tests")


# ── Save results ─────────────────────────────────────────────────────

output_file = f"baseline_scripts/L1_baseline_{BACKEND}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Captured {len(results)} values → {output_file}")
