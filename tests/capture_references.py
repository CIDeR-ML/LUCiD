"""
Run this ONCE on the pre-refactor code to capture reference values.
These values become the hardcoded assertions in the pytest suite.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

# ── Optics (simulation.py) ──────────────────────────────────────────
from tools.simulation import (
    normalize, jax_normalize, compute_reflection_direction,
    create_local_frame, sample_scatter_distance, solve_rayleigh_inverse_cdf,
    compute_scatter_direction, sample_cosine_hemisphere, jax_rotate_vector,
    photon_iteration_sample, photon_iteration_update_factors,
    make_hits_simulation, make_hits_data, make_hits_likelihood,
)

refs = {}

# 1. normalize
v = jnp.array([3.0, 4.0, 0.0])
refs["normalize_3_4_0"] = normalize(v).tolist()

v_zero = jnp.array([0.0, 0.0, 0.0])
refs["normalize_zero"] = normalize(v_zero).tolist()

# 2. jax_normalize
refs["jax_normalize_3_4_0"] = jax_normalize(jnp.array([3.0, 4.0, 0.0])).tolist()
refs["jax_normalize_zero"] = jax_normalize(jnp.array([0.0, 0.0, 0.0])).tolist()

# 3. compute_reflection_direction
incident = jnp.array([1.0, -1.0, 0.0]) / jnp.sqrt(2.0)
normal = jnp.array([0.0, 1.0, 0.0])
refs["reflection"] = compute_reflection_direction(incident, normal).tolist()

# 4. create_local_frame
frame_z = create_local_frame(jnp.array([0.0, 0.0, 1.0]))
refs["local_frame_z"] = frame_z.tolist()
frame_x = create_local_frame(jnp.array([1.0, 0.0, 0.0]))
refs["local_frame_x"] = frame_x.tolist()

# 5. solve_rayleigh_inverse_cdf
refs["rayleigh_0.5"] = float(solve_rayleigh_inverse_cdf(0.5))
refs["rayleigh_0.0"] = float(solve_rayleigh_inverse_cdf(0.0))
refs["rayleigh_1.0"] = float(solve_rayleigh_inverse_cdf(1.0))

# 6. jax_rotate_vector
vec = jnp.array([1.0, 0.0, 0.0])
axis = jnp.array([0.0, 0.0, 1.0])
angle = jnp.pi / 2
refs["rotate_90_z"] = jax_rotate_vector(vec, axis, angle).tolist()

# 7. Stochastic: sample_scatter_distance
key = jax.random.PRNGKey(42)
refs["scatter_dist_D5_S2"] = float(sample_scatter_distance(5.0, 2.0, key))

# 8. Stochastic: compute_scatter_direction
key2 = jax.random.PRNGKey(42)
inc_dir = jnp.array([0.0, 0.0, 1.0])
refs["scatter_dir"] = compute_scatter_direction(inc_dir, key2).tolist()

# 9. Stochastic: sample_cosine_hemisphere
key3 = jax.random.PRNGKey(42)
nrm = jnp.array([0.0, 0.0, 1.0])
refs["cosine_hemi"] = sample_cosine_hemisphere(nrm, key3).tolist()

# ── Losses ───────────────────────────────────────────────────────────
from tools.losses import poisson_nll
from tools.optimization.losses import (
    energy_loss, counts_loss, origin_time_loss, first_arrival_nll,
    segment_logsumexp,
)

true_q = jnp.array([10.0, 5.0, 0.0, 3.0, 8.0])
pred_q = jnp.array([9.0, 6.0, 0.1, 2.5, 7.5])

refs["poisson_nll"] = float(poisson_nll(true_q, pred_q))
refs["energy_loss"] = float(energy_loss(pred_q, true_q))
refs["counts_loss"] = float(counts_loss(true_q, pred_q))

# segment_logsumexp
data = jnp.array([1.0, 2.0, 3.0, 0.5, 1.5])
indices = jnp.array([0, 0, 1, 1, 2])
refs["segment_logsumexp"] = segment_logsumexp(data, indices, 3).tolist()

# ── DetectorParams / ParticleParams ──────────────────────────────────
from tools.detector_params import (
    DetectorParams, ParticleParams,
    normalize_params, denormalize_params, default_bounds,
)

dp = DetectorParams(
    scatter_length=50.0,
    wall_reflection_rate=0.5,
    sensor_reflection_rate=0.3,
    absorption_length=100.0,
    qe=0.2,
    qe_corrections=jnp.ones(10),
)
refs["detector_params_fields"] = [
    float(dp.scatter_length), float(dp.wall_reflection_rate),
    float(dp.sensor_reflection_rate), float(dp.absorption_length),
    float(dp.qe),
]

pp = ParticleParams(energy=500.0, position=jnp.array([0.0, 0.0, 0.0]),
                    theta=0.5, phi=1.0, t0=0.0)
refs["particle_direction"] = pp.direction.tolist()

# normalize/denormalize round-trip
bounds_min, bounds_max = default_bounds(10)
normed = normalize_params(dp, bounds_min, bounds_max)
back = denormalize_params(normed, bounds_min, bounds_max)
refs["round_trip_scatter"] = float(back.scatter_length)
refs["round_trip_qe"] = float(back.qe)
refs["default_bounds_scatter_min"] = float(bounds_min.scatter_length)
refs["default_bounds_scatter_max"] = float(bounds_max.scatter_length)

# ── Utils ────────────────────────────────────────────────────────────
from tools.utils import spherical_to_cartesian, smear_times, smear_charges_SK_like

refs["sph_to_cart_0_0"] = spherical_to_cartesian(0.0, 0.0).tolist()
refs["sph_to_cart_pi2_0"] = spherical_to_cartesian(jnp.pi/2, 0.0).tolist()
refs["sph_to_cart_pi2_pi2"] = spherical_to_cartesian(jnp.pi/2, jnp.pi/2).tolist()

# smear with fixed seed
key_s = jax.random.PRNGKey(42)
times = jnp.array([100.0, 200.0, 300.0, 1e6])
refs["smear_times"] = smear_times(times, 0.4, key_s).tolist()

key_c = jax.random.PRNGKey(42)
counts = jnp.array([5.0, 15.0, 50.0, 200.0])
refs["smear_charges"] = smear_charges_SK_like(counts, key_c).tolist()

# ── Geometry ─────────────────────────────────────────────────────────
from tools.geometry import generate_detector

det_cyl = generate_detector("config/WCTE_geom_config.json")
refs["cyl_num_sensors"] = det_cyl.n_sensors
refs["cyl_all_points_shape"] = list(det_cyl.all_points.shape)
refs["cyl_first5"] = det_cyl.all_points[:5].tolist()
refs["cyl_last5"] = det_cyl.all_points[-5:].tolist()

# bounds check: inside, outside, boundary
from tools.propagation.cylinder import cylinder_bounds_check
test_pts = jnp.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [1.9, 0.0, 0.0]])
bounds_result = cylinder_bounds_check(test_pts, det_cyl.r, det_cyl.H)
refs["cyl_bounds"] = bounds_result.tolist()

# ── make_hits_simulation ─────────────────────────────────────────────
n_det = 20
flat_w = jnp.array([0.5, 0.3, 0.8, 0.1, 0.6])
flat_i = jnp.array([0, 2, 5, 5, 10])
flat_t = jnp.array([10.0, 15.0, 12.0, 20.0, 8.0])
qe_corr = jnp.ones(n_det)
q_sim, t_sim = make_hits_simulation(flat_w, flat_i, flat_t, n_det, qe=0.2,
                                     qe_corrections=qe_corr, temperature=0.01)
refs["hits_sim_charge"] = q_sim.tolist()
refs["hits_sim_time"] = t_sim.tolist()

# make_hits_likelihood
log_w, safe_t, fi, total_q = make_hits_likelihood(flat_w, flat_i, flat_t, n_det, qe=0.2,
                                                    qe_corrections=qe_corr)
refs["hits_like_log_w"] = log_w.tolist()
refs["hits_like_total_q"] = total_q.tolist()

# ── Photon iteration (stochastic with fixed seed) ────────────────────
key_pi = jax.random.PRNGKey(42)
pi_args = dict(
    position=jnp.array([0.5, 0.5, 0.5]),
    direction=jnp.array([0.0, 0.0, 1.0]),
    time=0.0,
    surface_distance=1.5,
    normal=jnp.array([0.0, 0.0, -1.0]),
    scatter_length=50.0,
    wall_reflection_rate=0.5,
    sensor_reflection_rate=0.3,
    absorption_length=100.0,
    hit_sensor=True,
    rng_key=key_pi,
    speed_of_light=0.2253,
)
sample_out = photon_iteration_sample(**pi_args)
refs["pi_sample"] = [x.tolist() if hasattr(x, 'tolist') else float(x) for x in sample_out]

update_out = photon_iteration_update_factors(**pi_args)
refs["pi_update"] = [x.tolist() if hasattr(x, 'tolist') else float(x) for x in update_out]

# ── Write out ────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "reference_values.json")
with open(out_path, "w") as f:
    json.dump(refs, f, indent=2)

print(f"Captured {len(refs)} reference values → {out_path}")
