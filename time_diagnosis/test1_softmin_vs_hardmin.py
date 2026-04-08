#!/usr/bin/env python3
"""
TEST 1: Pure soft-min vs hard-min bias on synthetic data.
"""
import jax
import jax.numpy as jnp
import numpy as np

def softmin_time(flat_weights, flat_indices, flat_times, num_det,
                 qe=0.065, threshold=1e-10, temperature=0.01):
    """Reproduce make_hits_simulation logic."""
    qe_w = flat_weights * qe
    valid = (qe_w > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    filt_t = jnp.where(valid, flat_times, jnp.inf)

    det_mins = jax.ops.segment_min(filt_t, flat_indices, num_segments=num_det)
    offsets = det_mins[flat_indices]
    shifted = jnp.where(valid, flat_times - offsets, jnp.inf)
    exp_terms = jnp.where(valid, jnp.exp(-shifted / temperature), 0.0)
    exp_sums = jax.ops.segment_sum(exp_terms, flat_indices, num_segments=num_det)

    sm_time = det_mins - temperature * jnp.log(exp_sums + 1e-20)
    has = jnp.isfinite(det_mins)
    sm_time = jnp.where(has, sm_time, 0.0)

    total_q = jax.ops.segment_sum(qe_w, flat_indices, num_segments=num_det)
    hit = (total_q > threshold) & jnp.isfinite(sm_time) & (sm_time > 0)
    return jnp.where(hit, sm_time, 0.0), jnp.where(hit, total_q, 0.0)


def hardmin_time(flat_weights, flat_indices, flat_times, num_det,
                 qe=0.065, rng_key=None):
    """Reproduce make_hits_data logic — no threshold on weights, just QE sampling."""
    timing_mask = (flat_weights > 1e-10) & (flat_times > 0)

    det_probs = jax.random.uniform(rng_key, shape=flat_weights.shape)
    detected = det_probs < qe
    qe_w = flat_weights * detected.astype(jnp.float32)
    qe_t = jnp.where(detected & timing_mask, flat_times, jnp.inf)

    total_q = jax.ops.segment_sum(qe_w, flat_indices, num_segments=num_det)
    det_mins = jax.ops.segment_min(qe_t, flat_indices, num_segments=num_det)

    hit = (total_q > 1e-10) & (det_mins > 0) & jnp.isfinite(det_mins)
    return jnp.where(hit, det_mins, 0.0), jnp.where(hit, total_q, 0.0)


# ── synthetic photon arrays ─────────────────────────────────────────
rng = np.random.default_rng(42)
N_sensors = 11000
N_phot_per_sensor = 8
N_total = N_sensors * N_phot_per_sensor

indices = rng.integers(0, N_sensors, size=N_total).astype(np.int32)
weights = rng.exponential(0.02, size=N_total).astype(np.float32)
base_t  = rng.uniform(20, 80, size=N_sensors).astype(np.float32)
times   = (base_t[indices] + rng.exponential(0.5, size=N_total)).astype(np.float32)

flat_w = jnp.array(weights)
flat_i = jnp.array(indices)
flat_t = jnp.array(times)

# Reference: hard-min
key = jax.random.PRNGKey(0)
ref_t, ref_q = hardmin_time(flat_w, flat_i, flat_t, N_sensors, rng_key=key)
hit_mask = ref_q > 0
n_hit = int(jnp.sum(hit_mask))
print(f"Reference hard-min: {n_hit} sensors hit")

# Quick debug: check soft-min output
sm_t, sm_q = softmin_time(flat_w, flat_i, flat_t, N_sensors,
                           threshold=1e-10, temperature=0.01)
sm_hit = sm_q > 0
print(f"Soft-min: {int(jnp.sum(sm_hit))} sensors hit")

both = hit_mask & sm_hit
n_both = int(jnp.sum(both))
print(f"Both hit: {n_both}")

# Debug: check for inf/nan in the time arrays at hit sensors
ref_t_vals = ref_t[both]
sm_t_vals  = sm_t[both]
print(f"ref_t: min={float(jnp.min(ref_t_vals)):.4f}  max={float(jnp.max(ref_t_vals)):.4f}  "
      f"inf={int(jnp.sum(jnp.isinf(ref_t_vals)))}  nan={int(jnp.sum(jnp.isnan(ref_t_vals)))}")
print(f"sm_t:  min={float(jnp.min(sm_t_vals)):.4f}  max={float(jnp.max(sm_t_vals)):.4f}  "
      f"inf={int(jnp.sum(jnp.isinf(sm_t_vals)))}  nan={int(jnp.sum(jnp.isnan(sm_t_vals)))}")

dt = ref_t_vals - sm_t_vals
print(f"dt:    min={float(jnp.min(dt)):.4f}  max={float(jnp.max(dt)):.4f}  "
      f"mean={float(jnp.mean(dt)):.6f}  "
      f"inf={int(jnp.sum(jnp.isinf(dt)))}  nan={int(jnp.sum(jnp.isnan(dt)))}")

# ── scan threshold ───────────────────────────────────────────────────
print("\n=== Threshold scan (temperature=0.01) ===")
for thr in [1e-15, 1e-10, 1e-7, 1e-5, 1e-3]:
    sm_t2, sm_q2 = softmin_time(flat_w, flat_i, flat_t, N_sensors, threshold=thr)
    both2 = hit_mask & (sm_q2 > 0)
    n = int(jnp.sum(both2))
    vals = (ref_t[both2] - sm_t2[both2])
    # use median to avoid inf contamination
    mean_dt = float(jnp.mean(vals))
    med_dt  = float(jnp.median(vals))
    print(f"  thr={thr:<8.0e}  mean={mean_dt:+.4f}  median={med_dt:+.4f} ns  ({n} sensors)")

# ── scan temperature ─────────────────────────────────────────────────
print("\n=== Temperature scan (threshold=1e-10) ===")
for temp in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    sm_t2, sm_q2 = softmin_time(flat_w, flat_i, flat_t, N_sensors,
                                 threshold=1e-10, temperature=temp)
    both2 = hit_mask & (sm_q2 > 0)
    n = int(jnp.sum(both2))
    vals = (ref_t[both2] - sm_t2[both2])
    mean_dt = float(jnp.mean(vals))
    med_dt  = float(jnp.median(vals))
    print(f"  T={temp:<6.3f}  mean={mean_dt:+.4f}  median={med_dt:+.4f} ns  ({n} sensors)")
