"""Debug: find photons where old/new charges differ at K=0."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_platform_name", "cpu")

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.propagator import create_string_propagator
from lucid.overlap import create_overlap_prob

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
det = StringTelescope.from_npz(os.path.join(CONFIG_DIR, "icecube86_simple.npz"))
sp = jnp.array(det.all_points)
TEMPERATURE = 0.2
N = 50_000
n_closest = 4
n_dom_snap = 2

z_mid = (det.envelope_z_min + det.envelope_z_max) / 2
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
origins = jax.random.uniform(k1, (N, 3),
    minval=jnp.array([-100, -100, z_mid - 200]),
    maxval=jnp.array([100, 100, z_mid + 200]))
dirs_raw = jax.random.normal(k2, (N, 3))
dirs = dirs_raw / (jnp.linalg.norm(dirs_raw, axis=1, keepdims=True) + 1e-10)

# --- OLD: get K=0 per-photon charges ---
print("Running OLD propagator...")
prop = create_string_propagator(det, sp, det.S_radius, temperature=TEMPERATURE)
old_result = prop(origins, dirs)
old_w = np.array(old_result['sensor_weights'])       # (36, N)
old_idx = np.array(old_result['sensor_indices'])     # (36, N)
old_d = np.array(old_result['sensor_distances']).squeeze(-1)  # (36, N)

old_charge_per_photon = np.zeros(N)
for s in range(old_w.shape[0]):
    valid = (old_idx[s] >= 0) & (old_w[s] > 1e-15)
    reach = np.exp(-np.maximum(old_d[s], 0) / 30.0)
    old_charge_per_photon += np.where(valid, old_w[s] * reach, 0.0)

# --- NEW: compute K=0 charges manually (same geometry lookups as fast.py) ---
print("Computing NEW charges...")
tables = det.get_jax_tables()
string_anchors = tables['string_anchors']
dom_s_offsets = tables['dom_s_offsets']
dom_global_ids = tables['dom_global_ids']
string_s_min = tables['string_s_min']
string_s_max = tables['string_s_max']
n_dom_per_str = jnp.array(det.n_dom_per_str_np)
string_dz = (string_s_max - string_s_min) / jnp.maximum(n_dom_per_str - 1, 1)
sensor_positions = jnp.array(det.all_points)
overlap_fn = create_overlap_prob(TEMPERATURE * det.S_radius, det.S_radius)
anch0, anch1, anch2 = string_anchors[:, 0], string_anchors[:, 1], string_anchors[:, 2]

d0, d1, d2 = dirs[:, 0], dirs[:, 1], dirs[:, 2]
o0, o1, o2 = origins[:, 0], origins[:, 1], origins[:, 2]

# Vertical specialization
dxy_sq = d0**2 + d1**2
od_cross = o0 * d1 - o1 * d0
scalar_triple = (od_cross[:, None] + d0[:, None] * anch1[None, :] - d1[:, None] * anch0[None, :])
dist_all = jnp.abs(scalar_triple) / jnp.sqrt(dxy_sq[:, None] + 1e-18)

_, top_idx = jax.lax.top_k(-jnp.array(dist_all), n_closest)

# s_string
dd = dxy_sq + d2**2
sel_wa = o2[:, None] - anch2[top_idx]
sel_wd = ((o0[:, None] - anch0[top_idx]) * d0[:, None] +
          (o1[:, None] - anch1[top_idx]) * d1[:, None] +
          sel_wa * d2[:, None])
sel_denom = dxy_sq[:, None] + 1e-9
sel_s = (sel_wa * dd[:, None] - sel_wd * d2[:, None]) / sel_denom

# DOM snap
sel_offsets = dom_s_offsets[top_idx]
below = sel_offsets <= sel_s[:, :, None]
k_right = below.sum(axis=-1)
max_dom = dom_s_offsets.shape[1]
k_start = jnp.clip(k_right - n_dom_snap // 2, 0, max_dom - n_dom_snap)
dom_local = k_start[:, :, None] + jnp.arange(n_dom_snap)[None, None, :]
dom_local = jnp.clip(dom_local, 0, max_dom - 1)
cand_ids = dom_global_ids[top_idx[:, :, None], dom_local].reshape(N, n_closest * n_dom_snap)

# Ray-DOM
cand_pos = sensor_positions[cand_ids]
oc = origins[:, None, :] - cand_pos
d_norm = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10)
t_closest = -jnp.sum(oc * d_norm[:, None, :], axis=-1)
closest_pts = origins[:, None, :] + t_closest[:, :, None] * d_norm[:, None, :]
perp_dist = jnp.sqrt(jnp.sum((closest_pts - cand_pos)**2, axis=-1) + 1e-18)

ov = jax.vmap(jax.vmap(overlap_fn))(perp_dist)
valid_mask = cand_ids >= 0
ov = jnp.where(valid_mask, ov, 0.0)
safe_t = jnp.maximum(t_closest, 0.0)
reach = jnp.exp(-safe_t / 30.0)
new_per_dom = ov * reach
new_charge_per_photon = np.array(jnp.sum(new_per_dom, axis=1))

# --- Compare ---
print(f"\nOLD total K=0: {old_charge_per_photon.sum():.6f}")
print(f"NEW total K=0: {new_charge_per_photon.sum():.6f}")
print(f"Ratio: {new_charge_per_photon.sum() / (old_charge_per_photon.sum() + 1e-30):.2f}")

# Photons with nonzero charge in either
has_old = old_charge_per_photon > 1e-10
has_new = new_charge_per_photon > 1e-10
print(f"\nPhotons with charge: OLD={has_old.sum()}, NEW={has_new.sum()}")
print(f"  Both: {(has_old & has_new).sum()}")
print(f"  OLD only: {(has_old & ~has_new).sum()}")
print(f"  NEW only: {(~has_old & has_new).sum()}")

# Show specific discrepant photons
both = has_old & has_new
if both.sum() > 0:
    ratios = new_charge_per_photon[both] / (old_charge_per_photon[both] + 1e-30)
    print(f"\nFor photons with charge in BOTH:")
    print(f"  Charge ratio NEW/OLD: median={np.median(ratios):.4f}, "
          f"mean={np.mean(ratios):.4f}, min={ratios.min():.4f}, max={ratios.max():.4f}")

# Analyze ALL NEW-only photons
new_only_mask = ~has_old & has_new
new_only_total = new_charge_per_photon[new_only_mask].sum()
print(f"\nNEW-only total charge: {new_only_total:.6f}")

# Check how much comes from behind DOMs
t_closest_np = np.array(t_closest)
behind_charge = np.array(jnp.where((t_closest < 0) & valid_mask, new_per_dom, 0.0))
total_behind = behind_charge.sum()
total_forward = np.array(jnp.where((t_closest >= 0) & valid_mask, new_per_dom, 0.0)).sum()
print(f"Total charge from BEHIND DOMs (t<0): {total_behind:.6f}")
print(f"Total charge from AHEAD DOMs (t>=0):  {total_forward:.6f}")

# Per-photon behind charge
behind_per_photon = behind_charge.sum(axis=1)
n_behind = (behind_per_photon > 1e-10).sum()
print(f"Photons with behind-DOM charge: {n_behind}")
print(f"Behind charge fraction: {total_behind / (total_behind + total_forward + 1e-30):.1%}")

# Show a few NEW-only photons with actual charge
new_only_idx = np.where(~has_old & has_new)[0][:5]
for idx in new_only_idx:
    print(f"\n  NEW-only photon {idx}: charge={new_charge_per_photon[idx]:.6f}")
    print(f"    pos={np.array(origins[idx])}, dir={np.array(dirs[idx])}")
    print(f"    dxy_sq={float(dxy_sq[idx]):.8f}")
    for j in range(n_closest * n_dom_snap):
        cid = int(cand_ids[idx, j])
        if cid >= 0 and float(new_per_dom[idx, j]) > 1e-15:
            tc = float(t_closest[idx, j])
            pd = float(perp_dist[idx, j])
            o_val = float(ov[idx, j])
            r_val = float(reach[idx, j])
            dom_pos = np.array(sensor_positions[cid])
            print(f"    DOM {cid}: pos={dom_pos}, perp_d={pd:.4f}m, t_closest={tc:.1f}m, "
                  f"overlap={o_val:.6f}, reach={r_val:.6f}, "
                  f"{'BEHIND' if tc < 0 else 'ahead'}")
