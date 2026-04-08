#!/usr/bin/env python3
"""
TEST 20: Compare data_sim and pred_sim hit counts and timing between clean and v2.

From test18b, v2 data hits 2941 sensors but clean data hits a different set.
They use the SAME ROOT photons. Something in the propagation loop differs.

Key _common_propagation differences:
1. Clean has NaN protection (recovers NaN photons)
2. Clean uses stop_gradient on survival everywhere; v2 is conditional
3. Clean applies stop_gradient on positions/directions always; v2 is conditional on n_grad_iters
4. V2 uses jax.remat on propagation_step; clean doesn't

#2,3,4 only affect backward pass. #1 could affect forward pass if NaNs occur.

This test counts exact hit differences between clean and v2 data/pred sims.
"""
import sys, os, subprocess, json

# Run v2
v2_script = '''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import jax, jax.numpy as jnp, numpy as np, json
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.detector_params import ParticleParams
from tools.generate import read_photon_data_from_photonsim

BASE = os.path.join(os.path.dirname(__file__), '..')
GEOM = os.path.join(BASE, 'config/SK_geom_config.json')
PHYS = os.path.join(BASE, 'config/SK_physics_config.json')
DATA = os.path.join(BASE, 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')
detector = generate_detector(GEOM)

def load_and_pad(entry_idx):
    pd = read_photon_data_from_photonsim(DATA, entry_idx)
    N = len(pd['photon_origins'])
    pad = max(0, 1_000_000 - N)
    pd['photon_origins'] = jnp.pad(pd['photon_origins'], ((0,pad),(0,0)), constant_values=0)
    dd = jnp.array([0.,0.,1.])
    if pad > 0:
        pd['photon_directions'] = jnp.concatenate([pd['photon_directions'], jnp.tile(dd, (pad,1))])
    pd['photon_times'] = jnp.pad(pd['photon_times'], (0,pad), constant_values=0)
    pd['N'] = N
    pd['apply_rotation'] = jnp.array(False)
    pd['rotation_axis'] = jnp.array([1.,0.,0.])
    pd['rotation_angle'] = jnp.array(0.)
    return pd

def set_transform(pd, pp):
    orig = jnp.array([0.,0.,1.])
    tgt = pp.direction / (jnp.linalg.norm(pp.direction)+1e-8)
    ax = jnp.cross(orig, tgt); an = jnp.linalg.norm(ax)
    ax = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax; pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = pp.position
    pd['apply_translation'] = jnp.array(True)
    return pd

pos = jnp.array([0., 0., 0.])
pp = ParticleParams(energy=jnp.array(1050.), position=pos, theta=jnp.array(1.0), phi=jnp.array(0.5), t0=jnp.array(0.))

data_sim = setup_event_simulator(GEOM, 300_000, temperature=0.0, K=20, is_data=True, is_calibration=False, physics_config=PHYS, default_detector_params=True)
pred_sim = setup_event_simulator(GEOM, 300_000, temperature=0.10, K=9, is_data=False, max_sensors_per_cell=4, physics_config=PHYS, default_detector_params=True)

pd = load_and_pad(0); pd = set_transform(pd, pp)
key_d = jax.random.PRNGKey(100)
data_q, data_t = jax.lax.stop_gradient(data_sim(pp, key_d, pd))
key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(pp, key_p)

result = {
    "data_nhit": int(jnp.sum(data_q > 0)),
    "data_total_q": float(jnp.sum(data_q)),
    "data_mean_t": float(jnp.mean(data_t[data_q > 0])),
    "pred_nhit": int(jnp.sum(pred_q > 0)),
    "pred_total_q": float(jnp.sum(pred_q)),
    "pred_mean_t": float(jnp.mean(pred_t[pred_q > 0])),
}
np.savez(os.path.join(os.path.dirname(__file__), 'v2_test20.npz'),
         data_q=np.array(data_q), data_t=np.array(data_t),
         pred_q=np.array(pred_q), pred_t=np.array(pred_t))
print(json.dumps(result))
'''

# Write and run v2 script
script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, '_run_v2.py'), 'w') as f:
    f.write(v2_script)

print("Running V2...")
r = subprocess.run(['python3', os.path.join(script_dir, '_run_v2.py')], capture_output=True, text=True, timeout=300)
if r.returncode != 0:
    print("V2 STDERR:", r.stderr[-500:] if r.stderr else "")
    sys.exit(1)
v2_result = json.loads(r.stdout.strip().split('\n')[-1])
print(f"  V2: data_nhit={v2_result['data_nhit']} pred_nhit={v2_result['pred_nhit']}")
print(f"      data_total_q={v2_result['data_total_q']:.2f} pred_total_q={v2_result['pred_total_q']:.2f}")
print(f"      data_mean_t={v2_result['data_mean_t']:.4f} pred_mean_t={v2_result['pred_mean_t']:.4f}")

# Now run clean
clean_script = '''
import sys, os
CLEAN_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'clean_run', 'LUCiD')
sys.path.insert(0, CLEAN_BASE)
import jax, jax.numpy as jnp, numpy as np, json
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.generate import read_photon_data_from_photonsim

GEOM = os.path.join(CLEAN_BASE, 'config/SK_geom_config.json')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root')
detector = generate_detector(GEOM)

def load_and_pad(entry_idx):
    pd = read_photon_data_from_photonsim(DATA, entry_idx)
    N = len(pd['photon_origins'])
    pad = max(0, 1_000_000 - N)
    pd['photon_origins'] = jnp.pad(pd['photon_origins'], ((0,pad),(0,0)), constant_values=0)
    dd = jnp.array([0.,0.,1.])
    if pad > 0:
        pd['photon_directions'] = jnp.concatenate([pd['photon_directions'], jnp.tile(dd, (pad,1))])
    pd['photon_times'] = jnp.pad(pd['photon_times'], (0,pad), constant_values=0)
    pd['N'] = N
    pd['apply_rotation'] = jnp.array(False)
    pd['rotation_axis'] = jnp.array([1.,0.,0.])
    pd['rotation_angle'] = jnp.array(0.)
    return pd

def set_transform(pd, pos, th, ph):
    orig = jnp.array([0.,0.,1.])
    direction = jnp.array([jnp.sin(th)*jnp.cos(ph), jnp.sin(th)*jnp.sin(ph), jnp.cos(th)])
    tgt = direction / (jnp.linalg.norm(direction)+1e-8)
    ax = jnp.cross(orig, tgt); an = jnp.linalg.norm(ax)
    ax = jnp.where(an<1e-6, jnp.array([1.,0.,0.]), ax/(an+1e-8))
    ang = jnp.arccos(jnp.clip(jnp.dot(orig, tgt),-1.,1.))
    pd['rotation_axis'] = ax; pd['rotation_angle'] = ang
    pd['apply_rotation'] = jnp.array(True)
    pd['translation_vector'] = pos
    pd['apply_translation'] = jnp.array(True)
    return pd

pos = jnp.array([0., 0., 0.])
th, ph = jnp.array(1.0), jnp.array(0.5)
energy = jnp.array(1050.)
particle_params = (energy, pos, jnp.array([th, ph]))
detector_params = (jnp.array(100.0), jnp.array(0.2), jnp.array(60.0), jnp.array(0.065))

data_sim = setup_event_simulator(GEOM, 300_000, temperature=0.0, K=20, is_data=True, is_calibration=False)
pred_sim = setup_event_simulator(GEOM, 300_000, temperature=0.10, K=9, is_data=False, max_sensors_per_cell=4)

pd = load_and_pad(0); pd = set_transform(pd, pos, th, ph)
key_d = jax.random.PRNGKey(100)
data_q, data_t = jax.lax.stop_gradient(data_sim(particle_params, detector_params, key_d, pd))
key_p = jax.random.PRNGKey(42)
pred_q, pred_t = pred_sim(particle_params, detector_params, key_p)

result = {
    "data_nhit": int(jnp.sum(data_q > 0)),
    "data_total_q": float(jnp.sum(data_q)),
    "data_mean_t": float(jnp.mean(data_t[data_q > 0])),
    "pred_nhit": int(jnp.sum(pred_q > 0)),
    "pred_total_q": float(jnp.sum(pred_q)),
    "pred_mean_t": float(jnp.mean(pred_t[pred_q > 0])),
}
np.savez(os.path.join(os.path.dirname(__file__), 'clean_test20.npz'),
         data_q=np.array(data_q), data_t=np.array(data_t),
         pred_q=np.array(pred_q), pred_t=np.array(pred_t))
print(json.dumps(result))
'''

with open(os.path.join(script_dir, '_run_clean.py'), 'w') as f:
    f.write(clean_script)

print("\nRunning CLEAN...")
r = subprocess.run(['python3', os.path.join(script_dir, '_run_clean.py')], capture_output=True, text=True, timeout=300)
if r.returncode != 0:
    print("CLEAN STDERR:", r.stderr[-500:] if r.stderr else "")
    sys.exit(1)
clean_result = json.loads(r.stdout.strip().split('\n')[-1])
print(f"  Clean: data_nhit={clean_result['data_nhit']} pred_nhit={clean_result['pred_nhit']}")
print(f"         data_total_q={clean_result['data_total_q']:.2f} pred_total_q={clean_result['pred_total_q']:.2f}")
print(f"         data_mean_t={clean_result['data_mean_t']:.4f} pred_mean_t={clean_result['pred_mean_t']:.4f}")

# Load and compare
import numpy as np
import jax.numpy as jnp
v2 = np.load(os.path.join(script_dir, 'v2_test20.npz'))
cl = np.load(os.path.join(script_dir, 'clean_test20.npz'))

print(f"\n=== Detailed comparison ===")
print(f"  DATA nhit: v2={v2_result['data_nhit']} clean={clean_result['data_nhit']}  diff={v2_result['data_nhit']-clean_result['data_nhit']}")
print(f"  PRED nhit: v2={v2_result['pred_nhit']} clean={clean_result['pred_nhit']}  diff={v2_result['pred_nhit']-clean_result['pred_nhit']}")
print(f"  DATA total_q: v2={v2_result['data_total_q']:.2f} clean={clean_result['data_total_q']:.2f}")
print(f"  PRED total_q: v2={v2_result['pred_total_q']:.2f} clean={clean_result['pred_total_q']:.2f}")
print(f"  DATA mean_t: v2={v2_result['data_mean_t']:.4f} clean={clean_result['data_mean_t']:.4f}  diff={v2_result['data_mean_t']-clean_result['data_mean_t']:.4f}")
print(f"  PRED mean_t: v2={v2_result['pred_mean_t']:.4f} clean={clean_result['pred_mean_t']:.4f}  diff={v2_result['pred_mean_t']-clean_result['pred_mean_t']:.4f}")

# Compare sensor-by-sensor
v2_dq, v2_dt = jnp.array(v2['data_q']), jnp.array(v2['data_t'])
cl_dq, cl_dt = jnp.array(cl['data_q']), jnp.array(cl['data_t'])
v2_pq, v2_pt = jnp.array(v2['pred_q']), jnp.array(v2['pred_t'])
cl_pq, cl_pt = jnp.array(cl['pred_q']), jnp.array(cl['pred_t'])

both_data = (v2_dq > 0) & (cl_dq > 0)
v2_only_data = (v2_dq > 0) & (cl_dq == 0)
cl_only_data = (cl_dq > 0) & (v2_dq == 0)
print(f"\n  DATA sensor overlap: both={int(jnp.sum(both_data))} v2_only={int(jnp.sum(v2_only_data))} clean_only={int(jnp.sum(cl_only_data))}")

both_pred = (v2_pq > 0) & (cl_pq > 0)
print(f"  PRED sensor overlap: both={int(jnp.sum(both_pred))} v2_only={int(jnp.sum((v2_pq>0)&(cl_pq==0)))} clean_only={int(jnp.sum((cl_pq>0)&(v2_pq==0)))}")

if int(jnp.sum(both_data)) > 0:
    dd = v2_dt[both_data] - cl_dt[both_data]
    print(f"\n  DATA timing diff (v2 - clean), jointly hit:")
    print(f"    mean={float(jnp.mean(dd)):.4f} std={float(jnp.std(dd)):.4f}")
    for p in [10, 23, 50]:
        print(f"    {p}th pct: {float(jnp.percentile(dd, p)):+.4f}")

if int(jnp.sum(both_pred)) > 0:
    dp = v2_pt[both_pred] - cl_pt[both_pred]
    print(f"\n  PRED timing diff (v2 - clean), jointly hit:")
    print(f"    mean={float(jnp.mean(dp)):.4f} std={float(jnp.std(dp)):.4f}")
    for p in [10, 23, 50]:
        print(f"    {p}th pct: {float(jnp.percentile(dp, p)):+.4f}")

# Cleanup
os.remove(os.path.join(script_dir, '_run_v2.py'))
os.remove(os.path.join(script_dir, '_run_clean.py'))
