
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
