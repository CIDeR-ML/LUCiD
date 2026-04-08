
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
