"""B3/B2 confirmation: the energy gradient through the IMPORTANCE emitter (siren_rays.py, the one
main ships) is pathwise — sign-reliable and AD==FD — NOT the score-noise-sign-unreliable behaviour
the old plan attributed to the (now-removed) DiCE-score path.

For one event: at truth and at E offsets ±150 MeV, compute dL/dE via AD (model.grad) and central FD
(model.loss), averaged over K keys. Check (a) AD≈FD, (b) sign points toward truth (negative above
truth, positive below), (c) variance shrinks with K (finite-sample, not score)."""
import os, sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_unification')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.fitting import ReconModel, track_from_vec9
from lucid.fitting.recon import vec9_from_track
ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM = 'config/SK_like_geom_config.json'; PHYS = 'config/SK_like_physics_config.json'
K, NBUF, GRID = 8, 400_000, dict(n_cap=80, n_angular=120, n_height=80)
det = generate_detector(GEOM); ND = len(det.all_points)
dp = load_detector_params(PHYS, num_sensors=ND)._replace()
dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                 physics_config=PHYS, default_detector_params=dp, particle='muon',
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, 250_000, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
ev = 0
raw = read_photon_data_from_photonsim(ROOT, ev)
E0 = float(raw['energy']); vtx = np.array([0., 0., 0.]); d = np.array([0., 0., 1.])
th9 = vec9_from_track(E0, vtx, d, t0=0.)
n = int(raw['photon_origins'].shape[0]); reps = int(np.ceil(NBUF / n))
tile = lambda a: jnp.asarray(np.tile(np.asarray(a), (reps,) + (1,) * (np.asarray(a).ndim - 1))[:NBUF])
pd = {'photon_origins': tile(raw['photon_origins']), 'photon_directions': tile(raw['photon_directions']),
      'photon_times': tile(raw['photon_times']),
      'N': jnp.asarray(n), 'apply_rotation': False, 'rotation_axis': jnp.array([1., 0., 0.]),
      'rotation_angle': jnp.array(0.), 'apply_translation': False, 'translation_vector': jnp.zeros(3)}
if 'wavelengths' in raw: pd['wavelengths'] = tile(raw['wavelengths'])
c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000 + ev), pd))
oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)


def dE_ad(th, keys):     # AD energy-gradient (component 0), averaged over keys
    return np.mean([float(np.asarray(model.grad(th, oc, ot, k))[0]) for k in keys])


def dE_fd(th, keys, h=20.):
    out = []
    for k in keys:
        thp = th.copy(); thp[0] += h; thm = th.copy(); thm[0] -= h
        out.append((float(model.loss(thp, oc, ot, k)) - float(model.loss(thm, oc, ot, k))) / (2 * h))
    return np.mean(out)


print(f'event {ev}: truth E={E0:.0f} MeV, n_hit={(oc>0).sum()}')
print(f'{"E (MeV)":>9s} {"K":>3s} | {"dL/dE AD":>10s} {"dL/dE FD":>10s} {"AD/FD":>7s} {"sign->truth?":>12s}')
for dEoff in (-150., 0., +150.):
    th = th9.copy().astype(float); th[0] = E0 + dEoff
    for Kk in (1, 4, 16):
        keys = [jax.random.PRNGKey(s) for s in range(Kk)]
        a, f = dE_ad(th, keys), dE_fd(th, keys)
        # descent (θ -= lr·dL/dθ) moves toward truth iff sign(dL/dE) == sign(offset)
        toward = 'n/a' if dEoff == 0 else ('YES' if np.sign(a) == np.sign(dEoff) else 'NO')
        print(f'{E0+dEoff:9.0f} {Kk:3d} | {a:10.4f} {f:10.4f} {a/f if f else np.nan:7.3f} {toward:>12s}')
