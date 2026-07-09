"""seed_reconstruct — DATA-DRIVEN reconstruction (no perturbed truth) via a robust two-start seed.

Unlike hello_reconstruct (which starts from a perturbed truth), this builds the initial guess
from the observed charge+time ONLY, using the tracking seeder, then refines with the
Fisher-Gauss-Newton fit. To avoid the single-seed "wanderer" failure (a mis-multilaterated t0
dropping the fit into a wrong basin), it uses **two complementary seeds** and keeps the better
basin — the same recipe as the `track_optimization` tutorial:

  Stage 0  energy      — total-charge scan at the origin                          [forward]
  Seed A   charge ring — hierarchical position grid search + cone direction        [forward]
  Seed B   arrival time — time multilateration (seed_vertex_time) + cone direction  [geometric]
  Refine   fit_track_multistart(model, [A, B])  -> keeps the lower-loss basin.

Run: python examples/seed_reconstruct.py
"""
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import (ReconModel, fit_track_multistart, seed_vertex_time,
                           vec9_from_track, vec9_dir, track_from_vec9)
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds
from lucid.optimization.utils.functions import (hierarchical_direction_search_cone,
                                                energy_scan_optimization)

GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
GRID = dict(n_cap=80, n_angular=120, n_height=80)
det = generate_detector(GEOM); ND = len(det.all_points)
POS = np.asarray(det.all_points); bounds = get_detector_bounds(det)

pred = setup_event_simulator(GEOM, 250_000, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
dp_data = load_detector_params(PHYS, num_sensors=ND)
dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, 250_000, temperature=None, K=K, use_expected_value=False,
                                 hit_mode='realistic', apply_smearing=False, particle='muon',
                                 physics_config=PHYS, default_detector_params=dp_data,
                                 wavelength_mode=True, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)

# truth track + sampled data (the ONLY thing the seeder sees: oc, ot)
TE, TPOS, TDIR = 1050., [1.5, -0.8, 2.0], [0.3, 0.1, 0.95]
th9 = vec9_from_track(TE, TPOS, TDIR, t0=0.); td = vec9_dir(th9)
c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(0)))
oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
ocf, otf, POSf = jnp.asarray(oc), jnp.asarray(ot), jnp.asarray(POS)
print(f'data: {(oc>0).sum()} hit PMTs, total charge {oc.sum():.0f} pe\n', flush=True)

# Stage 0 — energy at the origin (total-charge scan)
e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1/jnp.sqrt(3)), jnp.pi/4, 0.,
                              POSf, otf, ocf, (ocf, otf), 1000., 700., 12, 0)['best_energy']

def make_seed(vtx, t0g):
    """energy e0 + given vertex/t0 + a cone-searched direction -> a vec9 seed."""
    c2 = hierarchical_direction_search_cone(pred, jnp.asarray(vtx), t0g, POSf, otf, ocf,
                                            (ocf, otf), e0, 3, 8, 90., 0.5, 0)
    dg = np.array([np.sin(c2['best_theta'])*np.cos(c2['best_phi']),
                   np.sin(c2['best_theta'])*np.sin(c2['best_phi']), np.cos(c2['best_theta'])])
    return vec9_from_track(e0, np.asarray(vtx), dg, t0=t0g)

# Seed A: charge-ring grid search;  Seed B: arrival-time multilateration
p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.zeros(3), 0., 0., bounds,
                                       n_div=5, t0_n_div=5, levels=6, verbosity=0)
seedA = make_seed(np.asarray(p1['best_position']), float(p1['best_t0']))
seedB = make_seed(*seed_vertex_time(POS, oc, ot))
for name, s in [('A charge-ring', seedA), ('B arrival-time', seedB)]:
    dv = np.linalg.norm(s[1:4] - th9[1:4]) * 100
    dd = np.degrees(np.arccos(np.clip(vec9_dir(s) @ td, -1, 1)))
    print(f'seed {name:14s}: vtx |Δ| {dv:5.1f} cm  dir {dd:4.1f}°  (E {e0:.0f} MeV)')

# Refine: two-start keeps the better basin
res, MS = fit_track_multistart(model, oc, ot, [seedA, seedB], nkeys=4, niters=130)
fv = np.linalg.norm((res - th9)[1:4]) * 100
fd = np.degrees(np.arccos(np.clip(vec9_dir(res) @ td, -1, 1)))
print(f'\nREFINED (seed -> fit): winning seed {"AB"[MS["which"]]}  |  '
      f'vtx {fv:5.1f} cm   E {res[0]-TE:+6.1f} MeV   dir {fd:4.2f}°   t0 {res[8]:+.2f} ns   (truth {TE:.0f} MeV)')
