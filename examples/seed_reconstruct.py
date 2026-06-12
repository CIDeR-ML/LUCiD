"""seed_reconstruct — DATA-DRIVEN initial guess (the tracking-notebook seeder) vs truth.

Instead of starting the fit from a perturbed truth, this runs the real initial-guess stages
from `lucid.optimization` (the 5-stage pipeline's seeding half, as in
good_notebooks/tracking_opt_development.ipynb) on the observed charge+time ONLY:
  Stage 0  energy   — scan energy at the origin (total-charge match)         [forward]
  Stage 1  vertex+t0 — hierarchical grid search on light-arrival timing       [geometric, no fwd]
  Stage 2  direction — hierarchical cone search (charge-ring match)           [forward]
then refines energy at the found pose. Reports each GUESS vs truth, and finally feeds the
guess as the start to lucid.fitting.recon.fit_track (the honest pipeline: seed -> refine).
Run: python examples/seed_reconstruct.py
"""
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import make_sim_pair
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_from_track, vec9_dir, track_from_vec9
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds
from lucid.optimization.utils.functions import (
    hierarchical_direction_search_cone, energy_scan_optimization)

GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
GRID = dict(n_cap=40, n_angular=80, n_height=40)
det = generate_detector(GEOM); ND = len(det.all_points)
POS = np.asarray(det.all_points); bounds = get_detector_bounds(det)

dp_data = load_detector_params(PHYS, num_sensors=ND)
dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(2.5)))
pred, data_sim = make_sim_pair(GEOM, 250_000, K=K, particle='muon', physics_config=PHYS,
                               default_detector_params=True, hard_detector_params=dp_data,
                               wavelength_mode=True, **GRID)

# truth track + sampled data (the ONLY thing the seeder sees: oc, ot at hit PMTs)
TE, TPOS, TDIR = 1050., [1.5, -0.8, 2.0], [0.3, 0.1, 0.95]
th9 = vec9_from_track(TE, TPOS, TDIR, t0=0.); td = vec9_dir(th9)
c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(0)))
oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
hit = oc > 0
# hit-subset (for the GEOMETRIC vertex search) vs FULL per-PMT (for the FORWARD-based stages,
# whose counts_loss/energy_loss compare to the full (n_sensors,) model charge).
hp, otimes, ocharge = jnp.asarray(POS[hit]), jnp.asarray(ot[hit]), jnp.asarray(oc[hit])
oc_full, ot_full, POS_full = jnp.asarray(oc), jnp.asarray(ot), jnp.asarray(POS)
print(f'data: {hit.sum()} hit PMTs, total charge {oc.sum():.0f} pe\n', flush=True)

# Stage 0 — energy at origin (std direction), total-charge scan [forward; full arrays]
e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1/jnp.sqrt(3)), jnp.pi/4, 0.,
                              POS_full, ot_full, oc_full, (oc_full, ot_full), energy_guess=1000.,
                              energy_delta=700., n_steps=12, verbosity=0)['best_energy']
# Stage 1 — vertex + t0 from arrival-time geometry [geometric, no forward; hit-subset]
p1 = hierarchical_position_grid_search(hp, otimes, ocharge, jnp.asarray(TPOS), 0.0, 0.0,
                                       bounds, n_div=5, t0_n_div=5, levels=6, verbosity=0)
vtx, t0g = np.asarray(p1['best_position']), float(p1['best_t0'])
# Stage 2 — direction by cone search at the found vertex/t0/energy [forward; full arrays]
c2 = hierarchical_direction_search_cone(pred, jnp.asarray(vtx), t0g, POS_full, ot_full, oc_full,
                                        (oc_full, ot_full), e0, levels=4, initial_div=12,
                                        max_angle_deg=90., reduction=0.5, verbosity=0)
dirg = np.asarray([np.sin(c2['best_theta'])*np.cos(c2['best_phi']),
                   np.sin(c2['best_theta'])*np.sin(c2['best_phi']), np.cos(c2['best_theta'])])
# Stage 3 — refine energy at the found pose [forward; full arrays]
e3 = energy_scan_optimization(pred, jnp.asarray(vtx), c2['best_theta'], c2['best_phi'], t0g,
                              POS_full, ot_full, oc_full, (oc_full, ot_full), energy_guess=e0,
                              energy_delta=400., n_steps=12, verbosity=0)['best_energy']

dv = np.linalg.norm(vtx - np.asarray(TPOS)) * 100
dd = np.degrees(np.arccos(np.clip(dirg @ td, -1, 1)))
print('GUESS (from data only) vs truth:')
print(f'  energy   {e3:7.0f} MeV   (truth {TE:.0f},  err {e3-TE:+.0f})')
print(f'  vertex   |Δ| {dv:5.1f} cm  guess {np.round(vtx,2)}  truth {TPOS}')
print(f'  t0       {t0g:+6.2f} ns   (truth 0.0)')
print(f'  direction {dd:5.1f}°  off truth\n', flush=True)

# honest pipeline: refine the GUESS with the Fisher-GN (NOT a perturbed truth)
start = vec9_from_track(e3, vtx, dirg, t0=t0g)
res = fit_track(ReconModel(pred, ND, sigma=2.5, delta=1.0), oc, ot, start, nkeys=2, niters=130)
fv = np.linalg.norm((res - th9)[1:4]) * 100; fd = np.degrees(np.arccos(np.clip(vec9_dir(res) @ td, -1, 1)))
print(f'REFINED (seed -> fit_track): vtx {fv:5.1f} cm  E {res[0]-TE:+6.1f} MeV  dir {fd:4.2f}°  t0 {res[8]:+.2f} ns')
