"""hello_reconstruct — full track reconstruction via lucid.fitting.recon.

Mirrors LUCiD_recon's latest case (gn_fisher_recon.py / RECO_PIPELINE.md), now as library
code: a 9-parameter [E, x, y, z, dir, t0] Fisher-Gauss-Newton fit on a Poisson-charge +
first-arrival ORDER-STATISTIC-time loss, SCALE9-preconditioned, FD Jacobians (the DiCE
custom_vjp blocks jacfwd, and the autodiff track-Hessian is indefinite, so a PSD Fisher
metric is built and stepped against). AMP_DETACH, min-grad readout — all in `lucid.fitting`.

Self-contained-demo simplifications: SK_like geometry (not the measured SK .npz) and
SIREN-SAMPLED truth (no GEANT4 ROOT) — exercises the optimizer + loss + its self-consistent
floor, not the GEANT4-vs-SIREN cone mismatch that sets the ~13 cm physics floor (§6). TTS is
now baked via dp.response.tts (post de-env), not an env var. Fast on GPU (~2-3 min).
Run: python examples/hello_reconstruct.py
"""
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import make_sim_pair
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_from_track, vec9_dir, track_from_vec9, SCALE9

GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
GRID = dict(n_cap=40, n_angular=80, n_height=40)
ND = len(generate_detector(GEOM).all_points)

# soft per-photon predictor + hard sampled-data sim, sharing config (make_sim_pair guarantees
# both get physics_config — a dropped one mis-normalises the data ~6x and runs energy away).
# Data carries 2.5 ns per-photon TTS via dp.response.tts (post de-env: a field, not an env var).
dp_data = load_detector_params(PHYS, num_sensors=ND)
dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(2.5)))
pred, data_sim = make_sim_pair(GEOM, 250_000, K=K, particle='muon', physics_config=PHYS,
                               default_detector_params=True, hard_detector_params=dp_data,
                               wavelength_mode=True, **GRID)

model = ReconModel(pred, ND, sigma=2.5, delta=1.0)

# truth muon (1050 MeV, fiducial vertex+direction) -> sampled data -> offset start -> recover
th9 = vec9_from_track(1050., position=[1.5, -0.8, 2.0], direction=[0.3, 0.1, 0.95], t0=0.)
c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(0)))
oc = jnp.asarray(np.asarray(c)); ot = jnp.asarray(np.where(np.asarray(c) > 0, np.asarray(t), 0.))

start = th9 + 2.5 * SCALE9 * np.random.default_rng(1).uniform(-1, 1, 9)   # from-scratch start (~0.7 m off)
res = fit_track(model, oc, ot, start, nkeys=2, niters=130)
td = vec9_dir(th9)
sv = np.linalg.norm((start - th9)[1:4]) * 100; sd = np.degrees(np.arccos(np.clip(vec9_dir(start) @ td, -1, 1)))
fv = np.linalg.norm((res - th9)[1:4]) * 100; fd = np.degrees(np.arccos(np.clip(vec9_dir(res) @ td, -1, 1)))
print(f'start  vtx {sv:5.1f} cm   E {start[0]-1050:+6.0f} MeV   dir {sd:4.1f}°   t0 {start[8]:+5.2f} ns')
print(f'fit    vtx {fv:5.1f} cm   E {res[0]-1050:+6.1f} MeV   dir {fd:4.2f}°   t0 {res[8]:+5.2f} ns   (truth 1050 MeV)')
