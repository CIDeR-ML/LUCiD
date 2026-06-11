"""hello_reconstruct — full track reconstruction, mirroring LUCiD_recon's latest case.

This follows gn_fisher_recon.py / RECO_PIPELINE.md: a 9-parameter Fisher-Gauss-Newton fit of
the track [E, x, y, z, dir(sin/cos θ,φ), t0] on a Poisson-charge + first-arrival
ORDER-STATISTIC-time loss, run in SCALE9-preconditioned coordinates with finite-difference
Jacobians (the DiCE custom_vjp blocks jacfwd, and the autodiff track-Hessian is indefinite,
so we build a PSD Fisher metric F = Jμᵀdiag(1/μ)Jμ + JlᵀJl and step against it). AMP_DETACH:
the time term's gradient flows only through predicted times, not amplitude.

Self-contained-demo simplifications: SK_like geometry (not the measured SK .npz) and
SIREN-SAMPLED truth (build_truth_sim style) instead of the GEANT4/PhotonSim ROOT table — so
this exercises the optimizer + loss + Fisher-GN machinery and its self-consistent floor, NOT
the GEANT4-vs-SIREN cone mismatch that sets the ~13 cm physics floor (RECO_PIPELINE.md §6).
Fast on GPU (~2-3 min). Run: python examples/hello_reconstruct.py
"""
import os
os.environ.setdefault('TTS_NS', '2.5'); os.environ.setdefault('MAXCELL', '4')
import jax, jax.numpy as jnp, numpy as np
from jax.scipy.special import erf
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
K, SIGMA, DELTA, SQ = 8, 2.5, 1.0, np.sqrt(2.0)
SCALE9 = np.array([50., .2, .2, .2, .02, .02, .02, .02, .2]); FDH = 0.4 * SCALE9
GRID = dict(n_cap=40, n_angular=80, n_height=40)
ND = len(generate_detector(GEOM).all_points)

# SIREN predictor (SOFT, per-photon; gradients flow all K iterations) + sampled-truth data (HARD)
pred = setup_event_simulator(GEOM, 250_000, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
data_sim = setup_event_simulator(GEOM, 250_000, temperature=None, K=K, use_expected_value=False,
                                 hit_mode='realistic', apply_smearing=False, physics_config=PHYS,
                                 default_detector_params=True, particle='muon', wavelength_mode=True, **GRID)


def sc2(t):                                              # 9-vector -> ParticleParams (sin/cos dir)
    st, ct, sp, cp = t[4], t[5], t[6], t[7]; nt = jnp.hypot(st, ct) + 1e-12; npp = jnp.hypot(sp, cp) + 1e-12
    return ParticleParams(energy=t[0], position=t[1:4],
                          theta=jnp.arctan2(st / nt, ct / nt), phi=jnp.arctan2(sp / npp, cp / npp), t0=t[8])


def th9dir(t):
    st, ct, sp, cp = t[4], t[5], t[6], t[7]; nt = np.hypot(st, ct); npp = np.hypot(sp, cp)
    st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp; return np.array([st * cp, st * sp, ct])


def log1mexp(a):
    a = jnp.minimum(a, -1e-7); return jnp.where(a > -0.6931, jnp.log(-jnp.expm1(a)), jnp.log1p(-jnp.exp(a)))


@jax.jit
def perpmt(t9, oc, ot, key):                            # per-PMT (μ charge, first-arrival time-NLL)
    lw, ft, fi, tot = pred(sc2(t9), key)
    ww = jax.lax.stop_gradient(jnp.exp(jnp.clip(lw, -60, 20)))      # AMP_DETACH: time grad only via ft
    tobs = (ot - t9[8])[fi]; mu = jnp.maximum(tot, 1e-8); muS = jax.lax.stop_gradient(mu)
    Rlo = jax.ops.segment_sum(ww * 0.5 * (1 + erf((tobs - DELTA / 2 - ft) / (SIGMA * SQ))), fi, num_segments=ND)
    Rhi = jax.ops.segment_sum(ww * 0.5 * (1 + erf((tobs + DELTA / 2 - ft) / (SIGMA * SQ))), fi, num_segments=ND)
    Slo = jnp.clip((muS - Rlo) / muS, 1e-12, 1.); Shi = jnp.clip((muS - Rhi) / muS, 1e-12, 1.)
    n = jnp.maximum(oc, 0.); a = jnp.minimum(n * (jnp.log(Shi) - jnp.log(Slo)), -1e-9)
    tnll = jnp.where(oc > 0, -n * jnp.log(Slo) - log1mexp(a), 0.)
    return mu, tnll


@jax.jit
def lossf(t9, oc, ot, key):
    mu, tnll = perpmt(t9, oc, ot, key)
    return jnp.sum(mu - oc * jnp.log(mu)) + jnp.sum(tnll)           # Poisson charge + order-stat time


gradf = jax.jit(jax.grad(lossf))
KEYS = [jax.random.PRNGKey(s) for s in range(2)]
def G(th, oc, ot): return np.mean([np.asarray(gradf(jnp.asarray(th), oc, ot, k)) for k in KEYS], 0)


def fisher(th, oc, ot):                                 # PSD Gauss-Newton metric, FD Jacobians
    mu0 = np.maximum(np.mean([np.asarray(perpmt(jnp.asarray(th), oc, ot, k)[0]) for k in KEYS], 0), 1e-8)
    Jm = np.zeros((ND, 9)); Jl = np.zeros((ND, 9))
    for j in range(9):
        h = FDH[j]; ej = np.eye(9)[j]
        mp, lp = [np.mean([np.asarray(x) for x in xs], 0) for xs in zip(*[perpmt(jnp.asarray(th + h * ej), oc, ot, k) for k in KEYS])]
        mm, lm = [np.mean([np.asarray(x) for x in xs], 0) for xs in zip(*[perpmt(jnp.asarray(th - h * ej), oc, ot, k) for k in KEYS])]
        Jm[:, j] = (mp - mm) / (2 * h); Jl[:, j] = (lp - lm) / (2 * h)
    Jmn = Jm / np.sqrt(mu0)[:, None]
    return Jmn.T @ Jmn + Jl.T @ Jl


def fit(oc, ot, start, niters=130, lr=8., ridge=0.3, lam=0.01, refresh=8):
    th = np.array(start, float); best = (1e18, th.copy()); F = None; g = G(th, oc, ot)
    for it in range(niters):
        if F is None or it % refresh == 0: F = fisher(th, oc, ot)
        S = SCALE9; Fs = S[:, None] * F * S[None, :]; gs = S * g          # SCALE9-precond (mandatory)
        rI = ridge * np.median(np.clip(np.diag(Fs), 1e-12, None)) * np.eye(9)  # Levenberg floor (t0↔lon)
        du = -lr * np.linalg.solve(Fs + lam * np.diag(Fs) + rI + 1e-9 * np.eye(9), gs)
        th = th + S * du; g = G(th, oc, ot); gn = float(np.linalg.norm(g * S))
        if gn < best[0]: best = (gn, th.copy())                          # min‖g‖ readout
    return best[1]


# --- truth muon (1050 MeV, fiducial vertex+direction) -> sampled data -> offset start -> recover ---
td = np.array([0.3, 0.1, 0.95]); td /= np.linalg.norm(td)
pol, az = np.arccos(td[2]), np.arctan2(td[1], td[0])
th9 = np.array([1050., 1.5, -0.8, 2.0, np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
c, t = jax.lax.stop_gradient(data_sim(sc2(jnp.asarray(th9)), jax.random.PRNGKey(0)))
oc = jnp.asarray(np.asarray(c)); ot = jnp.asarray(np.where(np.asarray(c) > 0, np.asarray(t), 0.))

start = th9 + 2.5 * SCALE9 * np.random.default_rng(1).uniform(-1, 1, 9)   # from-scratch start (~0.7 m off)
res = fit(oc, ot, start)
sv = np.linalg.norm((start - th9)[1:4]) * 100; sd = np.degrees(np.arccos(np.clip(th9dir(start) @ td, -1, 1)))
fv = np.linalg.norm((res - th9)[1:4]) * 100; fd = np.degrees(np.arccos(np.clip(th9dir(res) @ td, -1, 1)))
print(f'start  vtx {sv:5.1f} cm   E {start[0]-1050:+6.0f} MeV   dir {sd:4.1f}°   t0 {start[8]-0:+5.2f} ns')
print(f'fit    vtx {fv:5.1f} cm   E {res[0]-1050:+6.1f} MeV   dir {fd:4.2f}°   t0 {res[8]-0:+5.2f} ns   (truth 1050 MeV)')
