"""9-parameter track reconstruction — the consistent Fisher-Gauss-Newton recipe.

Ported from the recon study (``RECO_PIPELINE.md`` / ``gn_fisher_recon.py``): fit the track
``θ = [E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`` against the per-PMT (charge, first-arrival
time) observables by Gauss-Newton on a PSD Fisher metric

    F = Jμᵀ diag(1/μ) Jμ   (charge Poisson Fisher)  +  Jlᵀ Jl   (time NLL score-covariance)

built by FINITE DIFFERENCES (the DiCE ``custom_vjp`` blocks ``jacfwd``, and the autodiff
track-Hessian is indefinite) and solved in **SCALE9-preconditioned** coordinates — mandatory:
the raw F is position-dominated (F_xx ~ 1e5 vs F_EE ~ 1) so energy freezes without it. The
gradient ``g`` is reverse-mode autodiff (it DOES flow through the custom_vjp). Charge = Poisson
NLL (un-normalised — carries E + longitudinal/transverse vertex); time = the windowed
first-arrival ORDER-STATISTIC NLL (:func:`lucid.losses.first_arrival_window_nll` — carries
direction, t0, transverse vertex), AMP_DETACH baked in. Readout = the min‖g‖ iterate.

``pred`` is a per-photon track simulator: ``setup_event_simulator(..., hit_mode='per_photon',
pos_grad_threshold=K, n_grad_iters=K)`` returning ``(log_w, flat_times, flat_indices,
total_charge)``. Finalized knobs (RECO_PIPELINE §4): LR=8, RIDGE_I=0.3, REFRESH=8, NKEYS=4,
SIGMA=2.5(=TTS), DELTA=1.0, min‖g‖.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import ParticleParams
from lucid.losses import counts_loss, first_arrival_window_nll

# Natural per-parameter scales (RECO_PIPELINE §2): ~50 MeV, 0.2 m, 0.02 cos-units, 0.2 ns.
SCALE9 = np.array([50., .2, .2, .2, .02, .02, .02, .02, .2])
PARAM_NAMES = ['E', 'x', 'y', 'z', 'sin_t', 'cos_t', 'sin_p', 'cos_p', 't0']


def track_from_vec9(t9):
    """9-vector ``[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]`` -> :class:`ParticleParams` (the ``sc2``
    bridge). Direction carried as sin/cos pairs (no atan2 wrap / kinky angle gradient)."""
    st, ct, sp, cp = t9[4], t9[5], t9[6], t9[7]
    nt = jnp.hypot(st, ct) + 1e-12
    npp = jnp.hypot(sp, cp) + 1e-12
    return ParticleParams(energy=t9[0], position=t9[1:4],
                          theta=jnp.arctan2(st / nt, ct / nt),
                          phi=jnp.arctan2(sp / npp, cp / npp), t0=t9[8])


def vec9_dir(t9):
    """Unit direction (numpy) from a 9-vector — for reporting angular error."""
    st, ct, sp, cp = float(t9[4]), float(t9[5]), float(t9[6]), float(t9[7])
    nt = np.hypot(st, ct); npp = np.hypot(sp, cp)
    st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp
    return np.array([st * cp, st * sp, ct])


def vec9_from_track(energy, position, direction, t0=0.0):
    """Build the 9-vector from physical ``(energy, position, direction, t0)``."""
    d = np.asarray(direction, float); d = d / (np.linalg.norm(d) + 1e-12)
    pol = np.arccos(np.clip(d[2], -1, 1)); az = np.arctan2(d[1], d[0])
    p = np.asarray(position, float)
    return np.array([float(energy), p[0], p[1], p[2],
                     np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), float(t0)])


def seed_vertex_time(pos, obs_counts, obs_times, *, vspeed=0.2167, vgrid=11, tankr=None,
                     tankz=None, n_refine=20, gate=18.0, bright_frac=0.6, q_edge=0.10):
    """Robust multilateration vertex + t0 seed from first-arrival times — pure geometry, NO
    forward sim.

    The charge ring fixes direction but is degenerate in *where along the track* the vertex sits;
    first-arrival *times* break that by triangulation (GPS run backwards). A DIRECT photon at PMT
    ``P`` arriving at ``T`` left the vertex ``v`` at ``t0`` and travelled straight at the water
    group velocity ``vspeed`` (m/ns): ``T ≈ t0 + |P-v|/vspeed``. The vertex is where the spheres
    of constant time-of-flight intersect.

    ROBUSTNESS is essential on real data: a large fraction of hits are SCATTERED / REFLECTED
    photons that arrive tens-to-hundreds of ns LATE (one-sided), and dim PMTs are scattered-
    dominated (a bright PMT is ~90% direct light). A plain least-squares fit lets those late
    outliers drag the vertex metres off (porting the recon's un-robust ``time_vertex`` put the
    seed ~15 m out). So: (1) keep the brightest ``bright_frac`` of hits; (2) the coarse ``vgrid³``
    tank scan scores each candidate by WEIGHTED INLIER COUNT — ``t0`` from the early edge
    (quantile ``q_edge`` of the back-projected emission time), a hit is an inlier if its residual
    is within ``±gate`` ns (scattered light, being late, falls outside); (3) a 20-step 3D
    Gauss-Newton polish runs on INLIERS ONLY (re-selected each step), ``t0`` the closed-form
    weighted-mean inlier residual. ``gate`` ~18 ns admits the direct core (track extent + TTS)
    while rejecting scatter. ``tankr``/``tankz`` default to the PMT-array extent.

    Returns ``(vtx (3,), t0 float)`` — feed straight into :func:`vec9_from_track`.
    """
    pos = np.asarray(pos, float)
    if tankr is None: tankr = float(np.hypot(pos[:, 0], pos[:, 1]).max())
    if tankz is None: tankz = float(np.abs(pos[:, 2]).max())
    oc = np.asarray(obs_counts, float); ot = np.asarray(obs_times, float)
    hit = oc > 0
    P = pos[hit]; T = ot[hit]; q = np.maximum(oc[hit], 0.)
    if 0. < bright_frac < 1. and hit.sum() > 50:                # drop the dim (scatter-dominated) tail
        bm = q >= np.quantile(q, 1. - bright_frac); P, T, q = P[bm], T[bm], q[bm]
    w = np.sqrt(q)

    gx, gy, gz = np.meshgrid(np.linspace(-tankr, tankr, vgrid), np.linspace(-tankr, tankr, vgrid),
                             np.linspace(-tankz, tankz, vgrid), indexing='ij')
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    grid = grid[np.hypot(grid[:, 0], grid[:, 1]) <= tankr]
    best = (-1.0, grid[0])
    for v in grid:                                             # coarse scan: maximise weighted inliers
        d = T - np.linalg.norm(P - v, axis=1) / vspeed         # back-projected emission time
        res = d - np.quantile(d, q_edge)                       # reference to the early edge
        s = w[(res > -gate) & (res < gate)].sum()
        if s > best[0]: best = (s, v)
    v = best[1].astype(float); t0 = float(np.quantile(T - np.linalg.norm(P - v, axis=1) / vspeed, q_edge))
    for _ in range(n_refine):                                  # GN polish on INLIERS only
        r = np.linalg.norm(P - v, axis=1) + 1e-6; pt = r / vspeed; d = T - pt
        res = d - np.quantile(d, q_edge); inl = (res > -gate) & (res < gate)
        Pi, ri, pti, wi, Ti = P[inl], r[inl], pt[inl], w[inl], T[inl]
        sw = wi.sum() + 1e-12; t0 = float(np.sum(wi * (Ti - pti)) / sw); resid = Ti - t0 - pti
        g = (Pi - v) / ri[:, None]; J = g / vspeed
        Hh = (J * wi[:, None]).T @ J + 1e-3 * np.eye(3); gg = (J * wi[:, None]).T @ resid
        v = v - np.clip(np.linalg.solve(Hh, gg), -2., 2.)      # v <- v - (JᵀWJ)⁻¹JᵀW·resid (inliers)
        rxy = np.hypot(v[0], v[1]); s = min(1.0, tankr / max(rxy, 1e-6))
        v[0] *= s; v[1] *= s; v[2] = np.clip(v[2], -tankz, tankz)
    return v, float(t0)


class ReconModel:
    """Wraps a per-photon track predictor into the per-PMT ``(μ charge, time-NLL)`` the recon
    Fisher-GN consumes, plus the assembled loss / gradient / FD Fisher metric.

    ``pred(track, key) -> (log_w, flat_times, flat_indices, total_charge)`` is a track
    simulator from ``setup_event_simulator(..., hit_mode='per_photon')``. ``tot_n_scale`` is
    the single charge calibration constant (RECO_PIPELINE §3.4; 0.982 for the SIREN muon
    emitter, 1.0 for a self-consistent forward).
    """

    def __init__(self, pred, num_detectors, sigma=2.5, delta=1.0, tot_n_scale=1.0):
        self.pred = pred
        self.ND = int(num_detectors)
        self.sigma = float(sigma); self.delta = float(delta); self.tot_n_scale = float(tot_n_scale)

        def _perpmt(t9, oc, ot, key):
            lw, ft, fi, tot = pred(track_from_vec9(t9), key)
            mu = jnp.maximum(tot * self.tot_n_scale, 1e-8)       # SCALED charge (carries energy)
            mu_surv = jnp.maximum(tot, 1e-8)                     # UNSCALED survival denom — must NOT
            tobs = ot - t9[8]                                    # be scaled (else far-capture dies,
            tnll = first_arrival_window_nll(lw, ft, fi, tobs, mu_surv, oc, self.ND,  # RECO_PIPELINE §3.4)
                                            sigma=self.sigma, delta=self.delta)
            return mu, tnll

        def _loss(t9, oc, ot, key):
            mu, tnll = _perpmt(t9, oc, ot, key)
            # eps=0: μ is already floored at 1e-8 so log(μ) is safe, and this matches the
            # validated recon RAW Poisson exactly (a nonzero eps shifts the dim-PMT gradient).
            return counts_loss(oc, mu, eps=0.0, normalize=False) + jnp.sum(tnll)

        self._perpmt = jax.jit(_perpmt)
        self._loss = jax.jit(_loss)
        self._grad = jax.jit(jax.grad(_loss))

    def perpmt(self, t9, oc, ot, key):
        return self._perpmt(jnp.asarray(t9), oc, ot, key)

    def loss(self, t9, oc, ot, key):
        return self._loss(jnp.asarray(t9), oc, ot, key)

    def grad(self, t9, oc, ot, key):
        return self._grad(jnp.asarray(t9), oc, ot, key)

    def fisher(self, t9, oc, ot, keys, fdh):
        """PSD Fisher metric ``Jμᵀ diag(1/μ) Jμ + Jlᵀ Jl`` via central FD, averaged over ``keys``."""
        t9 = np.asarray(t9, float); ND = self.ND
        mu0 = np.maximum(np.mean([np.asarray(self.perpmt(t9, oc, ot, k)[0]) for k in keys], 0), 1e-8)
        Jm = np.zeros((ND, 9)); Jl = np.zeros((ND, 9))
        for j in range(9):
            h = fdh[j]; ej = np.eye(9)[j]
            mp, lp = [np.mean([np.asarray(x) for x in xs], 0)
                      for xs in zip(*[self.perpmt(t9 + h * ej, oc, ot, k) for k in keys])]
            mm, lm = [np.mean([np.asarray(x) for x in xs], 0)
                      for xs in zip(*[self.perpmt(t9 - h * ej, oc, ot, k) for k in keys])]
            Jm[:, j] = (mp - mm) / (2 * h); Jl[:, j] = (lp - lm) / (2 * h)
        Jmn = Jm / np.sqrt(mu0)[:, None]
        return Jmn.T @ Jmn + Jl.T @ Jl


def fit_track(model, obs_counts, obs_times, start, *, nkeys=4, niters=250, lr=8.0,
              ridge_i=0.3, lam=0.01, refresh=8, seed=0, readout='ming', hist=False):
    """Consistent Fisher-Gauss-Newton track fit, SCALE9-preconditioned (RECO_PIPELINE §4).

    Parameters mirror the finalized recipe. The step is solved in SCALE9-scaled coordinates
    (``Fs = S⊗F⊗S``, ``gs = S·g``) with a Marquardt term ``lam·diag(Fs)`` and an ADDITIVE
    Levenberg floor ``ridge_i·median(diag Fs)·I`` (bounds the flat t0↔longitudinal eigenvalue —
    ``ridge_i=0`` is forbidden, runs t0 away). ``lr`` amplifies the consistent GN under-shoot.

    Returns the 9-vector at the min‖g‖ iterate (``readout='ming'``, robust to the LR overshoot)
    or the final iterate (``'final'``). If ``hist=True``, returns ``(theta, history)`` where
    ``history`` is a dict with the full per-iteration ``traj`` ((niters+1, 9) θ trajectory) and
    ``gnorm`` ((niters+1,) scaled ‖g‖) — for campaign trajectory analysis.
    """
    oc = jnp.asarray(obs_counts); ot = jnp.asarray(obs_times)
    keys = [jax.random.PRNGKey(seed + s) for s in range(nkeys)]
    fdh = 0.4 * SCALE9
    S = SCALE9

    def G(th):
        return np.mean([np.asarray(model.grad(th, oc, ot, k)) for k in keys], 0)

    th = np.asarray(start, float); best = (1e18, th.copy()); F = None
    g = G(th); traj = [th.copy()]; gnorms = [float(np.linalg.norm(g * S))]
    for it in range(niters):
        if F is None or it % refresh == 0:
            F = model.fisher(th, oc, ot, keys, fdh)
        Fs = S[:, None] * F * S[None, :]; gs = S * g                       # SCALE9 preconditioning
        marq = np.diag(lam * np.diag(Fs))                                  # Marquardt: a true diagonal
        rI = ridge_i * np.median(np.clip(np.diag(Fs), 1e-12, None)) * np.eye(9)  # additive Levenberg floor
        du = -lr * np.linalg.solve(Fs + marq + rI + 1e-9 * np.eye(9), gs)
        th = th + S * du; g = G(th); gn = float(np.linalg.norm(g * S))
        if gn < best[0]: best = (gn, th.copy())
        traj.append(th.copy()); gnorms.append(gn)
    out = best[1] if readout == 'ming' else th
    if hist:
        return out, dict(traj=np.array(traj), gnorm=np.array(gnorms), best_iter=int(np.argmin(gnorms)))
    return out


def fit_track_multistart(model, obs_counts, obs_times, starts, *, nkeys=4, seed=0,
                         prefer=0, margin=0.01, **kw):
    """Run :func:`fit_track` from several seeds and keep the best by a MARGIN-GATED data loss.

    The time-multilateration seed (:func:`seed_vertex_time`) nails the transverse vertex but is
    biased forward along the track (it finds the Cherenkov time-centroid, ~½ a track-length ahead
    of the vertex); the charge-grid seed is the reverse — fine longitudinally on most events,
    but it loses inward-pointing tracks where the ring is vertex-degenerate. The two are
    COMPLEMENTARY, so fitting from both and arbitrating by the converged data loss rescues the
    inward-track tail (wanderers >1 m → 0, RMS halved on a 100-event GEANT4 study).

    But plain argmin OVER-SELECTS the time seed: the SIREN-emission bias means the loss minimum
    isn't exactly at truth, so on easy events the forward-biased basin can score a marginally
    lower loss yet a worse vertex (28/100 events regressed >5 cm at margin=0). The fix is a
    relative ``margin``: keep the ``prefer``-th seed (the charge grid — longitudinally unbiased
    on the bulk) unless another seed beats it by ``margin`` × |loss|. A 1% margin picks the time
    seed only on the ~4/100 decisive rescues, leaving the median tied with the charge-only
    baseline while still killing the tail (validated retrospectively over the loss/error grid).

    ``starts`` is a list of 9-vectors (put the safe default first → ``prefer=0``). Returns
    ``(best_theta, info)`` with ``info`` = ``which`` (winning seed index), ``losses`` (per-seed
    converged loss), and ``per_seed`` (each ``(theta, history)`` from :func:`fit_track`).
    """
    oc = jnp.asarray(obs_counts); ot = jnp.asarray(obs_times)
    keys = [jax.random.PRNGKey(seed + s) for s in range(nkeys)]

    def dloss(th):                                              # data loss at th, averaged over keys
        return float(np.mean([float(model.loss(th, oc, ot, k)) for k in keys]))

    per_seed = [fit_track(model, oc, ot, s, nkeys=nkeys, seed=seed, hist=True, **kw) for s in starts]
    losses = [dloss(th) for th, _ in per_seed]
    # margin gate: switch off the preferred seed only when another beats it DECISIVELY
    base = losses[prefer]; thr = base - margin * abs(base); which = prefer
    cand = [i for i in range(len(losses)) if i != prefer and losses[i] < thr]
    if cand:
        which = min(cand, key=lambda i: losses[i])
    return per_seed[which][0], dict(which=which, losses=losses, per_seed=per_seed)
