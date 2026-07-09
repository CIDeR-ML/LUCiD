"""9-parameter track reconstruction — the consistent Fisher-Gauss-Newton recipe.

Ported from the recon study (``RECO_PIPELINE.md`` / ``gn_fisher_recon.py``): fit the track
``θ = [E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`` against the per-PMT (charge, first-arrival
time) observables by Gauss-Newton on a PSD Fisher metric

    F = Jμᵀ diag(1/μ) Jμ   (charge Poisson Fisher)  +  Jlᵀ Jl   (time NLL score-covariance)

built by FORWARD-MODE AD (``jacfwd`` of ``perpmt`` — the DiCE ``custom_vjp`` that used to block it
is gone; this is the PSD GN metric, NOT the indefinite raw autodiff Hessian; ``'fd'`` central-FD
kept as legacy) and solved in **SCALE9-preconditioned** coordinates — mandatory:
the raw F is position-dominated (F_xx ~ 1e5 vs F_EE ~ 1) so energy freezes without it. The
gradient ``g`` is reverse-mode autodiff (it DOES flow through the custom_vjp). Charge = Poisson
NLL (un-normalised — carries E + longitudinal/transverse vertex); time = the windowed
first-arrival ORDER-STATISTIC NLL (:func:`lucid.losses.first_arrival_window_nll` — carries
direction, t0, transverse vertex), AMP_DETACH baked in. Readout = the min‖g‖ iterate.

``pred`` is a per-photon track simulator: ``setup_event_simulator(..., hit_mode='per_photon',
pos_grad_threshold=K, n_grad_iters=K)`` returning ``(log_w, flat_times, flat_indices,
total_charge)``. Tuned knobs (AD HP sweep, SK muon, NPH-robust 50k↔150k): ``fisher_mode='ad'``,
``lr=4`` annealed to ``lr_final=1.5``, ``ridge_i=0.1``, ``refresh=8``, ``nkeys=8``, ``niters=150``.
The annealed lr (big early steps → settle low) converges ~2× faster than the old constant ``lr=2``
(conv P90 ≤81 vs ~86-150 iters), VALIDATED resolution-neutral + 0-divergence across muon+electron ×
{500,1000,1500} MeV @150k (150 events, same-event paired vs the old lr2/250 recipe — Δvtx within
±1.7cm/bootstrap, mean ~0). ``lr=5`` was faster still but diverged ~1.3% of events, so ``lr=4`` is the
robust pick; keep ``ridge_i=0.1`` (raising it with high lr destabilises). ``niters=150`` covers the
slowest cell's P90 + the last-40 Polyak window; ``readout='polyak'`` (median ties min‖g‖ BUT a single iterate has a ~2× worse tail — ‖g‖ dips on
noise fluctuations where the vtx is far; the last-40 average suppresses that). The ~15cm vertex floor
is SIREN-emitter-bias-limited (readout-probe-proven: min-data-loss readout = Polyak = the loss
minimum; the ~4cm "oracle" is an unreachable fluctuation toward truth), NOT optimizer-limited.
SIGMA=2.5(=TTS), DELTA=1.0.
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

    def __init__(self, pred, num_detectors, sigma=2.5, delta=1.0, tot_n_scale=1.0,
                 time_weight=1.0, energy_from_scale=True, nphot_fn=None,
                 energy_scale_mode='simtotal'):
        self.pred = pred
        self.ND = int(num_detectors)
        self.sigma = float(sigma); self.delta = float(delta); self.tot_n_scale = float(tot_n_scale)
        self.time_weight = float(time_weight)                # explicit charge↔time weight on the time NLL.
        #   1.0 = the from-first-principles joint NLL  P(n)·P(t|n) = charge + conditional first-arrival;
        #   validated optimal (a balance sweep found tw=1 best; tw>1 trades a hair of P68 for failures).
        # energy_from_scale (DEFAULT True): route the charge term's ENERGY gradient through the TOTAL
        #   charge only (the Poisson factorises exactly into Poisson(Q;A)·Multinomial(shape) — energy
        #   lives in the scale A, geometry in the shape). Removes the profile bias a small emission-
        #   shape misspecification otherwise puts on energy via the soft (E↔geometry) degeneracy:
        #   validated 200 fits @1 GeV muon, dE −29.7±2.1 → +2.6±1.5, σ 30→21, geometry unchanged.
        #   Write μ = scale(E)·μ(sgE,g): forward value unchanged, ∂/∂E flows only through the global
        #   scale → ∂L/∂E ∝ (Σμ−Q); ∂/∂geometry still carries the per-PMT shape. Charge analog of the
        #   time term's AMP_DETACH. Set False for the raw per-PMT-Poisson energy estimate.
        #   energy_scale_mode: 'simtotal' (default, self-contained, uses the sim's own total; costs a
        #   2nd forward pass per _perpmt -> ~2x the Fisher/Jacobian cost) or 'nphot' (single pass,
        #   needs nphot_fn = the emitter's ctx.n_photons_fn). NOTE: under energy_from_scale the time
        #   term is built from the energy-DETACHED shape pass too, so (like AMP_DETACH for its
        #   amplitude) it carries no energy gradient — energy comes ONLY from the total-charge scale.
        self.energy_from_scale = bool(energy_from_scale)
        self.nphot_fn = nphot_fn
        self.energy_scale_mode = str(energy_scale_mode)       # 'nphot' | 'simtotal'
        #   'nphot'   : energy scale = analytic nphot(E) ratio (noise-free scalar; needs nphot_fn).
        #   'simtotal': energy scale = the SIM'S OWN total A=Σμ(E,g) — a 2nd forward pass at LIVE E
        #               supplies A's exact ∂/∂E (real acceptance E-dependence), no nphot curve. The
        #               shape pass (detached E) supplies pᵢ=μᵢ/A → μ=A·pᵢ routes ∂/∂E through A only.
        if self.energy_from_scale and self.energy_scale_mode == 'nphot' and nphot_fn is None:
            raise ValueError("energy_from_scale + mode 'nphot' requires nphot_fn (the emitter n_photons_fn)")

        def _perpmt(t9, oc, ot, key):
            if self.energy_from_scale:
                t9s = t9.at[0].set(jax.lax.stop_gradient(t9[0]))   # detach E in the shape
                lw, ft, fi, tot = pred(track_from_vec9(t9s), key)  # μ(sgE, g): geometry-live shape
                if self.energy_scale_mode == 'simtotal':
                    # energy scale from the sim's own total at LIVE E (values identical to the shape
                    # pass since sgE==E in value; only the E-gradient differs). μ = A_live · pᵢ.
                    _, _, _, tot_live = pred(track_from_vec9(t9), key)
                    A_live = jnp.sum(tot_live)                     # Σμ(E,g): carries ∂A/∂E and ∂A/∂g
                    A_shape = jnp.maximum(jnp.sum(tot), 1e-8)      # Σμ(sgE,g): ∂/∂g only
                    mu = jnp.maximum(tot * (A_live / A_shape) * self.tot_n_scale, 1e-8)
                else:                                              # 'nphot': analytic scale ratio
                    # nphot(E)=A·E^b+c crosses zero near ~137 MeV -> escale sign-flip below it; floor it.
                    _npn = jnp.maximum(self.nphot_fn(t9[0]), 1e3)
                    _npd = jnp.maximum(self.nphot_fn(jax.lax.stop_gradient(t9[0])), 1e3)
                    escale = _npn / _npd                          # =1 in value, carries ∂/∂E
                    mu = jnp.maximum(tot * escale * self.tot_n_scale, 1e-8)
            else:
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
            return counts_loss(oc, mu, eps=0.0, normalize=False) + self.time_weight * jnp.sum(tnll)

        self._perpmt = jax.jit(_perpmt)
        self._loss = jax.jit(_loss)
        self._grad = jax.jit(jax.grad(_loss))
        # Forward-mode Jacobians (∂μ/∂t9, ∂tnll/∂t9), both (ND, 9), in ONE jacfwd pass — replaces
        # the 9×2 FD perpmt evals. Unblocked now that the step is custom_vjp-free; key is traced.
        self._pjac = jax.jit(jax.jacfwd(_perpmt, argnums=0))

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

    def fisher_ad(self, t9, oc, ot, keys, fdh=None):
        """Same PSD Fisher metric via FORWARD-MODE AD (``jacfwd`` of ``perpmt``), averaged over
        ``keys`` — one jacfwd pass replaces the 9×2 FD perpmt evals (``fdh`` unused, kept for a
        drop-in signature). Lower variance + faster than the central-FD ``fisher``."""
        t9 = jnp.asarray(t9, float); ND = self.ND
        Jm = np.zeros((ND, 9)); Jl = np.zeros((ND, 9)); mu0 = np.zeros(ND)
        for k in keys:
            (jm, jl) = self._pjac(t9, oc, ot, k)     # each (ND, 9)
            Jm += np.asarray(jm); Jl += np.asarray(jl)
            mu0 += np.asarray(self._perpmt(t9, oc, ot, k)[0])
        n = len(keys); Jm /= n; Jl /= n; mu0 = np.maximum(mu0 / n, 1e-8)
        Jmn = Jm / np.sqrt(mu0)[:, None]
        return Jmn.T @ Jmn + Jl.T @ Jl


def fit_track(model, obs_counts, obs_times, start, *, nkeys=8, niters=150, lr=4.0,
              lr_final=1.5, ridge_i=0.1, lam=0.01, refresh=8, refresh_final=None, refresh_switch=0.5,
              seed=0, readout='polyak', polyak_w=40, hist=False, fisher_mode='ad',
              verbose=False, truth=None, trust='auto'):
    """Consistent Fisher-Gauss-Newton track fit, SCALE9-preconditioned (RECO_PIPELINE §4).

    Parameters mirror the finalized recipe. The step is solved in SCALE9-scaled coordinates
    (``Fs = S⊗F⊗S``, ``gs = S·g``) with a Marquardt term ``lam·diag(Fs)`` and an ADDITIVE
    Levenberg floor ``ridge_i·median(diag Fs)·I`` (bounds the flat t0↔longitudinal eigenvalue —
    ``ridge_i=0`` is forbidden, runs t0 away). ``lr`` amplifies the consistent GN under-shoot.

    Readout (which iterate to return): ``'polyak'`` (DEFAULT) averages the last ``polyak_w`` iterates —
    robust to the noise-floor wandering (the AD-tuned win); ``'ming'`` the min‖g‖ iterate; ``'final'``
    the last. If ``hist=True``, returns ``(theta, history)`` where
    ``history`` is a dict with the full per-iteration ``traj`` ((niters+1, 9) θ trajectory) and
    ``gnorm`` ((niters+1,) scaled ‖g‖) — for campaign trajectory analysis.

    ``fisher_mode`` : ``'ad'`` (DEFAULT) builds the PSD Gauss-Newton/Fisher metric
    ``Jμᵀdiag(1/μ)Jμ + JlᵀJl`` by forward-mode autodiff (``ReconModel.fisher_ad``, jacfwd of
    ``perpmt`` — unblocked since the step is custom_vjp-free). It is the TRUER metric (the FD
    diagonal is inflated by per-sensor estimation variance ∝ 1/nkeys) and ~2.8× faster, and is
    PSD by construction (NOT the indefinite raw autodiff Hessian). ``'fd'`` keeps the legacy central
    finite-difference metric (``ReconModel.fisher``). ⚠️ The AD metric is ~1–137× SMALLER per param
    than FD, so the FD-tuned ``lr=8`` OVERSHOOTS with ``'ad'`` — retune the step/damping
    (``lr``/``lr_final``/``ridge_i``) for AD (rough validated point: ``lr≈1``). With AD the metric is
    cheap+low-variance, so ``refresh=1`` (recompute every step) is affordable.

    ``verbose=True`` shows a live ‖g‖ progress bar and prints a result table (pass ``truth`` — a
    9-vector — to include per-parameter errors).
    """
    from . import report
    # trust='auto': apply the step clip ONLY when the model routes energy through the total-charge
    # scale (which steepens the energy gradient and needs it); the plain per-PMT recipe gets trust=None
    # → byte-identical to the pre-trust behavior. Pass an explicit float/None to override.
    if trust == 'auto':
        trust = 3.0 if getattr(model, 'energy_from_scale', False) else None
    oc = jnp.asarray(obs_counts); ot = jnp.asarray(obs_times)
    keys = [jax.random.PRNGKey(seed + s) for s in range(nkeys)]
    fdh = 0.4 * SCALE9
    S = SCALE9

    def G(th):
        return np.mean([np.asarray(model.grad(th, oc, ot, k)) for k in keys], 0)

    fisher_fn = model.fisher_ad if fisher_mode == 'ad' else model.fisher
    th = np.asarray(start, float); best = (1e18, th.copy()); F = None
    g = G(th); traj = [th.copy()]; gnorms = [float(np.linalg.norm(g * S))]
    # Refresh-cadence schedule: recompute the Fisher every `refresh` iters early, then every
    # `refresh_final` (smaller = fresher) after `refresh_switch·niters`. The ref2/ref1 resolution gain
    # comes from a fresh metric in the LATE precision phase; spending it only there recovers most of the
    # gain at far less cost than constant low refresh. refresh_final=None → constant `refresh` (unchanged).
    sw = int(refresh_switch * niters); since = 0
    pbar = report.progress(range(niters), desc='track fit', total=niters, verbose=verbose)
    for it in pbar:
        r_it = refresh if (refresh_final is None or it < sw) else refresh_final
        if F is None or since >= r_it or (refresh_final is not None and it == sw):
            F = fisher_fn(th, oc, ot, keys, fdh); since = 0
        since += 1
        Fs = S[:, None] * F * S[None, :]; gs = S * g                       # SCALE9 preconditioning
        marq = np.diag(lam * np.diag(Fs))                                  # Marquardt: a true diagonal
        rI = ridge_i * np.median(np.clip(np.diag(Fs), 1e-12, None)) * np.eye(9)  # additive Levenberg floor
        # optional LR anneal lr->lr_final (linear): small late steps can't kick the fit OUT of a
        # converged basin (the late-divergence failure mode), so the trajectory settles at the min.
        lr_it = lr if lr_final is None else lr + (lr_final - lr) * (it / max(1, niters - 1))
        du = -lr_it * np.linalg.solve(Fs + marq + rI + 1e-9 * np.eye(9), gs)
        if trust is not None:                       # trust-region step clip (see trust='auto' above):
            du = np.clip(du, -trust, trust)         # |Δθ_k| ≤ trust·SCALE9_k. Active by default only
            #   for energy_from_scale models (else disabled → the plain recipe is byte-identical);
            #   required once energy_from_scale steepens the energy gradient (else ~1/10 events run away).
        th_new = th + S * du; g_new = G(th_new)
        # NaN/Inf-reject trust guard: the big early steps of the annealed lr=4 occasionally overshoot
        # into a degenerate region (~0.3% of events, seen at 250k) where the next gradient blows up.
        # Reject any step that produces a non-finite θ/gradient (keep the previous iterate) so a single
        # bad step can't poison the Polyak readout into NaN. Clean fits never trip this → resolution
        # unchanged; would-be-divergent fits get a bounded result instead.
        if np.isfinite(th_new).all() and np.isfinite(g_new).all():
            th, g = th_new, g_new
        gn = float(np.linalg.norm(g * S))
        if gn < best[0]: best = (gn, th.copy())
        traj.append(th.copy()); gnorms.append(gn)
        pbar.set_postfix_str(f'‖g‖={gn:.2e}')
    if readout == 'polyak':        # avg the last polyak_w iterates — robust to the floor wandering
        out = np.mean(np.array(traj)[-polyak_w:], axis=0)   # (the ‖g‖ never vanishes at the biased
    elif readout == 'ming':        # minimum, so a single iterate wanders; averaging settles it)
        out = best[1]
    else:
        out = th
    if verbose:
        report.emit(report.track_table(out, truth=truth, dir_of=vec9_dir))
    if hist:
        return out, dict(traj=np.array(traj), gnorm=np.array(gnorms), best_iter=int(np.argmin(gnorms)))
    return out


def fit_track_multistart(model, obs_counts, obs_times, starts, *, nkeys=4, seed=0,
                         prefer=0, margin=0.01, verbose=False, truth=None, **kw):
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

    per_seed = [fit_track(model, oc, ot, s, nkeys=nkeys, seed=seed, hist=True, verbose=verbose, **kw)
                for s in starts]
    losses = [dloss(th) for th, _ in per_seed]
    # margin gate: switch off the preferred seed only when another beats it DECISIVELY
    base = losses[prefer]; thr = base - margin * abs(base); which = prefer
    cand = [i for i in range(len(losses)) if i != prefer and losses[i] < thr]
    if cand:
        which = min(cand, key=lambda i: losses[i])
    if verbose:
        from . import report
        rows = [[f'seed {i}' + ('  ← kept' if i == which else ''), f'{losses[i]:.4e}']
                for i in range(len(losses))]
        report.emit(report.rule('multistart — margin-gated seed selection'))
        report.emit(report.table(['start', 'converged data loss'], rows))
        report.emit(report.track_table(per_seed[which][0], truth=truth, dir_of=vec9_dir))
    return per_seed[which][0], dict(which=which, losses=losses, per_seed=per_seed)
