"""9-parameter track reconstruction — the consistent Fisher-Gauss-Newton recipe.

Ported from the recon study's ``gn_fisher_recon.py``: fit the track
``θ = [E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`` against the per-PMT (charge, first-arrival
time) observables by Gauss-Newton on a PSD Fisher metric

    F = Jμᵀ diag(1/μ) Jμ   (charge Poisson Fisher)  +  Jlᵀ Jl   (time NLL score-covariance)

built by FORWARD-MODE AD (``jacfwd`` of ``perpmt`` — the DiCE ``custom_vjp`` that used to block it
is gone; this is the PSD GN metric, NOT the indefinite raw autodiff Hessian) and solved in
**SCALE9-preconditioned** coordinates — mandatory:
the raw F is position-dominated (F_xx ~ 1e5 vs F_EE ~ 1) so energy freezes without it. The
gradient ``g`` is reverse-mode autodiff (it DOES flow through the custom_vjp). Charge = Poisson
NLL (un-normalised — carries E + longitudinal/transverse vertex); time = the windowed
first-arrival ORDER-STATISTIC NLL (:func:`lucid.losses.first_arrival_window_nll` — carries
direction, t0, transverse vertex), AMP_DETACH baked in. Readout = the Polyak tail average.

``pred`` is a per-photon track simulator: ``setup_event_simulator(..., source='particle')``
(``hits='photon_list'``) returning ``(log_w, flat_times, flat_indices, total_charge)``. The
published working point is :data:`FIT_RECIPE` (the fit) and :data:`MODEL_RECIPE` (the model).
Tuned knobs (AD HP sweep, SK muon, NPH-robust 50k↔150k): the forward-mode AD Fisher metric,
``lr=4`` annealed to ``lr_final=1.5``, ``mu=0.1``, ``refresh=8``, ``nkeys=8``, ``steps=150``.
The annealed lr (big early steps → settle low) converges ~2× faster than the old constant ``lr=2``
(conv P90 ≤81 vs ~86-150 iters), VALIDATED resolution-neutral + 0-divergence across muon+electron ×
{500,1000,1500} MeV @150k (150 events, same-event paired vs the old lr2/250 recipe — Δvtx within
±1.7cm/bootstrap, mean ~0). ``lr=5`` was faster still but diverged ~1.3% of events, so ``lr=4`` is the
robust pick; keep ``mu=0.1`` (raising it with high lr destabilises). ``steps=150`` covers the
slowest cell's P90 + the last-40 Polyak window; the Polyak readout (median ties min‖g‖ BUT a single iterate has a ~2× worse tail — ‖g‖ dips on
noise fluctuations where the vtx is far; the last-40 average suppresses that). The ~15cm vertex floor
is SIREN-emitter-bias-limited (readout-probe-proven: min-data-loss readout = Polyak = the loss
minimum; the ~4cm "oracle" is an unreachable fluctuation toward truth), NOT optimizer-limited.
DELTA=1.0. SIGMA is the per-photon time resolution and must match the TTS of the data, so it is
a property of the data rather than of the method and has no default: the paper pipeline uses
``tts=2.1`` with ``sigma=2.1`` (``analysis/paper/utils/studies.py``, ``TTS``).
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import ParticleParams
from lucid.losses import counts_loss, first_arrival_window_nll
from lucid.fitting.gn import gauss_newton
from lucid.utils import cartesian_to_spherical

# Natural per-parameter scales: ~50 MeV, 0.2 m, 0.02 cos-units, 0.2 ns.
SCALE9 = np.array([50., .2, .2, .2, .02, .02, .02, .02, .2])



@dataclass(frozen=True)
class FitRecipe:
    """The Fisher-Gauss-Newton fit of :func:`fit_track`; the names are :func:`gauss_newton`'s."""
    nkeys: int           # PRNG keys the gradient and metric average over, fixed for the whole fit
    steps: int           # Gauss-Newton iterations
    lr: float            # step scale, annealed linearly to lr_final
    lr_final: float
    lam: float           # Marquardt term lam * diag(Fs)
    mu: float            # additive Levenberg floor mu * median(diag Fs) * I; 0 runs t0 away
    refresh: int         # the metric is rebuilt every `refresh` steps
    max_step: float      # per-component clip on the SCALE9-scaled step
    polyak: int          # readout: the mean of the last `polyak` iterates

    def keys(self, key):
        """The ``nkeys`` fit keys of the event whose fit key is ``key``."""
        return list(jax.random.split(key, self.nkeys))


# The published fit (see the module docstring for how each value was chosen).
FIT_RECIPE = FitRecipe(nkeys=8, steps=150, lr=4.0, lr_final=1.5, lam=0.01, mu=0.1, refresh=8,
                       max_step=3.0, polyak=40)


@dataclass(frozen=True)
class ModelRecipe:
    """The predictor and the likelihood: everything about the model but the data's TTS (``sigma``)."""
    n_rays: int                                 # photons per forward
    K: int                                      # scattering iterations, and the gradient ones
    temperature: float                          # soft-overlap temperature of the predictor
    delta: float                                # first-arrival window (ns)
    time_weight: float                          # weight of the time NLL against the charge NLL
    energy_scale: str                           # 'nphot' (analytic nphot(E)) | 'simtotal'


# The published model. The Cherenkov lambda band is the emitter's own (its emission_band_nm, the
# PhotonSim band). 'nphot' is one forward pass and agrees with 'simtotal' within 0.4 MeV p68 per
# event (paired 250k test), 1.8x faster.
MODEL_RECIPE = ModelRecipe(n_rays=250_000, K=8, temperature=0.1, delta=1.0, time_weight=1.0,
                           energy_scale='nphot')


def vec9_angles(t9, xp=jnp):
    """``(theta, phi)`` of a 9-vector's ``(sinθ, cosθ)`` and ``(sinφ, cosφ)`` pairs.

    Each pair is normalised first, so a pair the fit has moved off the unit circle decodes to its
    angle (a zero pair to 0). ``xp`` is the array module: ``jax.numpy``, or ``numpy`` for float64.
    """
    st, ct, sp, cp = t9[4], t9[5], t9[6], t9[7]
    nt = xp.hypot(st, ct) + 1e-12
    npp = xp.hypot(sp, cp) + 1e-12
    return xp.arctan2(st / nt, ct / nt), xp.arctan2(sp / npp, cp / npp)


def track_from_vec9(t9):
    """9-vector ``[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]`` -> :class:`ParticleParams` (the ``sc2``
    bridge). Direction carried as sin/cos pairs (no atan2 wrap / kinky angle gradient)."""
    theta, phi = vec9_angles(t9)
    return ParticleParams(energy=t9[0], position=t9[1:4], theta=theta, phi=phi, t0=t9[8])


def vec9_dir(t9):
    """Unit direction (numpy, float64) from a 9-vector, or from a stack of them (``(..., 9)`` ->
    ``(..., 3)``, e.g. a fit trajectory) — for reporting angular error."""
    t9 = np.asarray(t9, float)
    st, ct, sp, cp = t9[..., 4], t9[..., 5], t9[..., 6], t9[..., 7]
    nt = np.hypot(st, ct); npp = np.hypot(sp, cp)
    if np.any(nt == 0.0) or np.any(npp == 0.0):
        raise ValueError(f'9-vector has a zero sin/cos pair, no direction: {t9[..., 4:8].tolist()}')
    st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp
    return np.stack([st * cp, st * sp, ct], -1)


def vec9_from_track(energy, position, direction, t0=0.0):
    """Build the 9-vector from physical ``(energy, position, direction, t0)``."""
    d = np.asarray(direction, float)
    pol, az = cartesian_to_spherical(d / np.linalg.norm(d), np)
    p = np.asarray(position, float)
    return np.array([float(energy), p[0], p[1], p[2],
                     np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), float(t0)])


class ReconProblem:
    """Wraps a :class:`ReconModel` for the shared loop in :mod:`lucid.fitting.gn`.

    Reconstruction is a LIKELIHOOD problem: the gradient is reverse-mode AD of a scalar and the
    metric is a separately-built PSD Fisher. There is no residual vector, which is why the shared
    loop is written against ``(g, H, loss)`` rather than anything phrased in residuals — that is
    the only interface both this and the least-squares calibration problem can satisfy.

    One policy lives here rather than in the loop, deliberately:

    * **Keys are fixed across iterations.** The same ``nkeys`` draws are reused every step, so
      ``grad`` is a deterministic function of ``theta`` — common random numbers, which removes the
      step-to-step sampling noise that would otherwise swamp a converging fit. Calibration does
      the opposite and redraws every step, because a residual linear in the model needs the
      expected objective rather than one noisy realisation. Only the problem can hold both.

    ``loss`` is ``None``: the loop only records it for diagnostics, so no scalar objective needs
    to be evaluated here. ``readout='ming'`` uses ``‖S·g‖``, not the loss.
    """

    def __init__(self, model, obs_counts, obs_times, keys):
        self.model = model
        self.oc = obs_counts
        self.ot = obs_times
        self.keys = list(keys)
        self._H = None

    def grad_metric_loss(self, theta, step, refresh=True):
        g = np.mean([np.asarray(self.model.grad(theta, self.oc, self.ot, k))
                     for k in self.keys], 0)
        if refresh or self._H is None:
            self._H = self.model.fisher_ad(theta, self.oc, self.ot, self.keys)
        return g, self._H, None


class ReconModel:
    """Wraps a per-photon track predictor into the per-PMT ``(μ charge, time-NLL)`` the recon
    Fisher-GN consumes, plus the assembled loss / gradient / Fisher metric.

    ``pred(track, key) -> (log_w, flat_times, flat_indices, total_charge)`` is a track
    simulator from ``setup_event_simulator(..., source='particle')``. The model carries its
    detector: the sensor count, and for the seeders the sensors, volume and light speed, are
    ``pred.det_geom``'s. :meth:`build` makes the predictor and the model from a
    :class:`ModelRecipe`.

    ``nphot_fn`` sets the energy scale: the emitter's analytic ``nphot(E)`` (one forward pass),
    or ``None`` for the simulator's own total charge (a second forward pass at live energy, ~2x
    the Fisher/Jacobian cost; for media or emitters without an analytic curve).

    There is no scalar charge normalisation: a factor on μ moves the fitted energy alone (about
    -884 MeV per unit; vertex, direction and t0 barely change), and without one the residual
    energy bias (-0.72% at 1 GeV) is well inside the 2.2% energy resolution. The forward is
    self-consistent, so tuning such a factor fits a scalar to noise.
    """

    def __init__(self, pred, *, sigma, nphot_fn, delta=1.0, time_weight=1.0):
        self.pred = pred
        self.ND = pred.det_geom.num_sensors
        self.sigma = float(sigma); self.delta = float(delta)
        self.time_weight = float(time_weight)                # explicit charge↔time weight on the time NLL.
        #   1.0 = the from-first-principles joint NLL  P(n)·P(t|n) = charge + conditional first-arrival;
        #   validated optimal (a balance sweep found tw=1 best; tw>1 trades a hair of P68 for failures).
        # The charge term's ENERGY gradient is routed through the TOTAL
        #   charge only (the Poisson factorises exactly into Poisson(Q;A)·Multinomial(shape) — energy
        #   lives in the scale A, geometry in the shape). Removes the profile bias a small emission-
        #   shape misspecification otherwise puts on energy via the soft (E↔geometry) degeneracy:
        #   validated 200 fits @1 GeV muon, dE −29.7±2.1 → +2.6±1.5, σ 30→21, geometry unchanged.
        #   Write μ = scale(E)·μ(sgE,g): forward value unchanged, ∂/∂E flows only through the global
        #   scale → ∂L/∂E ∝ (Σμ−Q); ∂/∂geometry still carries the per-PMT shape. Charge analog of the
        #   time term's AMP_DETACH. NOTE: the time term is built from the energy-DETACHED shape
        #   pass too, so (like AMP_DETACH for its amplitude) it carries no energy gradient —
        #   energy comes ONLY from the total-charge scale.
        self.nphot_fn = nphot_fn
        #   nphot_fn    : energy scale = analytic nphot(E) ratio (noise-free scalar).
        #   None        : energy scale = the SIM'S OWN total A=Σμ(E,g) — a 2nd forward pass at LIVE E
        #                 supplies A's exact ∂/∂E (real acceptance E-dependence), no nphot curve. The
        #                 shape pass (detached E) supplies pᵢ=μᵢ/A → μ=A·pᵢ routes ∂/∂E through A only.

        def _perpmt(t9, oc, ot, key):
            t9s = t9.at[0].set(jax.lax.stop_gradient(t9[0]))   # detach E in the shape
            lw, ft, fi, tot = pred(track_from_vec9(t9s), key)  # μ(sgE, g): geometry-live shape
            if self.nphot_fn is None:
                # energy scale from the sim's own total at LIVE E (values identical to the shape
                # pass since sgE==E in value; only the E-gradient differs). μ = A_live · pᵢ.
                _, _, _, tot_live = pred(track_from_vec9(t9), key)
                A_live = jnp.sum(tot_live)                     # Σμ(E,g): carries ∂A/∂E and ∂A/∂g
                A_shape = jnp.maximum(jnp.sum(tot), 1e-8)      # Σμ(sgE,g): ∂/∂g only
                mu = jnp.maximum(tot * (A_live / A_shape), 1e-8)
            else:                                              # analytic nphot(E) scale ratio
                # nphot(E)=A·E^b+c crosses zero near ~137 MeV -> escale sign-flip below it; floor it.
                _npn = jnp.maximum(self.nphot_fn(t9[0]), 1e3)
                _npd = jnp.maximum(self.nphot_fn(jax.lax.stop_gradient(t9[0])), 1e3)
                escale = _npn / _npd                          # =1 in value, carries ∂/∂E
                mu = jnp.maximum(tot * escale, 1e-8)
            mu_surv = jnp.maximum(tot, 1e-8)                     # UNSCALED survival denom — must NOT
            #                                                      be scaled, else far-capture dies.
            tnll = first_arrival_window_nll(lw, ft, fi, ot, mu_surv, oc, self.ND,
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

    @classmethod
    def build(cls, geom, physics, particle, *, sigma, recipe, grid):
        """The model at ``recipe`` (a :class:`ModelRecipe`) for ``particle`` on the geometry and
        physics configs ``geom`` and ``physics``; ``sigma`` is the data's TTS (ns) and ``grid``
        the propagation grid keywords of ``setup_event_simulator``."""
        from lucid.simulation import setup_event_simulator
        if recipe.energy_scale not in ('nphot', 'simtotal'):
            raise ValueError(f"energy_scale must be 'nphot' or 'simtotal', got {recipe.energy_scale!r}")
        pred = setup_event_simulator(
            geom, recipe.n_rays, source='particle', temperature=recipe.temperature, K=recipe.K,
            physics_config=physics, default_detector_params=True, particle=particle,
            wavelength_mode=True, **grid)
        nphot_fn = pred.emitter.n_photons if recipe.energy_scale == 'nphot' else None
        return cls(pred, sigma=sigma, nphot_fn=nphot_fn, delta=recipe.delta,
                   time_weight=recipe.time_weight)

    def perpmt(self, t9, oc, ot, key):
        return self._perpmt(jnp.asarray(t9), oc, ot, key)

    def loss(self, t9, oc, ot, key):
        return self._loss(jnp.asarray(t9), oc, ot, key)

    def grad(self, t9, oc, ot, key):
        return self._grad(jnp.asarray(t9), oc, ot, key)

    def fisher_ad(self, t9, oc, ot, keys):
        """PSD Fisher metric ``Jμᵀ diag(1/μ) Jμ + Jlᵀ Jl`` via FORWARD-MODE AD (``jacfwd`` of
        ``perpmt``), averaged over ``keys``."""
        t9 = jnp.asarray(t9, float); ND = self.ND
        Jm = np.zeros((ND, 9)); Jl = np.zeros((ND, 9)); mu0 = np.zeros(ND)
        for k in keys:
            (jm, jl) = self._pjac(t9, oc, ot, k)     # each (ND, 9)
            Jm += np.asarray(jm); Jl += np.asarray(jl)
            mu0 += np.asarray(self._perpmt(t9, oc, ot, k)[0])
        n = len(keys); Jm /= n; Jl /= n; mu0 = np.maximum(mu0 / n, 1e-8)
        Jmn = Jm / np.sqrt(mu0)[:, None]
        return Jmn.T @ Jmn + Jl.T @ Jl


def fit_track(model, obs_counts, obs_times, start, fit, *, key, on_step=None):
    """Consistent Fisher-Gauss-Newton track fit from ``start``, SCALE9-preconditioned, at the
    recipe ``fit`` (a :class:`FitRecipe`; :data:`FIT_RECIPE` is the published one). ``key`` is
    the event's own fit key (``derive_event_keys(...)['fit_key']`` in the paper): the fit
    averages over its ``fit.keys(key)``, so two events never share draws and an event fits the
    same way whatever else is fitted.

    The step is solved in SCALE9-scaled coordinates (``Fs = S⊗F⊗S``, ``gs = S·g``) with a
    Marquardt term ``lam·diag(Fs)`` and an ADDITIVE Levenberg floor ``mu·median(diag Fs)·I``
    (bounds the flat t0↔longitudinal eigenvalue — ``mu=0`` runs t0 away). ``lr`` amplifies the
    consistent GN under-shoot. ``max_step`` clips each scaled step (needed because routing energy
    through the total-charge scale steepens the energy gradient).

    The metric is the PSD Gauss-Newton/Fisher ``Jμᵀdiag(1/μ)Jμ + JlᵀJl``, built by forward-mode
    autodiff (``ReconModel.fisher_ad``); not the indefinite raw autodiff Hessian. The readout is
    the mean of the last ``polyak`` iterates, robust to the noise-floor wandering.
    ``on_step(step, theta, g, H, loss)`` is called after every iteration.

    Returns ``(theta, history)``: ``history`` holds the trajectory ``traj`` ((steps+1, 9)), the
    scaled ``gnorm`` ((steps+1,)), its argmin ``best_iter``, and ``n_rejected`` /
    ``rejected_steps``, the steps the non-finite guard refused.
    """
    oc = jnp.asarray(obs_counts); ot = jnp.asarray(obs_times)
    # The loop is lucid.fitting.gn.gauss_newton, shared with the calibration fit. ReconProblem
    # supplies (g, H, loss) and owns the recon-specific policy (fixed keys).
    prob = ReconProblem(model, oc, ot, fit.keys(key))
    res = gauss_newton(prob, np.asarray(start, float), fit.steps,
                       lam=fit.lam, mu=fit.mu, jitter=1e-9,
                       lr=fit.lr, lr_final=fit.lr_final, scale=SCALE9, max_step=fit.max_step,
                       refresh=fit.refresh, readout='polyak', polyak=fit.polyak,
                       reject_nonfinite=True, on_step=on_step)
    gnorm = np.array(res['gnorm'])
    return res['theta'], dict(traj=np.array(res['history']), gnorm=gnorm,
                              best_iter=int(np.argmin(gnorm)), n_rejected=int(res['n_rejected']),
                              rejected_steps=tuple(res['rejected_steps']))
