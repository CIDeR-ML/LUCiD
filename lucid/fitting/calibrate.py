"""Calibrating a detector: the one call, and the closure that tells you whether to trust it.

The pieces below this module — a parameterisation, a forward, a Jacobian, a problem, a loop —
are each separately reviewable, which is why they are separate. But assembling five objects
before a first fit is a real cost, and a newcomer who pays it by copying a figure script inherits
whatever that script happened to do. So the assembly is written once, here.

Two entry points, and the order matters:

``closure``    generate data from parameters you chose, perturb, fit, and see what comes back.
               Do this FIRST on any new detector. It is the only way to distinguish a broken
               configuration from a genuinely hard one, and it costs nothing but compute — the
               truth is known because you set it.
``calibrate``  the same fit against real data.

``fit`` is the field-based driver: the same machinery reached through
:func:`lucid.fitting.problem.build_calibration_problem`, which parameterises the fit by naming
``DetectorParams`` leaf fields rather than by the published estimand's fixed layout.

The estimator
-------------
One estimator serves all three, and it is the one the calibration campaign selected:

* **Neyman χ²** residual ``(k·M − Q)/√Q``. The weight depends on the data alone. That is not a
  detail — the forward is a Monte-Carlo estimate redrawn every step, so any weight involving the
  model makes the residual nonlinear in it and permanently displaces the fixed point.
* **Profiled gains.** The per-PMT factor has an exact minimiser given the model, so it is solved
  each step rather than fitted. Order 10^4 nuisance parameters never enter the optimizer.
* **Damped Gauss-Newton**, shared with reconstruction (:mod:`lucid.fitting.gn`).

The earlier arm — a √-MSE residual with the gains carried as a free Schur block — is gone. It was
the arm the campaign rejected: a residual nonlinear in a re-drawn model, and a free per-sensor
gain that overfits a single noisy draw.
"""
import numpy as np
import jax.numpy as jnp

from lucid.fitting.calib import (
    CalibrationForward, CalibrationJacobian, CalibrationProblem,
)
from lucid.fitting.gn import gauss_newton
from lucid.fitting.params import LogParams

__all__ = ['calibrate', 'closure', 'closure_data', 'fit']


def _seeded_keys(seed):
    """The reference engine's per-seed key bases, so an ensemble is reproducible from its seed.

    The Jacobian's stream must be independent of the residual's — sharing them was measured at
    137 sigma of covariance — and must carry the seed. Without a seed term every ensemble member
    draws the SAME Jacobian noise, so any fixed point it displaces moves them all alike: the error
    appears as bias and contributes nothing to the spread, leaving the ensemble's own error bar
    blind to it.
    """
    return 1000 + 777 * int(seed), 9_000_000 + 1_234_567 * int(seed)


def calibrate(sim, sources, params, data, theta0, *,
              steps=150, lam=0.01, mu=0.1, max_step=0.5, jitter=0.0,
              refresh=2, refresh_final=None, refresh_switch=0.5,
              n_forward_draws=1, jacobian_draws=2,
              q_floor=None, q_floor_frac=0.01, gauge='log', fix=(), seed=0,
              lr=1.0, lr_final=None, scale=None, readout='final', polyak=0,
              map_fn=None, predict=None, forward=None, on_step=None):
    """Fit detector parameters to observed per-sensor charge.

    Parameters
    ----------
    sim : callable
        ``sim(source, detector_params, key) -> (charge, time)``, from
        ``setup_event_simulator(..., is_calibration=True, hit_mode='aggregated')``.
    sources : sequence
        The calibration sources. Source 0 is the singleton in the sharded dispatch; that affects
        execution only, never the result.
    params : parameterisation
        :class:`~lucid.fitting.params.CalibrationParams` for the published estimand, or
        :class:`~lucid.fitting.params.FieldParams` to fit named ``DetectorParams`` leaves.
    data : array ``(S, n_sensors)``
        Observed charge, rows ordered ``g = wl*n_sources + s`` to match the forward. With a single
        wavelength that is simply one row per source.
    theta0 : array ``(P,)``
        Starting fit vector.
    steps, lam, mu, max_step, refresh, lr, scale, readout, polyak, fix
        Passed through to :func:`lucid.fitting.gn.gauss_newton`; see it for the conventions.
    n_forward_draws : int
        Forward draws averaged into the residual each step. Averaging reduces the Monte-Carlo
        variance of the model, which is what a residual linear in the model needs for its fixed
        point to sit at truth.
    jacobian_draws : int
        Jacobian draws averaged per refresh, on a stream independent of the residual's.
    q_floor, q_floor_frac
        Floor on the Neyman weight's denominator, absolute or as a fraction of ``mean(data)``.
        It exists because ``1/sqrt(Q)`` diverges on an empty sensor: without it a handful of nearly
        unlit sensors dominate the fit.
    gauge : ``'log'`` or ``'linear'``
        Gain gauge; see :func:`lucid.fitting.calib.profile_gains`.
    seed : int
        Selects both key streams. Vary it to build an ensemble.
    predict, map_fn
        Forward dispatch overrides; see :class:`~lucid.fitting.calib.CalibrationForward`.
    forward : CalibrationForward or None
        A forward built earlier, to be reused rather than rebuilt. Purely a cost argument: a
        caller that already has one — because it generated its own data with
        :func:`closure_data`, as the paper driver does — would otherwise pay a second trace and
        compile of the identical computation. Ignored by the answer; validated against ``data``.

    Returns
    -------
    dict with ``theta`` (fit vector), ``real`` (``params.to_real(theta)``), ``gains``
    (the profiled per-PMT map at the answer), ``history``, ``loss``, ``gnorm``, and ``problem``.
    """
    data = jnp.asarray(data)
    n_sensors = int(data.shape[1])
    fkey, jkey = _seeded_keys(seed)
    if q_floor is None:
        q_floor = q_floor_frac * float(jnp.mean(data))

    fwd = forward if forward is not None else CalibrationForward(
        sim, sources, params, n_sensors, map_fn=map_fn, predict=predict)
    if forward is not None and fwd.NS != n_sensors:
        raise ValueError(f'the supplied forward has {fwd.NS} sensors but data has {n_sensors}')
    jac = CalibrationJacobian(sim, sources, params, n_sensors, key0=jkey,
                              map_fn=map_fn, predict=predict)
    if data.shape[0] != fwd.S:
        raise ValueError(f'data has {data.shape[0]} rows but the forward produces {fwd.S} '
                         f'({fwd.W} wavelengths x {fwd.n_sources} sources)')

    prob = CalibrationProblem(fwd, jac, params, data, q_floor, gauge=gauge,
                              n_forward_draws=n_forward_draws, jacobian_draws=jacobian_draws,
                              forward_key0=fkey)
    res = gauss_newton(prob, jnp.asarray(theta0, dtype=jnp.float32), steps,
                       lam=lam, mu=mu, max_step=max_step, jitter=jitter,
                       lr=lr, lr_final=lr_final, scale=scale,
                       refresh=refresh, refresh_final=refresh_final,
                       refresh_switch=refresh_switch,
                       readout=readout, polyak=polyak, fix=fix, on_step=on_step)
    theta = res['theta']
    return dict(theta=theta, real=params.to_real(theta),
                gains=np.asarray(prob.gains(jnp.asarray(theta, dtype=jnp.float32), steps)),
                history=res['history'], loss=res['loss'], gnorm=res['gnorm'], problem=prob)


def closure_data(forward, theta_true, gains=None, *, n_draws=8, key_base=5000):
    """Pseudo-data: the forward at known parameters, with known per-PMT gains baked in.

    ``n_draws`` is not a nicety. The Neyman weight ``1/sqrt(Q)`` is built from the data, so noise
    in the data is noise in the weight. Pass ``1`` to model a single real exposure — but then read
    the answer as one draw, not as a measurement of the estimator.

    The default is a round number, not a measured optimum: the published run uses 8
    (``CALIB_RECIPE['BTRUTH']``), and no sweep in this tree identifies a best value.
    """
    g = jnp.ones(forward.NS) if gains is None else jnp.asarray(gains)
    return forward.average(jnp.asarray(theta_true, dtype=jnp.float32), key_base, g, n_draws)


def closure(sim, sources, params, theta_true, *, gains=None, perturbation=0.4,
            n_truth_draws=8, truth_key_base=5000, seed=0, n_sensors=None, **kw):
    """Generate data from parameters you chose, perturb, fit, and report what came back.

    This is the first thing to run on a new detector, and it answers a question no fit to real
    data can: whether the setup can recover parameters at all. The truth is known because you set
    it, so a poor recovery here is a property of the configuration — the source layout, the photon
    budget, the parameterisation — and not of the data.

    Parameters
    ----------
    theta_true : array ``(P,)``
        The parameters the pseudo-data is generated from.
    gains : array ``(n_sensors,)`` or None
        Truth per-PMT gains baked into the data. ``None`` means unit gains — which makes the
        recovered gain map an unrealistically easy test, so pass a real spread when it matters.
    perturbation : float or array ``(P,)``
        The starting offset from truth. A scalar draws ``uniform(-p, p)`` per component from
        ``seed``; an array is used as given. A fit started AT truth tests almost nothing: it
        cannot show whether the basin reaches from where a real starting guess would be.
    n_truth_draws, truth_key_base
        Passed to :func:`closure_data`.
    n_sensors : int or None
        Needed only when it cannot be read from ``gains``; otherwise inferred.
    **kw
        Forwarded to :func:`calibrate`.

    Returns
    -------
    The :func:`calibrate` result, plus ``real_truth``, ``frac_error`` (per reported component),
    ``theta_true``, ``theta_start``, and ``gains_truth``.
    """
    if n_sensors is None:
        if gains is None:
            raise ValueError('pass n_sensors, or gains from which it can be read')
        n_sensors = int(np.asarray(gains).shape[0])
    theta_true = np.asarray(theta_true, dtype=float)

    fwd = CalibrationForward(sim, sources, params, n_sensors,
                             map_fn=kw.get('map_fn'), predict=kw.get('predict'))
    data = closure_data(fwd, theta_true, gains, n_draws=n_truth_draws, key_base=truth_key_base)

    if np.ndim(perturbation) == 0:
        rng = np.random.default_rng(4000 + int(seed))
        start = theta_true + rng.uniform(-float(perturbation), float(perturbation),
                                         theta_true.shape[0])
    else:
        start = theta_true + np.asarray(perturbation, dtype=float)

    res = calibrate(sim, sources, params, data, start, seed=seed, **kw)
    real_truth = params.to_real(theta_true)
    res.update(real_truth=real_truth,
               frac_error=np.asarray(res['real']) / np.asarray(real_truth) - 1.0,
               theta_true=theta_true, theta_start=start,
               gains_truth=None if gains is None else np.asarray(gains))
    return res


_RETIRED = {
    'eps': 'the sqrt-MSE residual offset; the Neyman weight is floored by q_floor instead',
    'bake_k': 'gains are now always profiled in closed form — this was the arm that did so',
    'kstep_max': 'there is no per-PMT iterate to clip; the gains are solved, not stepped',
    'lk0': 'there is no per-PMT iterate to initialise',
    'gauge_k': "replaced by gauge='log' or 'linear'",
    'ridge': 'renamed lam, the Marquardt term of lucid.fitting.gn.damped_matrix',
    'nb_r': 'was INERT: declared in the old signature and never read. n_forward_draws IS '
             'live, so carrying the old value across is a behaviour change, not a rename',
    'nb_h': 'renamed jacobian_draws',
    'step_max': 'renamed max_step',
}


def fit(sources, truth_list, theta0, n_sensors, *, steps=300, refresh=15,
        lam=0.01, mu=0.1, max_step=0.08, n_forward_draws=1, jacobian_draws=4,
        q_floor_frac=0.01, gauge='log', fix=(), seed=0, polyak=0, **retired):
    """Calibrate through the field-based bridge: one ``SourceModel`` forward per source.

    This is what :func:`lucid.fitting.problem.build_calibration_problem` produces, so the pairing
    is unchanged::

        prob = build_calibration_problem(sim, sources, dp_true, ['scatter_length', ...])
        res  = fit(prob['source_models'], prob['truth_charge'], prob['theta0'], prob['num_sensors'])

    What changed is underneath: the residual is Neyman, the gains are profiled, and the loop is the
    one reconstruction uses. The numbers this returns therefore differ from the pre-consolidation
    ``fit`` — deliberately, since that residual was nonlinear in a re-drawn Monte-Carlo model and
    the free per-sensor gain block overfitted a single noisy draw.

    Retired keyword arguments raise rather than being quietly ignored, because each named a piece
    of machinery that no longer exists; silently accepting them would let a caller believe a knob
    was doing something.

    ``polyak`` selects the averaged readout: the trajectory does not settle, it wanders on the
    Monte-Carlo noise floor, so the mean of the last few iterates is usually the honest answer.

    Returns the :func:`calibrate` result, plus ``k`` (the profiled gains), ``log_theta``, and
    ``theta`` in physical units — the keys the bridge's callers read.

    .. warning::
       ``history`` kept its name and changed its meaning, which is the one migration hazard here
       that does not announce itself. It was the LINEAR trajectory with one row per step; it is now
       the LOG-space trajectory INCLUDING the starting point, so it has ``steps + 1`` rows. Code
       that plotted it directly will now plot log values as if they were physical, silently. Use
       ``np.exp(res['history'][1:])`` for the old quantity.
    """
    for k in retired:
        why = _RETIRED.get(k, 'no longer part of the calibration estimator')
        raise TypeError(f'fit() no longer accepts {k!r}: {why}')

    n_sensors = int(n_sensors)
    data = jnp.stack([jnp.asarray(t) for t in truth_list])
    forwards = [m.forward if hasattr(m, 'forward') else m for m in sources]

    def predict(theta, forward, wl, gains, key):
        # SourceModel.forward takes (theta, engine_key, photon_key) and returns the per-sensor
        # mean charge at unit gain. No forward this package builds reads the second key, so it
        # receives the same one; the gains are applied outside, exactly as that contract states.
        return jnp.asarray(gains) * forward(theta, key, key)

    res = calibrate(None, forwards, LogParams(int(np.asarray(theta0).shape[0])), data, theta0,
                    steps=steps, refresh=refresh, lam=lam, mu=mu, max_step=max_step,
                    n_forward_draws=n_forward_draws, jacobian_draws=jacobian_draws,
                    q_floor_frac=q_floor_frac, gauge=gauge, fix=fix, seed=seed,
                    readout='polyak' if polyak else 'final', polyak=polyak,
                    predict=predict)
    log_theta = np.asarray(res['theta'], dtype=float)
    res.update(theta=np.exp(log_theta), log_theta=log_theta, k=res['gains'])
    return res
