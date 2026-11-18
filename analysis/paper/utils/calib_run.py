"""Run one seed of the published calibration, through the library.

This replaces the campaign's engine (not in this repository) with :mod:`lucid.fitting`. The
damped Gauss-Newton loop, damping convention, step clip and Polyak readout come from
:mod:`lucid.fitting.gn`; this module holds the study -- which sources, which wavelengths, what
truth, where to start, and how to shard the forward across the available devices. Arguments are
typed and explicit, so nothing is read from the environment and a variable exported in the
caller's shell cannot change the estimator. Several seeds can run in one process and share one
compiled forward.

The published numbers depend on these being preserved exactly: the keys (the library's scheme,
:mod:`lucid.fitting.calib`: truth on ``truth_seed``'s truth stream, the fit and the gain readout
on ``seed``'s forward, Jacobian and readout streams), the three-stage start perturbation and the
order of its random draws, and the ``(S, NS)`` row layout.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from analysis.paper.utils import calibration as C
from lucid.fitting import (CalibrationParams, CalibrationForward, CalibrationProblem, calibrate,
                           closure_data, gauss_newton_tx)
from lucid.fitting.calib import ROLE_READOUT

__all__ = ['make_map_fn', 'map_mode', 'build_start', 'run_seed', 'build_shared']


def make_map_fn(n_group=7):
    """``map_fn`` for the forward: a pmap across devices when there are enough, else ``None``,
    the forward's own serial loop over the group (the same per-source jit, stacked).

    The serial loop is NOT bit-identical to the pmap: trajectories differ at ~1e-3 relative, well
    inside the published seed-to-seed spread, so one device reproduces the published numbers to
    within run-to-run noise, not bit for bit.
    """
    devices = jax.devices()
    if len(devices) < n_group:
        return None

    def map_fn(fn, in_axes):
        pm = jax.pmap(fn, in_axes=in_axes)

        def mapped(*args):
            # Chunk when the group is larger than the device count: `pmap` needs one device
            # per element and otherwise raises at call time, minutes into a fit. The
            # published layout (seven sources) takes the single-shot path.
            d = len(devices)
            n = next((int(jax.tree_util.tree_leaves(a)[0].shape[0])
                      for a, ax in zip(args, in_axes) if ax == 0), 1)
            if n <= d:
                return pm(*args)
            out = [pm(*[jax.tree_util.tree_map(lambda x: x[i:i + d], a) if ax == 0 else a
                        for a, ax in zip(args, in_axes)])
                   for i in range(0, n, d)]
            return jnp.concatenate(out, axis=0)

        return mapped
    return map_fn


def map_mode(n_group=7):
    """Which forward ``make_map_fn`` runs here, for the record: the two are not bit-identical."""
    n = len(jax.devices())
    return dict(map_mode='pmap' if n >= n_group else 'serial', n_devices=n)


def build_start(theta_true, params, seed, *, pert=0.4, pert_ls=2.0, pert_refl=1.2):
    """The published start: a perturbation of truth, in three stages.

    The stages are not interchangeable and neither is their order. One ``uniform(-pert, pert)``
    draw is taken for every parameter first, and only then are the scattering lengths and the
    reflection block OVERWRITTEN with their own wider draws. The first pass therefore consumes
    random numbers whose values are discarded but whose position in the stream is not — reordering
    or skipping it changes every subsequent draw, and so changes the start of every seed.

    The widths differ because the parameters do: scattering length is the least constrained by the
    charge pattern and needs the widest basin test, reflection sits between, and the rest are
    comparatively well determined.
    """
    P, NG = params.P, params.NG
    rng = np.random.default_rng(4000 + int(seed))
    start = np.asarray(theta_true, dtype=float) + rng.uniform(-pert, pert, P)
    for wl in range(params.W):
        start[3 * wl] = theta_true[3 * wl] + rng.uniform(-pert_ls, pert_ls)
    for j in range(NG, P):
        start[j] = theta_true[j] + rng.uniform(-pert_refl, pert_refl)
    return start


def run_seed(seed, *, steps, n_ph, k, grid, intensity, btruth, n_forward_draws, jacobian_draws,
             refresh, polyak, lam, mu, max_step, q_floor_frac, gauge, basis, mie_scatter_length,
             g, pert, pert_ls, pert_refl, truth_seed,
             sim=None, sources=None, verbose=True, on_step=None, progress_every=0):
    """Fit one seed and return the record the convergence figure reads.

    Every recipe value is required: the published values live in ``calibration.CALIB`` and
    nowhere else. ``sim`` and ``sources`` are accepted so that several seeds can share one
    compiled forward; when omitted they are built here.

    ``progress_every`` prints the step, the objective and the worst parameter every N steps, and
    ``on_step`` forwards a callback to the loop. Without either, a run reports only when the truth
    is built and when it finishes, which cannot tell a slow run from a stuck one.

    Returns a dict with ``truth``, ``theta0``, ``hist``, ``fit``, ``fit_rlogit``, ``chi2``,
    ``khat``, ``truth_k`` and the settings the run used.
    """
    t_start = time.time()
    W, NS = len(C.WAVELENGTHS), C.NS
    params = CalibrationParams(n_wavelengths=W, basis=basis,
                               mie_scatter_length=mie_scatter_length, g=g)
    map_fn = make_map_fn()

    if sim is None:
        sim = build_shared(n_ph, k, grid)
    if sources is None:
        sources = C._laser_sources(intensity)

    truths = [C.effective_truth(w) for w in C.WAVELENGTHS]
    theta_true = params.theta_from_physical(truths, C.WALL_R_MIX, C.WALL_FSPEC,
                                            C.SENSOR_R_MIX, C.SENSOR_FSPEC)
    truth_k = jnp.asarray(C.TRUTH_K)

    # Truth: the expected forward at truth with the truth gains baked in, averaged over `btruth`
    # draws on `truth_seed`. Identical across seeds by design — the seed varies the FIT's noise,
    # not the data's, which is what isolates forward Monte-Carlo noise as the thing the ensemble
    # spread measures.
    fwd = CalibrationForward(sim, sources, params, map_fn=map_fn)
    data = closure_data(fwd, theta_true, truth_k, n_draws=btruth, seed=truth_seed)
    if verbose:
        print(f'[cfg] seed{seed} N_PH={n_ph:,} n_forward_draws={n_forward_draws} BTRUTH={btruth} '
              f'P={params.P}; '
              f'truth built ({time.time() - t_start:.0f}s)', flush=True)

    start = build_start(theta_true, params, seed, pert=pert, pert_ls=pert_ls, pert_refl=pert_refl)

    cb = on_step
    if progress_every:
        tv = params.to_real(theta_true)

        def cb(step, theta, g, H, loss, _n=int(progress_every), _t0=t_start):
            if step % _n and step != steps - 1:
                return
            worst = float(np.abs(params.to_real(theta) / tv - 1).max())
            print(f'  seed{seed} step{step:4d}  chi2 {float(loss):.6g}  worst {worst:.1%}  '
                  f'{time.time() - _t0:.0f}s', flush=True)
            if on_step is not None:
                on_step(step, theta, g, H, loss)

    problem = CalibrationProblem(fwd, data, seed=seed, q_floor_frac=q_floor_frac, gauge=gauge,
                                 n_forward_draws=n_forward_draws, jacobian_draws=jacobian_draws)
    res = calibrate(problem, start, tx=gauss_newton_tx(lam=lam, mu=mu), steps=steps,
                    refresh=refresh, max_step=max_step, polyak=polyak, on_step=cb)

    theta_fit = np.asarray(res['theta'], dtype=float)
    # Gains at the answer, from a SINGLE forward draw on the seed's readout stream, outside the
    # fit's, so the reported map is not the one the last step happened to be conditioned on.
    #
    # Done in numpy, not jnp: jnp's summation order (as in `profile_gains`) differs from numpy's at
    # float32 epsilon, and the published gain map was reduced in numpy. `data.sum(0)` stays a jnp
    # reduction for the same reason: that is how the published readout computed it.
    readout_key = jax.random.fold_in(jax.random.PRNGKey(seed), ROLE_READOUT)
    m_final = np.asarray(fwd(jnp.asarray(theta_fit, dtype=jnp.float32), readout_key, jnp.ones(NS)))
    khat = np.clip(np.asarray(data.sum(0)) / (m_final.sum(0) + 1e-12), 1e-6, None)
    khat = khat / np.mean(khat) if gauge == 'linear' else khat / np.exp(np.mean(np.log(khat)))
    # From C.TRUTH_K directly, NOT from the float32 `truth_k` cast used for the forward: the
    # reported truth map is gauged in float64, and the cast would round it at ~1e-7.
    tk = np.asarray(C.TRUTH_K, dtype=float)
    tk = tk / np.mean(tk) if gauge == 'linear' else tk / np.exp(np.mean(np.log(tk)))

    hist = np.stack([params.to_real(t) for t in res['history'][1:]])
    # Truth in reported units, built from the PHYSICAL values rather than by inverting theta_true.
    # `to_real(theta_true)` is the same quantity mathematically but passes through log and logit
    # and back, and the round trip is not exact — it moves the dotted truth lines the figure draws,
    # and the `fit/truth - 1` errors quoted against them.
    tvec_opt = []
    for t in truths:
        tvec_opt += [t['scatter_length'], t['absorption_length'], t['qe']]
    tvec = np.array(tvec_opt + [C.WALL_R_MIX * C.WALL_FSPEC, C.WALL_R_MIX * (1 - C.WALL_FSPEC),
                                C.SENSOR_R_MIX * C.SENSOR_FSPEC,
                                C.SENSOR_R_MIX * (1 - C.SENSOR_FSPEC)])
    if verbose:
        worst = np.max(np.abs(np.asarray(res['real']) / tvec - 1))
        print(f'[conv] seed{seed}: worst {worst:.1%}  k_RMS {np.std(khat / tk - 1) * 100:.2f}%  '
              f'{time.time() - t_start:.0f}s', flush=True)

    return dict(seed=seed, truth=tvec, theta0=np.asarray(theta_true), hist=hist,
                fit=np.asarray(res['real']), fit_rlogit=theta_fit, chi2=np.asarray(res['loss']),
                khat=khat, truth_k=tk, W=W, NG=params.NG, P=params.P,
                wls=np.asarray(C.WAVELENGTHS))


def build_shared(n_ph, k, grid):
    """The calibration simulator, built once so several seeds share one compiled forward."""
    return C._cal_sim(n_ph, k, grid)
