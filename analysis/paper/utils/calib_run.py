"""Run one seed of the published calibration, through the library.

The replacement for the campaign's engine (not in this repository). That engine executed at module level and read 38
environment variables across two modules, so it has to be launched as a subprocess per seed.
Everything it does that is *fitting* now lives in :mod:`lucid.fitting`, so what is left here is the
study: which sources, which wavelengths, what truth, where to start, and how to shard the forward
across the devices this machine happens to have.

Three things improve by construction, not by care:

* **No environment.** Arguments are typed and explicit, so a variable exported in the caller's
  shell cannot change the estimator. That was a live defect: of the 38 knobs the run read, 16
  were inherited — four estimator-changing, plus `INTEN`, which sets the photon budget of every
  source.
* **One process.** The seeds share a compiled forward instead of paying a fresh trace per
  subprocess.
* **One fitter.** The damped Gauss-Newton loop, the damping convention, the step clip and the
  Polyak readout come from :mod:`lucid.fitting.gn`, rather than being a fourth hand-written copy
  that has to be kept in step with the others by reading them.

What is preserved exactly, because the published numbers depend on it: the key arithmetic (truth
at ``5000 + 131*b``, residual at ``1000 + 777*seed + 13*step``, Jacobian at
``9_000_000 + 1_234_567*seed + ...``), the three-stage start perturbation and the order its random
draws are taken in, the gain readout at forward key 999, and the ``(S, NS)`` row layout. This was
verified bit-for-bit against the engine when it replaced it; that check is not shipped.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from analysis.paper.utils import calibration as C
from lucid.simulation import setup_event_simulator
from lucid.fitting import CalibrationParams, CalibrationForward, calibrate, closure_data

__all__ = ['make_map_fn', 'build_start', 'run_seed', 'recipe_to_kwargs']


def make_map_fn(n_group=7):
    """``map_fn`` for the forward: pmap across devices when there are enough, else a serial loop.

    The campaign ran on a 10-GPU node and dispatched the seven isotropic sources across it. With
    fewer devices than sources a pmap would crash, so the fallback runs the same per-source
    computation serially and stacks the results.

    That fallback is NOT numerically identical to the pmap, as this docstring used to claim.
    Measured at the published recipe, same allocation, same seed, only the visible device count
    differing (measured once): the trajectories part by ~1e-3
    relative (hist 8.7e-4, fit 5.4e-4, khat 1.3e-3). What holds is the weaker and sufficient
    statement — the PHYSICS agrees. Worst-parameter 2.07% on one device against 2.74% on seven is
    0.54x the published arm's own seed-to-seed spread, gain RMS is 0.88% either way, and both sit
    inside the engine's observed range across seeds. So one card reproduces the published numbers
    to within run-to-run noise, not bit for bit.

    Seven devices is also 3.5x faster: 22.7 min against 78.7 for the same fit.
    """
    devices = jax.devices()

    def map_fn(fn, in_axes):
        if len(devices) >= n_group:
            pm = jax.pmap(fn, in_axes=in_axes)

            def mapped(*args):
                # CHUNK when the group is larger than the device count. `pmap` demands one device
                # per element, so a group of 9 on 8 cards raises "requires 9 logical devices" --
                # at CALL time, minutes into a fit. That is reachable by any layout with more
                # structurally-identical sources than cards; the published layout has seven, so
                # it takes the single-shot path below exactly as before.
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
        jf = jax.jit(fn)

        def serial(*args):
            # Infer the count from the batched argument rather than closing over `n_group`.
            # CalibrationForward now groups sources by pytree structure and may map over a group
            # of any size, so a fallback fixed at one size would iterate the wrong number of
            # times on any layout but the published one. For that layout the leading axis IS
            # n_group, so this changes nothing there.
            n = n_group
            for arg, ax in zip(args, in_axes):
                if ax == 0:
                    n = int(jax.tree_util.tree_leaves(arg)[0].shape[0])
                    break
            outs = []
            for i in range(n):
                a = [jax.tree_util.tree_map(lambda x: x[i], arg) if ax == 0 else arg
                     for arg, ax in zip(args, in_axes)]
                outs.append(jf(*a))
            return jnp.stack(outs)
        return serial
    return map_fn


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


def run_seed(seed, *, steps=600, n_ph=int(1e6), k=12, btruth=8, nb_res=1, nbh=8,
             refresh=20, polyak=50, lam=0.01, mu=0.1, step_max=0.5, qfloor_frac=0.01,
             pert=0.4, pert_ls=2.0, pert_refl=1.2, gauge='log', basis='rlogit',
             intensity=1e8, truth_key_base=5000, gain_key=999,
             sim=None, sources=None, verbose=True, on_step=None, progress_every=0,
             deposit_leg_bound=False,
             tx=None):
    """Fit one seed and return the record the convergence figure reads.

    ``sim`` and ``sources`` are accepted so that several seeds can share one compiled forward;
    when omitted they are built here. Everything else is the published recipe, defaulted to the
    published values so that calling this with only a seed reproduces the paper.

    ``progress_every`` prints the step, the objective and the worst parameter every N steps, and
    ``on_step`` forwards a callback to the loop. A published run is 600 steps over more than an
    hour, and without either of these it reports twice — once when the truth is built and once
    when it is over — which is not enough to tell a slow run from a stuck one.

    Returns a dict with ``truth``, ``theta0``, ``hist``, ``fit``, ``fit_rlogit``, ``chi2``,
    ``khat``, ``truth_k`` and the settings the run used.
    """
    t_start = time.time()
    W, NS = len(C.WAVELENGTHS), C.NS
    params = CalibrationParams(n_wavelengths=W, basis=basis)
    map_fn = make_map_fn()

    if sim is None:
        # `deposit_leg_bound` reaches the deposit regardless of temperature=None -- the bound is
        # on the ray PARAMETER, not the overlap width -- so the hard-step calibration forward is
        # exposed to it like any other. Default False keeps the published calibration
        # bit-identical.
        sim = setup_event_simulator(C.GEOM, n_ph, temperature=None, K=k, is_calibration=True,
                                    hit_mode='aggregated', wavelength_mode=False,
                                    reflection_model='scalar_mix',
                                    deposit_leg_bound=deposit_leg_bound)
    if sources is None:
        # `intensity` defaults to the published VALUE, not to `calibration.INTEN`. That global is
        # a module-level environment read, and running in-process means it has already happened by
        # the time this function is called — so defaulting to it would leave exactly one argument
        # whose default came from the caller's shell, making the docstring's promise that a bare
        # `run_seed(seed)` reproduces the paper false on a dirty shell.
        sources = C._laser_sources(intensity=intensity)

    truths = [C.effective_truth(w) for w in C.WAVELENGTHS]
    theta_true = params.theta_from_physical(truths, C.WALL_R_MIX, C.WALL_FSPEC,
                                            C.SENSOR_R_MIX, C.SENSOR_FSPEC)
    truth_k = jnp.asarray(C.TRUTH_K)

    # Truth: the expected forward at truth with the truth gains baked in, averaged over `btruth`
    # draws. Identical across seeds by design — the seed varies the FIT's noise, not the data's,
    # which is what isolates forward Monte-Carlo noise as the thing the ensemble spread measures.
    fwd = CalibrationForward(sim, sources, params, NS, map_fn=map_fn)
    data = closure_data(fwd, theta_true, truth_k, n_draws=btruth, key_base=truth_key_base)
    if verbose:
        print(f'[cfg] seed{seed} N_PH={n_ph:,} NB_RES={nb_res} BTRUTH={btruth} P={params.P}; '
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

    # `tx` routes the SAME problem through an optax transformation instead of the damped
    # Gauss-Newton solve. None -- the default, and what every published figure uses -- reaches the
    # untouched path, so the bit-exact pins in tests/reconciliation/ still hold. lam/mu are
    # forwarded only in that case; with a transformation they belong inside it and calibrate()
    # raises rather than double-apply them.
    step_kw = {} if tx is not None else dict(lam=lam, mu=mu)
    res = calibrate(sim, sources, params, data, start, steps=steps,
                    max_step=step_max, refresh=refresh, n_forward_draws=nb_res,
                    jacobian_draws=nbh, q_floor_frac=qfloor_frac, gauge=gauge, seed=seed,
                    readout='polyak' if polyak else 'final', polyak=polyak, map_fn=map_fn,
                    on_step=cb, tx=tx,
                    # the forward built above for the truth data, reused rather than re-traced
                    forward=fwd, **step_kw)

    theta_fit = np.asarray(res['theta'], dtype=float)
    # Gains at the answer, from a SINGLE forward draw at a key outside the fit's stream, so the
    # reported map is not the one the last step happened to be conditioned on.
    #
    # Done in numpy, not jnp, and that is not stylistic: `profile_gains` reduces with jnp, whose
    # summation order differs from numpy's, and the two disagree at float32 epsilon (measured
    # 4.8e-07 relative). The engine's readout is the numpy one, so reproducing the published gain
    # map exactly means reducing the same way. `data.sum(0)` stays a jnp reduction because that is
    # where the engine performs it.
    m_final = np.asarray(fwd(jnp.asarray(theta_fit, dtype=jnp.float32), gain_key, jnp.ones(NS)))
    khat = np.clip(np.asarray(data.sum(0)) / (m_final.sum(0) + 1e-12), 1e-6, None)
    khat = khat / np.mean(khat) if gauge == 'linear' else khat / np.exp(np.mean(np.log(khat)))
    # From C.TRUTH_K directly, NOT from the float32 `truth_k` cast used for the forward. The engine
    # gauges the reported truth map in float64; going through the forward's cast rounds it to ~1e-7,
    # and neither pin covers this — the archived reference has no `truth_k` key.
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

    # The record is meant to be self-describing: a saved run must state the ESTIMATOR it used, not
    # only its cost knobs, or it cannot be checked against the recipe without trusting the driver
    # that wrote it. The engine recorded `loss`, `gauge` and `hcorr` for that reason; this driver
    # implements one arm, so it records that arm explicitly rather than dropping the fields.
    return dict(seed=seed, truth=tvec, theta0=np.asarray(theta_true), hist=hist,
                fit=np.asarray(res['real']), fit_rlogit=theta_fit, chi2=np.asarray(res['loss']),
                khat=khat, truth_k=tk, W=W, NG=params.NG, P=params.P, N_PH=n_ph,
                wls=np.asarray(C.WAVELENGTHS), steps=steps, refresh=refresh, nbh=nbh,
                btruth=btruth, nb_res=nb_res, polyak=polyak, lam=lam, mu=mu, gauge=gauge,
                basis=basis, k_bounces=k,
                # estimator provenance
                loss='neyman', gains_mode='profiled', metric='neyman', solver='solve',
                hcorr=0, tau=0.0, hfreeze=0, sample_truth=0, truth_random_iso=0, diff_high=0,
                jkey_seed=1, fixed_fwd=0,
                # the rest of what defines the run
                intensity=float(intensity), qfloor_frac=qfloor_frac, step_max=step_max,
                pert=pert, pert_ls=pert_ls, pert_refl=pert_refl,
                truth_key_base=truth_key_base, gain_key=gain_key,
                driver='calib_run.run_seed')


def build_shared(n_ph, k, intensity=None, deposit_leg_bound=False):
    """The simulator and sources, built once so several seeds share one compiled forward.

    Each seed used to be its own subprocess, which meant tracing and compiling the whole forward
    from scratch every time. Nothing about the fit required that — it was a consequence of the
    engine executing at module level and reading its configuration from the environment, so there
    was no way to run it twice in one process.
    """
    sim = setup_event_simulator(C.GEOM, n_ph, temperature=None, K=k, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,
                                reflection_model='scalar_mix',
                                deposit_leg_bound=deposit_leg_bound)
    return sim, C._laser_sources(intensity=intensity)


_INT = {'STEPS': 'steps', 'K': 'k', 'BTRUTH': 'btruth', 'NB_RES': 'nb_res', 'NBH': 'nbh',
        'REFRESH': 'refresh', 'POLYAK': 'polyak'}
_FLOAT = {'LAM': 'lam', 'MU': 'mu', 'STEP_MAX': 'step_max', 'QFLOOR_FRAC': 'qfloor_frac',
          'PERT': 'pert', 'PERT_LS': 'pert_ls', 'PERT_REFL': 'pert_refl', 'INTEN': 'intensity'}
_STR = {'GAUGE': 'gauge', 'REFL': 'basis'}
# Arms this driver does not implement, with the only value it can honour. Two groups: estimator
# arms the engine offers and this does not, and knobs that reach the ENGINE's simulator but not
# this one (GRID_MANUAL and the grid trio; FLOOR and HCORR_MIN are consumed only by the
# SOLVER=marq and HCORR=1 branches, which are themselves refused).
_REQUIRED = {'SAMPLE_TRUTH': '0', 'COMPUTE_CRB': '0', 'LOSS': 'neyman', 'GAINS': 'profiled',
             'SOLVER': 'solve', 'METRIC': 'neyman', 'TAU': '0', 'HCORR': '0', 'HFREEZE': '0',
             'DIFF_HIGH': '0', 'TRUTH_RANDOM_ISO': '0', 'FIXED_FWD_SEED': '', 'JKEY_SEED': '1',
             'GRID_MANUAL': '0', 'NCAP': '100', 'NANG': '150', 'NHGT': '100',
             'FLOOR': '1e-5', 'HCORR_MIN': '0.03'}


def recipe_to_kwargs(recipe):
    """Translate the environment-string recipe into typed arguments.

    Narrow by design: every key must be either translated or explicitly pinned, and anything else
    RAISES. Ignoring what it does not understand is how a configuration silently stops meaning what
    it says, which is the defect this module exists to remove — and an earlier version of this
    function had exactly that bug for six keys, `GRID_MANUAL` among them. That one is genuinely
    reachable: under it the engine applies a manual photon grid, and this driver cannot.

    Note what ``_REQUIRED`` pins: the value of the PUBLISHED arm, which for `LOSS`, `SOLVER` and
    `JKEY_SEED` is NOT the engine's own default. It states what this driver implements, not what
    the engine does in the absence of configuration.
    """
    kw = {}
    for env, name in _INT.items():
        if env in recipe:
            kw[name] = int(float(recipe[env]))
    for env, name in _FLOAT.items():
        if env in recipe:
            kw[name] = float(recipe[env])
    for env, name in _STR.items():
        if env in recipe:
            kw[name] = recipe[env]
    if 'N_PH' in recipe:
        kw['n_ph'] = int(float(recipe['N_PH']))

    bad = {k: recipe[k] for k, v in _REQUIRED.items() if k in recipe and recipe[k] != v}
    if bad:
        raise ValueError(f'calib_run implements the published arm only; these differ: {bad}')

    # Truth is generated at the fitted photon count. Compared NUMERICALLY and unconditionally: a
    # string compare would fire spuriously on '1e6' vs '1000000', and gating it on N_PH being
    # present let a lone TRUTH_NPH through to be silently ignored.
    t_nph = float(recipe.get('TRUTH_NPH', recipe.get('N_PH', 0)))
    n_ph_v = float(recipe.get('N_PH', 0))
    if t_nph != n_ph_v:
        raise ValueError('calib_run generates truth at the fitted photon count; '
                         f'TRUTH_NPH={t_nph:g} != N_PH={n_ph_v:g}')

    unknown = sorted(set(recipe) - set(_INT) - set(_FLOAT) - set(_STR) - set(_REQUIRED)
                     - {'N_PH', 'TRUTH_NPH', 'SEED', 'OUTDIR'})
    if unknown:
        raise ValueError(f'calib_run does not know what to do with {unknown}. Every recipe key '
                         'must be translated or explicitly pinned, or it is silently ignored.')
    return kw
