"""Calibration model and problem: the predicted per-sensor charge for every source × wavelength,
and the published estimator on it.

:class:`CalibrationForward` is the model, the calibration counterpart of
:class:`lucid.fitting.recon.ReconModel`. :class:`CalibrationProblem` adds the observed charge and
the estimator (Neyman residual, profiled gains, the key streams) and is what a solver runs on.

Preserved expression for expression from the paper's reference calibration engine, its keys
aside. The whole stack runs in float32 (``jax_enable_x64`` is never enabled), so an equivalent but
re-associated expression can move the result. The reference engine is not in this repository, so
no shipped test gates that agreement. Anything here that looks gratuitously specific (the
concatenate-then-transpose, the laser/isotropic split) is load-bearing for that reason.

Keys
----
Every random stream starts from ``fold_in(PRNGKey(seed), role)``, one role per stream
(:data:`ROLE_TRUTH` for :func:`~lucid.fitting.calibrate.closure_data`, :data:`ROLE_FORWARD` and
:data:`ROLE_JACOBIAN` for :class:`CalibrationProblem`, :data:`ROLE_READOUT` for a draw outside
the fit), and folds in its indices one at a time: step, then draw, then source, then wavelength.
No index is added to another, so no two (seed, role, step, draw, source, wavelength) share a key.

Layout
------
``S = W * n_sources`` configurations, indexed ``g = wl * n_sources + s`` — wavelength-major, so
each row of the returned ``(S, NS)`` array is one (wavelength, source) pair.

Dispatch
--------
Each source reaches the simulator as a **pytree of arrays**, so one compiled program serves every
source of one pytree structure with the source as a traced argument — that is what makes a
``pmap`` over sources possible at all.

Sharding
--------
The reference dispatches source 0 (a collimated laser) to one device and the remaining isotropic
sources to a ``pmap``. That is an execution strategy, not physics, so it is injected via
``map_fn`` rather than baked in. The default is a serial loop over sources, which is also what
the reference itself falls back to below 7 devices.
"""
import functools

import jax
import jax.numpy as jnp

__all__ = ['CalibrationForward', 'CalibrationProblem', 'profile_gains', 'neyman_residual']

ROLE_TRUTH, ROLE_FORWARD, ROLE_JACOBIAN, ROLE_READOUT = range(4)


class CalibrationForward:
    """Mean per-sensor charge at fit vector ``theta``, for every source × wavelength.

    Parameters
    ----------
    sim : callable
        ``sim(source, detector_params, key) -> (charge, time)``, from
        ``setup_event_simulator(..., source='light_source')`` (first-arrival hits, expected).
    sources : sequence
        Calibration sources. Source 0 is treated as the "singleton" (the laser in the reference
        layout) and the rest as the mappable group — this only affects dispatch, not the result.
    params : parameterisation
        Supplies ``W`` and ``to_dp`` (:mod:`lucid.fitting.params`).
    n_sensors : int or None
        Defaults to the simulator's, ``sim.det_geom.num_sensors``.
    map_fn : callable or None
        ``map_fn(fn, in_axes)`` returning a batched callable over the non-singleton sources.
        ``None`` (default) uses a serial loop that is numerically identical.
    """

    def __init__(self, sim, sources, params, n_sensors=None, *, map_fn=None, laser_device=None):
        self.sim = sim
        self.sources = list(sources)
        self.params = params
        self.NS = int(sim.det_geom.num_sensors if n_sensors is None else n_sensors)
        self.W = params.W
        self.n_sources = len(self.sources)
        self.S = self.W * self.n_sources
        self.map_fn = map_fn

        def _mbody(theta, src, keys, gains):                      # (W, NS); vmap over wavelengths
            return jax.vmap(lambda wl, k: sim(src, params.to_dp(theta, wl, gains), k)[0])(
                jnp.arange(self.W), keys)

        def _jbody(theta, src, wl, lk, key, q_cfg, q_floor):      # (NS, P); see `jacobian`
            def sm(th):
                mu = sim(src, params.to_dp(th, wl, jnp.ones(self.NS)), key)[0]
                return jnp.exp(lk) * mu / jnp.sqrt(jnp.clip(q_cfg, q_floor, None))
            return jax.jacfwd(sm)(theta)

        self._mbody = _mbody
        self._single = jax.jit(_mbody)
        self._jsingle = jax.jit(_jbody)
        # Where the singleton source runs. `map_fn` pmaps the grouped sources over devices
        # 0..n_group-1, and an un-placed `_single` runs on JAX's default device (device 0), so
        # the laser would serialise behind pmap shard 0 instead of overlapping it. Placement is
        # value-neutral: it buys wall clock and moves no number. `laser_device` overrides; None
        # picks the first card outside the mapped range, falling back to the last available when
        # there is none.
        self._laser_dev = laser_device
        if self._laser_dev is None and self.n_sources > 1:
            devs = jax.devices()
            self._laser_dev = devs[min(self.n_sources - 1, len(devs) - 1)]
        # Sources are grouped by pytree structure and each group is mapped over its own stack;
        # singletons are called directly. This admits mixed layouts (e.g. a `LaserSource` beside
        # `IsotropicSource`s), which source diversity in calibration needs. The published layout
        # still resolves to one singleton (the laser) plus one mapped group of seven: the same two
        # calls, in the same order, on the same keys, so it is bit-exact by construction.
        self._groups = _group_by_structure(self.sources)
        # Parts get disjoint devices. A single-source part runs unstacked on a card outside the
        # mapped range, so it runs alongside the mapped parts rather than queueing behind one of
        # their shards. In the published layout that part is the laser, but nothing here knows
        # what a laser is.
        self._placed = next((ix[0] for ix, _ in self._groups if len(ix) == 1), None)
        self._largest = max((len(ix) for ix, _ in self._groups), default=0)
        # The Jacobian maps over the SAME groups: its rows are assembled in source order to match
        # the forward's, so a different partition would pair row `g` with another source's column.
        self._group_fns, self._jgroup_fns, self._group_stacks = [], [], []
        for idx, stack in self._groups:
            if len(idx) == 1:
                self._group_fns.append(None)            # singleton: use self._single
                self._jgroup_fns.append(None)
                self._group_stacks.append(None)
                continue
            self._group_fns.append(self._make_map(_mbody, (None, 0, 0, None), len(idx)))
            self._jgroup_fns.append(self._make_map(_jbody, (None, 0, None, None, 0, 0, None),
                                                   len(idx)))
            self._group_stacks.append(stack)

    def _make_map(self, fn, in_axes, n):
        """A mapped callable over `n` items, honouring a caller-supplied `map_fn`.

        `map_fn(fn, in_axes)` is the established two-argument contract. A `pmap` built from it
        maps over whatever leading axis it is handed, so it serves any group; only the serial
        fallback needs the count `n`.
        """
        if self.map_fn is None:
            return _serial_map(fn, in_axes, n)
        return self.map_fn(fn, in_axes)

    def keys(self, key):
        """The ``(n_sources, W)`` keys of one draw on ``key``: ``fold_in(fold_in(key, s), wl)``.

        Indexed by the source's ABSOLUTE position, so a source's random stream does not depend on
        how the layout happens to be grouped.
        """
        return _fold_grid(key, self.n_sources, self.W)

    def __call__(self, theta, key, gains):
        """-> ``(S, NS)`` mean charge, row ``g = wl * n_sources + s``, on the draw ``key``."""
        allk = self.keys(key)

        slots = [None] * self.n_sources
        for (idx, _), fn, stack in zip(self._groups, self._group_fns, self._group_stacks):
            if fn is None:                                   # singleton
                i = idx[0]
                # Put a singleton on its own card so it overlaps a grouped pmap rather than
                # queueing behind shard 0 (see __init__).
                d = self._laser_dev if i == self._placed else None
                if d is None:
                    slots[i] = self._single(theta, self.sources[i], allk[i], gains)
                else:
                    slots[i] = self._single(jax.device_put(theta, d),
                                            jax.device_put(self.sources[i], d),
                                            jax.device_put(allk[i], d),
                                            jax.device_put(gains, d))
                continue
            out = fn(theta, stack, allk[jnp.asarray(idx)], gains)    # (len(idx), W, NS)
            for j, i in enumerate(idx):
                slots[i] = out[j]
        per = jnp.stack([jnp.asarray(v) for v in slots])      # (n, W, NS), SOURCE order
        return jnp.transpose(per, (1, 0, 2)).reshape(self.S, self.NS)

    def average(self, theta, key, gains, n_draws):
        """Mean over ``n_draws`` independent forward draws, draw ``b`` on ``fold_in(key, b)``.

        Averaging reduces the forward's Monte-Carlo variance. Were the residual linear in ``M``,
        ``E[r(M)] = r(E[M])`` and the fixed point would sit at truth at any variance. What makes
        ``n_draws`` matter is the profiled gain ``k = ΣQ/ΣM``: it is nonlinear in ``M``, so its
        expectation shifts with the forward's variance, and that shift propagates into the
        residual. The bias and its scaling with forward noise are measured in
        ``tests/fitting/test_fitting_estimator_unbiased.py``.
        """
        acc = jnp.zeros((self.S, self.NS))
        for b in range(n_draws):
            acc = acc + self(theta, jax.random.fold_in(key, b), gains)
        return acc / n_draws

    def jacobian(self, theta, lk, key, data, q_floor, draws=(0,)):
        """``∂r/∂θ`` of the Neyman residual, ``(S, NS, P)``, rows ordered like the forward's.

        Differentiated as ONE fused expression. It does not compute ``∂μ/∂θ`` and then apply the
        ``1/√Q`` weight; it runs ``jacfwd`` over the *already weighted* model::

            sm(θ) = exp(lk) · sim(src, dp(θ), key) / √clip(Q, floor)
            J     = jacfwd(sm)(θ)

        Splitting that into a model Jacobian followed by a scaling is algebraically the same and
        **numerically different in float32**, which is the only precision this stack ever runs
        in. So the weight stays inside the differentiated function, and the data enter the
        Jacobian rather than only the residual.

        Two further properties are deliberate, not incidental:

        * the gains are passed as ones and the per-PMT factor enters as the constant ``exp(lk)``
          multiplier, so ``∂k/∂θ`` is dropped by construction: the gains are profiled, not
          fitted;
        * :class:`CalibrationProblem` computes ``lk`` from the *residual's* draw, so J and r
          share that one scalar per sensor even though their photon streams are disjoint. It
          vanishes exactly when the gains are held fixed. The magnitude of the residual coupling
          has not been measured; do not assume it small.

        Draw ``h`` runs on ``keys(fold_in(key, h))``, as :meth:`average`'s draws do. ``draws`` is
        the set of draw indices to average over, not merely a count, so that disjoint halves can
        be requested independently.
        """
        draws = tuple(draws)
        nh = len(draws)
        draw_keys = {h: self.keys(jax.random.fold_in(key, h)) for h in draws}
        rows = []
        for wl in range(self.W):
            base = wl * self.n_sources
            slots = [None] * self.n_sources
            for (idx, _), fn, stack in zip(self._groups, self._jgroup_fns, self._group_stacks):
                if fn is None:                                   # singleton
                    i = idx[0]
                    # Placed like the forward's singleton (see __init__). Value-neutral.
                    d = self._laser_dev if i == self._placed else None
                    if d is None:
                        slots[i] = sum(self._jsingle(theta, self.sources[i], wl, lk,
                                                     draw_keys[h][i, wl], data[base + i],
                                                     q_floor) for h in draws) / nh
                    else:
                        th_l, lk_l = jax.device_put(theta, d), jax.device_put(lk, d)
                        src_l = jax.device_put(self.sources[i], d)
                        q_l = jax.device_put(data[base + i], d)
                        slots[i] = sum(self._jsingle(
                            th_l, src_l, wl, lk_l,
                            jax.device_put(draw_keys[h][i, wl], d), q_l, q_floor)
                            for h in draws) / nh
                    continue
                out = sum(fn(theta, stack, wl, lk,
                             draw_keys[h][jnp.asarray(idx), wl],
                             jnp.stack([data[base + i] for i in idx]), q_floor)
                          for h in draws) / nh
                for j, i in enumerate(idx):
                    slots[i] = out[j]
            rows.append(jnp.stack([jnp.asarray(v) for v in slots]))       # (n, NS, P)
        return jnp.concatenate(rows, axis=0)


def profile_gains(model_charge, observed_sum, gauge='log', clip_min=1e-6):
    """Closed-form per-PMT gains ``k = ΣQ / ΣM``, gauged — the nuisance, solved not fitted.

    There is one gain per sensor (order 10^4 of them). Fitting them jointly would swamp the
    handful of optical parameters that are actually of interest, so they are profiled out at every
    step in closed form, at the cost of one division.

    ``ΣQ/ΣM`` is the exact minimiser of an UNWEIGHTED least squares (equivalently the Poisson
    scale MLE). It is NOT the minimiser of the Neyman chi-square this module actually forms in
    :func:`neyman_residual`, which is ``ΣM / Σ(M²/Q)``. Worked example: ``M=(1,10)``, ``Q=(2,10)``
    gives ``k=1.0909`` and ``chi2=0.4959`` here against ``k=1.0476`` and ``chi2=0.4762`` at the
    true Neyman minimum -- 4.1% higher. Two consequences follow and neither is cosmetic. The
    reported ``loss`` is not the profiled minimum; and since ``dchi2/dk = +0.909 != 0`` at this
    ``k``, the envelope theorem does NOT hold, so dropping ``dk/dtheta`` from the Jacobian (a
    deliberate choice, documented at :meth:`CalibrationForward.jacobian`) omits a genuinely non-zero
    term rather than a vanishing one.

    This is the published estimator and it is consistent -- ``E[r] = 0`` at truth for any gain
    map -- so the discrepancy is recorded rather than fixed. Changing ``k`` would move every
    published number.

    The gauge removes the exact degeneracy between a global gain and the overall light yield —
    without it the two directions are unidentifiable.

    ``'log'``    ``mean(log k) = 0``  — what the published run used.
    ``'linear'`` ``mean(k) = 1``      — ``k̂ = ΣQ/ΣM`` is linear in the data and therefore
                 unbiased, whereas taking its log first incurs a Jensen shift that is larger on
                 dim sensors. Not the published choice; adopting it would move published numbers.
    """
    k = jnp.clip(observed_sum / model_charge, clip_min, None)
    if gauge == 'linear':
        return k / jnp.mean(k)
    if gauge == 'log':
        return jnp.exp(jnp.log(k) - jnp.log(k).mean())
    raise ValueError(f"gauge must be 'log' or 'linear', got {gauge!r}")


def neyman_residual(model_charge, data, q_floor):
    """``r = (k·M − Q) / √clip(Q, floor)``.

    The weight depends on the DATA only. That is the whole point: the forward model is a
    Monte-Carlo estimate redrawn every step, so a weight involving ``M`` would make the residual
    nonlinear in it and bias the gradient — permanently displacing the fixed point rather than
    merely adding noise. Neyman is the member of the χ² family whose weight is data-only.
    """
    return (model_charge - data) / jnp.sqrt(jnp.clip(data, q_floor, None))


class CalibrationProblem:
    """A model and the observed charge: the published estimator, ready for any solver.

    The estimator is the Neyman residual ``(k·M − Q)/√clip(Q, floor)`` with the per-PMT gains
    ``k`` profiled in closed form (:func:`profile_gains`). Every method is a plain function of the
    fit vector ``theta`` and a ``step`` index, which selects the Monte-Carlo draw: the forward is
    redrawn at every step, so a fixed ``step`` gives a deterministic objective (common random
    numbers) and successive steps give independent draws.

    Two things this class owns that a solver must not:

    * **The keys.** The residual's forward draws from one stream (:data:`ROLE_FORWARD`) and the
      Jacobian from another (:data:`ROLE_JACOBIAN`), which must be independent of it (shared keys
      correlate J with r) and must carry the seed.
      Without a seed term in the Jacobian's stream every ensemble member draws the SAME Jacobian
      noise, so any fixed point it displaces moves them all alike: the error appears as bias and
      contributes nothing to the spread, leaving the ensemble's own error bar blind to it.
    * **The nuisance.** The gains are profiled each step, so they never enter ``theta``. This is
      why calibration needs no Schur block in the fitter, and why one loop serves calibration and
      reconstruction.

    Parameters
    ----------
    forward : CalibrationForward
    data : array ``(S, n_sensors)``
        Observed charge, rows ordered ``g = wl*n_sources + s`` to match the forward. With a single
        wavelength that is one row per source.
    seed : int
        Selects both key streams. Vary it to build an ensemble.
    q_floor_frac : float
        Floor on the Neyman weight's denominator, as a fraction of ``mean(data)``. It exists
        because ``1/sqrt(Q)`` diverges on an empty sensor: without it a handful of nearly unlit
        sensors dominate the fit.
    gauge : ``'log'`` or ``'linear'``
        Gain gauge; see :func:`profile_gains`.
    n_forward_draws : int
        Forward draws averaged into the residual each step. The Neyman residual is linear in
        ``M``, so on its own averaging would not move the fixed point; it matters because the
        profiled gain ``k = ΣQ/ΣM`` is nonlinear in ``M``
        (``tests/fitting/test_fitting_estimator_unbiased.py`` measures the bias).
    jacobian_draws : int
        Jacobian draws averaged per refresh, on the independent stream.
    """

    def __init__(self, forward, data, *, seed=0, q_floor_frac=0.01, gauge='log',
                 n_forward_draws=1, jacobian_draws=8):
        data = jnp.asarray(data)
        if data.shape[0] != forward.S:
            raise ValueError(f'data has {data.shape[0]} rows but the forward produces '
                             f'{forward.S} ({forward.W} wavelengths x {forward.n_sources} sources)')
        self.forward = forward
        self.data = data
        self.data_sum = data.sum(0)
        self.q_floor = q_floor_frac * float(jnp.mean(data))
        self.gauge = gauge
        self.n_forward_draws = int(n_forward_draws)
        self.jacobian_draws = int(jacobian_draws)
        root = jax.random.PRNGKey(int(seed))
        self._forward_root = jax.random.fold_in(root, ROLE_FORWARD)
        self._jacobian_root = jax.random.fold_in(root, ROLE_JACOBIAN)
        self.n_configs = int(data.shape[0])
        self._J = None                                     # cached between refreshes

    def forward_key(self, step):
        """The residual's key on step ``step``; its draws fold in below it (:meth:`average`)."""
        return jax.random.fold_in(self._forward_root, step)

    def jacobian_key(self, step):
        """The Jacobian's key on step ``step``, on its own stream."""
        return jax.random.fold_in(self._jacobian_root, step)

    def predict(self, theta, step=0):
        """Model charge at unit gain, ``(S, n_sensors)``, on step ``step``'s draw."""
        return self.forward.average(theta, self.forward_key(step), jnp.ones(self.data.shape[1]),
                                    self.n_forward_draws)

    def _gains(self, mu):
        return profile_gains(mu.sum(0) + 1e-12, self.data_sum, gauge=self.gauge)

    def gains(self, theta, step=0):
        """The profiled per-PMT gains at ``theta``: the per-sensor QE/gain map, ``(n_sensors,)``."""
        return self._gains(self.predict(theta, step))

    def residuals(self, theta, step=0):
        """The Neyman residual, ``(S, n_sensors)``."""
        mu = self.predict(theta, step)
        return neyman_residual(self._gains(mu)[None, :] * mu, self.data, self.q_floor)

    def _jacobian(self, theta, lk, step):
        return self.forward.jacobian(theta, lk, self.jacobian_key(step), self.data, self.q_floor,
                                     draws=range(self.jacobian_draws))

    def jacobian(self, theta, step=0):
        """``∂r/∂θ``, ``(S, n_sensors, P)``, on the Jacobian's own stream; ``∂k/∂θ`` is dropped
        (:meth:`CalibrationForward.jacobian`)."""
        return self._jacobian(theta, jnp.log(self.gains(theta, step)), step)

    def loss(self, theta, step=0):
        """``Σr² / S``."""
        return float(jnp.sum(self.residuals(theta, step) ** 2) / self.n_configs)

    def grad_metric_loss(self, theta, step, refresh=True):
        """``(Jᵀr/S, JᵀJ/S, loss)``, the seam :func:`lucid.fitting.minimize.minimize` steps on.

        ``J`` is rebuilt only when ``refresh`` and is otherwise the cached one, so on a
        non-refresh step the search direction is stale too, not only its metric.
        """
        mu = self.predict(theta, step)
        k = self._gains(mu)
        r = neyman_residual(k[None, :] * mu, self.data, self.q_floor)

        if refresh or self._J is None:
            self._J = self._jacobian(theta, jnp.log(k), step)
        J = self._J
        n = self.n_configs
        g = jnp.einsum('gnp,gn->p', J, r) / n
        H = jnp.einsum('gnp,gnq->pq', J, J) / n
        loss = float(jnp.sum(r ** 2) / n)
        return g, H, loss

    def nuisance(self, theta, step=0):
        """The profiled nuisance at ``theta``: ``{'gains': (n_sensors,)}``."""
        return {'gains': self.gains(theta, step)}


@functools.partial(jax.jit, static_argnums=(1, 2))
def _fold_grid(key, n, m):
    """``out[i, j] = fold_in(fold_in(key, i), j)``, ``(n, m)`` keys in one dispatch."""
    fold = jax.vmap(jax.random.fold_in, in_axes=(None, 0))
    return jax.vmap(lambda k: fold(k, jnp.arange(m)))(fold(key, jnp.arange(n)))


def _group_by_structure(sources):
    """-> [(indices, stacked_pytree_or_None)], sources grouped by PYTREE STRUCTURE.

    Groups appear in order of first appearance and indices are ascending within a group, so a
    homogeneous tail groups as one `sources[1:]` stack and the published layout resolves to
    [([0], None), ([1..7], stack)].

    Grouping by structure rather than by type is what admits an arbitrary layout: two sources can
    be mapped together precisely when they can be stacked, which is a property of their pytree,
    not of their class.
    """
    groups = []                       # [(treedef, [indices])]
    for i, src in enumerate(sources):
        td = jax.tree_util.tree_structure(src)
        for g_td, ix in groups:
            if g_td == td:
                ix.append(i)
                break
        else:
            groups.append((td, [i]))
    out = []
    for _, ix in groups:
        stack = (jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[sources[i] for i in ix])
                 if len(ix) > 1 else None)
        out.append((ix, stack))
    return out


def _serial_map(fn, in_axes, n):
    """Serial stand-in for a mapped call: same per-item computation, results stacked.

    Numerically identical to a ``pmap``/``vmap`` over the leading axis for this use, and the only
    path available when fewer devices are present than sources.
    """
    jf = jax.jit(fn)

    def serial(*args):
        outs = []
        for i in range(n):
            a = [jax.tree_util.tree_map(lambda x: x[i], arg) if ax == 0 else arg
                 for arg, ax in zip(args, in_axes)]
            outs.append(jf(*a))
        return jnp.stack(outs)

    return serial
