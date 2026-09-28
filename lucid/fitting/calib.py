"""Calibration forward model: predicted per-sensor charge for every source × wavelength.

The calibration counterpart of :class:`lucid.fitting.recon.ReconModel`.

Preserved expression for expression from the paper's reference calibration engine. The whole
stack runs in float32 (``jax_enable_x64`` is never enabled), so an equivalent but re-associated
expression can move the result. The reference engine is not in this repository, so no shipped
test gates that agreement. Anything here that looks gratuitously specific (the key arithmetic,
the concatenate-then-transpose, the laser/isotropic split) is load-bearing for that reason.

Layout
------
``S = W * n_sources`` configurations, indexed ``g = wl * n_sources + s`` — wavelength-major, so
each row of the returned ``(S, NS)`` array is one (wavelength, source) pair.

Two dispatch routes
-------------------
Sources reach the simulator two ways, and the difference is not stylistic. The reference passes
each source as a **pytree of arrays**, so one compiled program serves all of them with the source
as a traced argument — that is what makes a ``pmap`` over sources possible at all. The field-based
bridge (:func:`lucid.fitting.problem.build_calibration_problem`) instead hands over one **closure
per source**, which cannot be traced and therefore gets its own compiled program.

Both routes run the same key arithmetic, the same assembly, and the same estimator downstream.
Only the compilation strategy differs, but in float32 that does not make the answers identical:
the forward agrees exactly, while the Jacobian differs at float32 epsilon because XLA folds a
closed-over source as a constant where the other route carries it as a traced argument. Gated by
``tests/test_fitting_calibrate.py::test_the_two_dispatch_routes_agree``. The traced route is the
default; the per-source route is selected by passing ``predict``.

Sharding
--------
The reference dispatches source 0 (a collimated laser) to one device and the remaining isotropic
sources to a ``pmap``. That is an execution strategy, not physics, so it is injected via
``map_fn`` rather than baked in. The default is a serial loop over sources, which is also what
the reference itself falls back to below 7 devices.
"""
import jax
import jax.numpy as jnp

__all__ = ['CalibrationForward', 'CalibrationJacobian', 'CalibrationProblem',
           'profile_gains', 'neyman_residual']


class CalibrationForward:
    """Mean per-sensor charge at fit vector ``theta``, for every source × wavelength.

    Parameters
    ----------
    sim : callable
        ``sim(source, detector_params, key) -> (charge, time)``, from
        ``setup_event_simulator(..., is_calibration=True, hit_mode='aggregated')``.
    sources : sequence
        Calibration sources. Source 0 is treated as the "singleton" (the laser in the reference
        layout) and the rest as the mappable group — this only affects dispatch, not the result.
    params : parameterisation, or None
        Supplies ``W`` and ``to_dp`` (:mod:`lucid.fitting.params`). May be ``None`` only when
        ``predict`` is given, in which case ``to_dp`` is never called and ``W`` defaults to 1.
    n_sensors : int
    map_fn : callable or None
        ``map_fn(fn, in_axes)`` returning a batched callable over the non-singleton sources.
        ``None`` (default) uses a serial loop that is numerically identical.
    predict : callable or None
        ``predict(theta, source, wl, gains, key) -> (n_sensors,)``, replacing the default
        composition ``sim(source, params.to_dp(theta, wl, gains), key)[0]``. Giving it selects the
        per-source route (see the module docstring): sources are then opaque Python objects rather
        than traced pytrees, so each gets its own compiled program and ``map_fn`` does not apply.
        This is how a list of ready-made forward closures is fitted.
    """

    def __init__(self, sim, sources, params, n_sensors, map_fn=None, predict=None,
                 laser_device=None):
        self.sim = sim
        self.sources = list(sources)
        self.params = params
        self.NS = int(n_sensors)
        self.W = 1 if params is None else params.W
        self.n_sources = len(self.sources)
        self.S = self.W * self.n_sources
        self._per_source = None

        self.map_fn = map_fn
        if predict is not None:
            self._per_source = [self._compile_one(predict, src) for src in self.sources]
            return

        def _mbody(theta, src, keys, gains):                      # (W, NS); vmap over wavelengths
            return jax.vmap(lambda wl, k: sim(src, params.to_dp(theta, wl, gains), k)[0])(
                jnp.arange(self.W), keys)

        self._mbody = _mbody
        self._single = jax.jit(_mbody)
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
        self._group_fns, self._group_stacks = [], []
        for idx, stack in self._groups:
            if len(idx) == 1:
                self._group_fns.append(None)            # singleton: use self._single
                self._group_stacks.append(None)
                continue
            self._group_fns.append(self._make_map(_mbody, (None, 0, 0, None), len(idx)))
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

    def _compile_one(self, predict, src):
        """One compiled ``(theta, keys, gains) -> (W, NS)`` program with ``src`` closed over."""
        W = self.W

        def body(theta, keys, gains):
            return jax.vmap(lambda wl, k: predict(theta, src, wl, gains, k))(jnp.arange(W), keys)
        return jax.jit(body)

    def keys(self, key_base):
        """The reference's key arithmetic, reproduced exactly.

        Singleton source: ``PRNGKey(kb + wl)``. Grouped source ``s`` (1-based):
        ``PRNGKey(kb + 1000*s + wl)``. Both are pure functions of the base and the indices — no
        global PRNG state — which is what makes a run reproducible from its seed alone.
        """
        single = jnp.stack([jax.random.PRNGKey(key_base + wl) for wl in range(self.W)])
        if self.n_sources == 1:
            return single, None          # nothing to group; jnp.stack([]) would raise
        group = jnp.stack([jnp.stack([jax.random.PRNGKey(key_base + 1000 * s + wl)
                                      for wl in range(self.W)])
                           for s in range(1, self.n_sources)])
        return single, group

    def __call__(self, theta, key_base, gains):
        """-> ``(S, NS)`` mean charge, row ``g = wl * n_sources + s``."""
        ks, kg = self.keys(key_base)
        if self._per_source is not None:
            per = jnp.stack([self._per_source[0](theta, ks, gains)]
                            + [self._per_source[s](theta, kg[s - 1], gains)
                               for s in range(1, self.n_sources)])          # (n, W, NS)
            return jnp.transpose(per, (1, 0, 2)).reshape(self.S, self.NS)
        # Every source's keys, indexed by its ABSOLUTE position, so a source's random stream does
        # not depend on how the layout happens to be grouped. `ks` is source 0's and `kg[s-1]` is
        # source s's, which is the reference's arithmetic (`kb + 1000*s + wl`, and `kb + wl` at
        # s=0) written once.
        allk = [ks] + [kg[s - 1] for s in range(1, self.n_sources)]

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
            gk = jnp.stack([allk[i] for i in idx])
            out = fn(theta, stack, gk, gains)                # (len(idx), W, NS)
            for j, i in enumerate(idx):
                slots[i] = out[j]
        per = jnp.stack([jnp.asarray(v) for v in slots])      # (n, W, NS), SOURCE order
        return jnp.transpose(per, (1, 0, 2)).reshape(self.S, self.NS)

    def average(self, theta, key_base, gains, n_draws):
        """Mean over ``n_draws`` independent forward draws.

        The draw offset is ``131*b``, matching the reference.

        Averaging reduces the forward's Monte-Carlo variance. Were the residual linear in ``M``,
        ``E[r(M)] = r(E[M])`` and the fixed point would sit at truth at any variance. What makes
        ``n_draws`` matter is the profiled gain ``k = ΣQ/ΣM``: it is nonlinear in ``M``, so its
        expectation shifts with the forward's variance, and that shift propagates into the
        residual. The bias and its scaling with forward noise are measured in
        ``tests/test_fitting_estimator_unbiased.py``.
        """
        acc = jnp.zeros((self.S, self.NS))
        for b in range(n_draws):
            acc = acc + self(theta, key_base + 131 * b, gains)
        return acc / n_draws


class CalibrationJacobian:
    """``∂r/∂θ`` for the Neyman residual — differentiated as ONE fused expression.

    It does not compute ``∂μ/∂θ`` and then apply the ``1/√Q`` weight; it runs ``jacfwd`` over the
    *already weighted* model::

        sm(θ) = exp(lk) · sim(src, dp(θ), key) / √clip(Q, floor)
        J     = jacfwd(sm)(θ)

    Splitting that into a model Jacobian followed by a scaling is algebraically the same and
    **numerically different in float32**, which is the only precision this stack ever runs in.
    So the weight stays inside the differentiated function, and the data enter the Jacobian
    rather than only the residual.

    Two further properties are deliberate, not incidental:

    * ``gains`` are passed as ones and the per-PMT factor enters as the constant ``exp(lk)``
      multiplier, so ``∂k/∂θ`` is dropped by construction — the gains are profiled, not fitted.
    * ``lk`` is computed from the *residual's* draw, so J and r share that one scalar per sensor
      even though their photon streams are disjoint. It vanishes exactly when the gains are held
      fixed. The magnitude of the residual coupling has not been measured; do not assume it small.

    The key stream is independent of the residual's and carries the seed (see :meth:`key`).
    """

    def __init__(self, sim, sources, params, n_sensors, key0=9_000_000, map_fn=None,
                 predict=None, laser_device=None):
        self.sim = sim
        self.sources = list(sources)
        self.params = params
        self.NS = int(n_sensors)
        self.W = 1 if params is None else params.W
        self.n_sources = len(self.sources)
        self.S = self.W * self.n_sources
        self.key0 = int(key0)
        self._per_source = None

        if predict is not None:
            # Per-source route (see the module docstring). The fused expression is rebuilt around
            # `predict` rather than around `sim . to_dp` — same weighted model, same jacfwd.
            def compile_one(src):
                def body(theta, wl, lk, key, q_cfg, q_floor):
                    def sm(th):
                        mu = predict(th, src, wl, jnp.ones(self.NS), key)
                        return jnp.exp(lk) * mu / jnp.sqrt(jnp.clip(q_cfg, q_floor, None))
                    return jax.jacfwd(sm)(theta)
                return jax.jit(body, static_argnums=(1,))
            self._per_source = [compile_one(src) for src in self.sources]
            return

        def _jbody(theta, src, wl, lk, key, q_cfg, q_floor):
            def sm(th):
                mu = sim(src, params.to_dp(th, wl, jnp.ones(self.NS)), key)[0]
                return jnp.exp(lk) * mu / jnp.sqrt(jnp.clip(q_cfg, q_floor, None))
            return jax.jacfwd(sm)(theta)

        self._jbody = _jbody
        self._single = jax.jit(_jbody)
        # Same placement as CalibrationForward: an un-placed `_single` runs on device 0, which
        # `map_fn` already uses for shard 0, so the laser column would serialise behind an
        # isotropic one. The reference engine places the Jacobian's laser inputs the same way.
        # Value-neutral: placement changes where arithmetic happens, not what it produces.
        self._laser_dev = laser_device
        if self._laser_dev is None and self.n_sources > 1:
            devs = jax.devices()
            self._laser_dev = devs[min(self.n_sources - 1, len(devs) - 1)]
        # Same arbitrary-layout grouping as CalibrationForward, and it must be the SAME grouping:
        # the Jacobian's rows are assembled in source order to match the forward's, so a
        # different partition here would silently pair row `g` with another source's column.
        self.map_fn = map_fn
        self._groups = _group_by_structure(self.sources)
        self._placed = next((ix[0] for ix, _ in self._groups if len(ix) == 1), None)
        self._group_fns, self._group_stacks = [], []
        AX = (None, 0, None, None, 0, 0, None)
        for idx, stack in self._groups:
            if len(idx) == 1:
                self._group_fns.append(None)
                self._group_stacks.append(None)
                continue
            self._group_fns.append(map_fn(_jbody, AX) if map_fn is not None
                                   else _serial_map(_jbody, AX, len(idx)))
            self._group_stacks.append(stack)

    def key(self, step, source, wl, draw):
        """``key0 + 7·step + 1000·source + wl + 50000·draw`` — the reference's arithmetic.

        ``key0`` should carry the seed (``9_000_000 + 1_234_567·seed``). Without a seed term the
        whole ensemble draws the SAME Jacobian noise, so any fixed point it displaces moves every
        member alike: the error appears as bias and contributes nothing to the spread, leaving the
        ensemble s.e.m. blind to it.
        """
        return jax.random.PRNGKey(self.key0 + 7 * step + 1000 * source + wl + 50000 * draw)

    def __call__(self, theta, lk, step, data, q_floor, draws=(0,)):
        """-> ``(S, NS, P)``, rows ordered ``g = wl·n_sources + s`` to match the forward.

        ``draws`` is the set of draw indices to average over, not merely a count, so that
        disjoint halves can be requested independently. That matters for the unbiased-Hessian
        variant: ``JᵀJ`` built from a single noisy draw is biased upward by the noise Gram
        (the same error enters both factors), and the cure is the cross product of two
        independent halves, whose expectation is exactly ``J̄ᵀJ̄``.
        """
        draws = tuple(draws)
        nh = len(draws)
        rows = []
        if self._per_source is not None:
            for wl in range(self.W):
                base = wl * self.n_sources
                rows.append(jnp.stack([
                    sum(self._per_source[s](theta, wl, lk, self.key(step, s, wl, h),
                                            data[base + s], q_floor) for h in draws) / nh
                    for s in range(self.n_sources)]))
            return jnp.concatenate(rows, axis=0)
        for wl in range(self.W):
            base = wl * self.n_sources
            slots = [None] * self.n_sources
            for (idx, _), fn, stack in zip(self._groups, self._group_fns, self._group_stacks):
                if fn is None:                                   # singleton
                    i = idx[0]
                    d = self._laser_dev if i == self._placed else None
                    if d is None:
                        slots[i] = sum(self._single(theta, self.sources[i], wl, lk,
                                                    self.key(step, i, wl, h), data[base + i],
                                                    q_floor) for h in draws) / nh
                    else:
                        th_l, lk_l = jax.device_put(theta, d), jax.device_put(lk, d)
                        src_l = jax.device_put(self.sources[i], d)
                        q_l = jax.device_put(data[base + i], d)
                        slots[i] = sum(self._single(
                            th_l, src_l, wl, lk_l,
                            jax.device_put(self.key(step, i, wl, h), d), q_l, q_floor)
                            for h in draws) / nh
                    continue
                out = sum(fn(theta, stack, wl, lk,
                             jnp.stack([self.key(step, i, wl, h) for i in idx]),
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
    deliberate choice, documented at :class:`CalibrationJacobian`) omits a genuinely non-zero
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
    """A calibration fit, in the shape :func:`lucid.fitting.gn.gauss_newton` consumes.

    Least squares: a residual and a Jacobian are formed, then assembled into ``g = Jᵀr`` and
    ``H = JᵀJ``. Reconstruction reaches the same ``(g, H, loss)`` by a different route — AD of a
    scalar likelihood, with a Fisher metric built separately — which is why the loop is written
    against ``(g, H, loss)`` and not against residuals.

    Three things this class owns that the loop must not:

    * **The keys.** The residual's forward is redrawn every step from one stream; the Jacobian
      draws from another, which must be independent of it (shared keys correlate J with r) and
      must carry the seed (see :meth:`CalibrationJacobian.key`).
    * **The nuisance.** Per-PMT gains are profiled in closed form each step, so they never enter
      the optimizer. This is why calibration needs no Schur block in the fitter, and therefore why
      one loop can serve both problems at all.
    * **Step application.** ``accumulate`` keeps the iterate in the dtype the reference uses. The
      stack is float32 throughout, and a step applied in float64 gives a different trajectory.
    """

    def __init__(self, forward, jacobian, params, data, q_floor, *,
                 gauge='log', n_forward_draws=1, jacobian_draws=2,
                 forward_key0=1000, forward_key_stride=13):
        self.forward = forward
        self.jacobian = jacobian
        self.params = params
        self.data = data                                   # (S, NS) observed charge
        self.data_sum = data.sum(0)
        self.q_floor = float(q_floor)
        self.gauge = gauge
        self.n_forward_draws = int(n_forward_draws)
        self.jacobian_draws = int(jacobian_draws)
        self.forward_key0 = int(forward_key0)
        self.forward_key_stride = int(forward_key_stride)
        self.n_configs = int(data.shape[0])
        self._J = None                                     # cached between refreshes

    def forward_key(self, step):
        return self.forward_key0 + self.forward_key_stride * step

    def accumulate(self, theta, dtheta):
        """Apply a step, preserving the reference's float32 iterate (see the class docstring)."""
        return theta + jnp.asarray(dtheta)

    def grad_metric_loss(self, theta, step, refresh=True):
        ones = jnp.ones(self.data.shape[1])
        mu = self.forward.average(theta, self.forward_key(step), ones, self.n_forward_draws)
        k = profile_gains(mu.sum(0) + 1e-12, self.data_sum, gauge=self.gauge)
        r = neyman_residual(k[None, :] * mu, self.data, self.q_floor)

        if refresh or self._J is None:
            self._J = self.jacobian(theta, jnp.log(k), step, self.data, self.q_floor,
                                    draws=range(self.jacobian_draws))
        J = self._J
        n = self.n_configs
        g = jnp.einsum('gnp,gn->p', J, r) / n
        H = jnp.einsum('gnp,gnq->pq', J, J) / n
        loss = float(jnp.sum(r ** 2) / n)
        return g, H, loss

    def gains(self, theta, step=0):
        """The profiled per-PMT gains at ``theta`` — the nuisance, read out rather than fitted.

        A detector calibration wants these: they are the per-sensor QE/gain map. They never enter
        ``theta``, so a caller that only reads the fit vector would never see them.
        """
        ones = jnp.ones(self.data.shape[1])
        mu = self.forward.average(theta, self.forward_key(step), ones, self.n_forward_draws)
        return profile_gains(mu.sum(0) + 1e-12, self.data_sum, gauge=self.gauge)


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
