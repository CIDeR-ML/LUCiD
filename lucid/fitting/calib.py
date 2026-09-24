"""Calibration forward model: predicted per-sensor charge for every source × wavelength.

This is the piece calibration was missing. Reconstruction's forward has been a library object
(:class:`lucid.fitting.recon.ReconModel`) since the start; calibration's lived inside a figure
script, which is why nothing importable could calibrate a detector.

Extracted from the reference engine (``analysis/paper/utils/calib_fit.py``) and preserved
expression for expression. The whole stack runs in float32 — ``jax_enable_x64`` is never enabled
— so an equivalent but re-associated expression can move the result, and the engine is pinned
bit-exactly by ``tests/reconciliation/test_calib_engine_pin.py``. Anything here that looks
gratuitously specific (the key arithmetic, the concatenate-then-transpose, the laser/isotropic
split) is load-bearing for that reason.

Layout
------
``S = W * n_sources`` configurations, indexed ``g = wl * n_sources + s`` — wavelength-major, so
each row of the returned ``(S, NS)`` array is one (wavelength, source) pair.

Sharding
--------
The reference dispatches source 0 (a collimated laser) to one device and the remaining isotropic
sources to a ``pmap``, because the campaign ran on a 10-GPU node. That is an execution strategy,
not physics, so it is injected via ``map_fn`` rather than baked in. The default is a serial loop
over sources — which is also what the reference itself falls back to below 7 devices, and
therefore the path the pin actually gates.
"""
import jax
import jax.numpy as jnp

__all__ = ['CalibrationForward']


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
    params : CalibrationParams
        The theta -> DetectorParams map (:mod:`lucid.fitting.params`).
    n_sensors : int
    map_fn : callable or None
        ``map_fn(fn, in_axes)`` returning a batched callable over the non-singleton sources.
        ``None`` (default) uses a serial loop that is numerically identical.
    """

    def __init__(self, sim, sources, params, n_sensors, map_fn=None):
        self.sim = sim
        self.sources = list(sources)
        self.params = params
        self.NS = int(n_sensors)
        self.W = params.W
        self.n_sources = len(self.sources)
        self.S = self.W * self.n_sources

        def _mbody(theta, src, keys, gains):                      # (W, NS); vmap over wavelengths
            return jax.vmap(lambda wl, k: sim(src, params.to_dp(theta, wl, gains), k)[0])(
                jnp.arange(self.W), keys)

        self._mbody = _mbody
        self._single = jax.jit(_mbody)
        self._group = (map_fn(_mbody, (None, 0, 0, None)) if map_fn is not None
                       else _serial_map(_mbody, (None, 0, 0, None), self.n_sources - 1))
        # The grouped sources are stacked once into a single pytree, as the reference does, so the
        # mapped call sees one batched argument rather than a Python list.
        self._group_stack = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *self.sources[1:]) if self.n_sources > 1 else None

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
        ml = self._single(theta, self.sources[0], ks, gains)               # (W, NS)
        if self.n_sources == 1:
            per = ml[None]
        else:
            mi = self._group(theta, self._group_stack, kg, gains)          # (n-1, W, NS)
            per = jnp.concatenate([jnp.asarray(ml)[None], mi], axis=0)     # (n, W, NS)
        return jnp.transpose(per, (1, 0, 2)).reshape(self.S, self.NS)

    def average(self, theta, key_base, gains, n_draws):
        """Mean over ``n_draws`` independent forward draws.

        The draw offset is ``131*b``, matching the reference. Averaging reduces the Monte-Carlo
        variance of the forward, which is what a residual linear in the model needs in order for
        its fixed point to sit at truth.
        """
        acc = jnp.zeros((self.S, self.NS))
        for b in range(n_draws):
            acc = acc + self(theta, key_base + 131 * b, gains)
        return acc / n_draws


class CalibrationJacobian:
    """``∂r/∂θ`` for the Neyman residual — differentiated as ONE fused expression.

    This is the subtlety that decides whether an extraction is correct. The reference does not
    compute ``∂μ/∂θ`` and then apply the ``1/√Q`` weight; it runs ``jacfwd`` over the *already
    weighted* model::

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

    The key stream is independent of the residual's by construction — sharing them was measured
    at 137σ of covariance — and carries the seed, so an ensemble can see its own spread.
    """

    def __init__(self, sim, sources, params, n_sensors, key0=9_000_000, map_fn=None):
        self.sim = sim
        self.sources = list(sources)
        self.params = params
        self.NS = int(n_sensors)
        self.W = params.W
        self.n_sources = len(self.sources)
        self.S = self.W * self.n_sources
        self.key0 = int(key0)

        def _jbody(theta, src, wl, lk, key, q_cfg, q_floor):
            def sm(th):
                mu = sim(src, params.to_dp(th, wl, jnp.ones(self.NS)), key)[0]
                return jnp.exp(lk) * mu / jnp.sqrt(jnp.clip(q_cfg, q_floor, None))
            return jax.jacfwd(sm)(theta)

        self._jbody = _jbody
        self._single = jax.jit(_jbody)
        self._group = (map_fn(_jbody, (None, 0, None, None, 0, 0, None)) if map_fn is not None
                       else _serial_map(_jbody, (None, 0, None, None, 0, 0, None),
                                        self.n_sources - 1))
        self._group_stack = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *self.sources[1:]) if self.n_sources > 1 else None

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
        for wl in range(self.W):
            base = wl * self.n_sources
            jl = sum(self._single(theta, self.sources[0], wl, lk,
                                  self.key(step, 0, wl, h), data[base], q_floor)
                     for h in draws) / nh
            rows.append(jnp.asarray(jl)[None])
            if self.n_sources > 1:
                ji = sum(self._group(theta, self._group_stack, wl, lk,
                                     jnp.stack([self.key(step, s, wl, h)
                                                for s in range(1, self.n_sources)]),
                                     data[base + 1: base + self.n_sources], q_floor)
                         for h in draws) / nh
                rows.append(ji)
        return jnp.concatenate(rows, axis=0)


def profile_gains(model_charge, observed_sum, gauge='log', clip_min=1e-6):
    """Closed-form per-PMT gains ``k = ΣQ / ΣM``, gauged — the nuisance, solved not fitted.

    There is one gain per sensor (order 10^4 of them). Fitting them jointly would swamp the
    handful of optical parameters that are actually of interest, so they are profiled out at every
    step: given the current model they have an exact minimiser, and it costs one division.

    The gauge removes the exact degeneracy between a global gain and the overall light yield —
    without it the two directions are unidentifiable.

    ``'log'``    ``mean(log k) = 0``  — what the published run used.
    ``'linear'`` ``mean(k) = 1``      — ``k̂ = ΣQ/ΣM`` is linear in the data and therefore
                 unbiased, whereas taking its log first incurs a Jensen shift that is larger on
                 dim sensors. Measured on a minimal toy: log gauge +0.402%, linear +0.068%.
                 Not the published choice; adopting it would move published numbers.
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
      draws from another, which must be independent (sharing them was measured at 137σ of
      covariance) and must carry the seed (without it, every member of an ensemble draws the same
      Jacobian noise, so any fixed point it displaces moves them all alike — the error shows up as
      bias and contributes nothing to the spread).
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


def _serial_map(fn, in_axes, n):
    """Serial stand-in for a mapped call: same per-item computation, results stacked.

    Numerically identical to a ``pmap``/``vmap`` over the leading axis for this use, and the only
    path available when fewer devices are present than sources — which is the common case off the
    campaign's node, and the one the engine pin exercises.
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
