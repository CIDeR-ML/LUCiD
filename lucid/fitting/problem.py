"""Bridge: a DetectorParams + setup_event_simulator calibration → the GN/Fisher inputs.

You name the leaves you fit; everything else is read from structure (see
:mod:`lucid.fitting.partition`). ``calibrate`` partitions ``dp_true`` into trained/frozen
pytrees, builds the per-source forwards that ``combine`` the optimiser vector back into a
full DetectorParams, and returns the inputs the generic
:func:`lucid.fitting.gauss_newton.fit` and :func:`lucid.fitting.fisher.crb` consume.

Global leaves (scalars + λ-curves) become the dense GN block, each optimised in its
natural space (log for positive-bounded, linear for signed); the single per-PMT leaf
(e.g. ``qe_corrections``) is the multiplicative Schur block ``k`` handled inside ``fit``.

    res = calibrate(sim, sources, dp_true,
                    train=['scatter_length', 'absorption_length', 'qe_corrections'])
"""

import numpy as np
import jax
import jax.numpy as jnp

from lucid.fitting.gauss_newton import SourceModel, fit as _gn_fit
from lucid.fitting.partition import partition, combine, classify
from lucid.detector_params import _FIELD_TO_SUBTUPLE as _F2S


def _sub(leaf):
    """Leaf name → owning sub-tuple attribute (e.g. 'scatter_length' → 'scattering')."""
    return _F2S[leaf]


def _to_opt(value, space):
    a = jnp.asarray(value)
    return jnp.log(jnp.clip(a, 1e-12, None)) if space == 'log' else a


def _from_opt(vec, space):
    return jnp.exp(vec) if space == 'log' else vec


def build_problem(sim, sources, dp_true, train, *, truth_k=None, key=None,
                  n_sensors=None):
    """Partition ``dp_true`` by ``train`` and assemble the fitter inputs.

    Parameters
    ----------
    sim : callable ``sim(source, detector_params, key) -> (charges, times)``.
    sources : list of calibration source objects.
    dp_true : DetectorParams truth.
    train : iterable[str]
        Leaf names / dotted paths to fit. Scalars & ``(N_CTRL,)`` curves form the global
        block; the ONE ``(NS,)`` leaf (if any) is the per-PMT Schur factor.
    truth_k : (NS,) or None — truth per-PMT factor baked into the observables (default 1).

    Returns
    -------
    dict with ``source_models``, ``theta0``/``theta_true`` (optimiser-space globals),
    ``truth_charge`` (per source), ``unravel`` (vec[,k] → DetectorParams), ``spec``
    (the :func:`classify` map), ``lk_true``, ``n_sensors``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    theta_tree, fixed_tree = partition(dp_true, train)
    spec = classify(theta_tree, n_sensors)
    ns = spec['n_sensors']
    glob = spec['globals']
    ppmt = spec['per_pmt']
    if len(ppmt) > 1:
        raise ValueError(f"charge calibration supports one per-PMT Schur leaf, got "
                         f"{[n for n, _, _ in ppmt]}")
    pp_name = ppmt[0][0] if ppmt else None

    # global optimiser vector (per-leaf space), in classify order
    layout, parts = [], []
    for n, sh, sp in glob:
        val = getattr(getattr(theta_tree, _sub(n)), n)
        parts.append(np.asarray(_to_opt(val, sp)).ravel())
        layout.append((n, int(np.prod(sh)) if sh else 1, sp, sh))
    theta0 = np.concatenate(parts) if parts else np.zeros(0)

    truth_k = np.ones(ns) if truth_k is None else np.asarray(truth_k, float)

    def unravel(theta_vec, k_value=1.0):
        """Optimiser vector (+ per-PMT factor) → full DetectorParams (JAX-traceable)."""
        theta_vec = jnp.asarray(theta_vec)
        updates, i = {}, 0
        for n, sz, sp, sh in layout:
            updates[n] = _from_opt(theta_vec[i:i + sz], sp).reshape(sh)
            i += sz
        th = theta_tree
        for n, val in updates.items():
            sub = _sub(n)
            th = th._replace(**{sub: getattr(th, sub)._replace(**{n: val})})
        if pp_name is not None:
            sub = _sub(pp_name)
            th = th._replace(**{sub: getattr(th, sub)._replace(
                **{pp_name: jnp.asarray(k_value) * jnp.ones(ns)})})
        return combine(th, fixed_tree)

    # truth observables: globals at truth, per-PMT factor = truth_k
    dp_truth = unravel(theta0, 1.0)
    if pp_name is not None:
        sub = _sub(pp_name)
        dp_truth = dp_truth._replace(**{sub: getattr(dp_truth, sub)._replace(
            **{pp_name: jnp.asarray(truth_k)})})
    truth_charge = [np.array(sim(src, dp_truth, key)[0]) for src in sources]

    def make_forward(src):
        def forward(theta_vec, ek, pk):
            return sim(src, unravel(theta_vec, 1.0), ek)[0]
        return forward
    source_models = [SourceModel(make_forward(src)) for src in sources]

    return dict(source_models=source_models, theta0=theta0, theta_true=theta0.copy(),
                truth_charge=truth_charge, unravel=unravel, spec=spec,
                lk_true=np.log(np.clip(truth_k, 1e-6, None)), n_sensors=ns,
                pp_name=pp_name, layout=layout)


def calibrate(sim, sources, dp_true, train, *, truth_k=None, key=None,
              n_sensors=None, **fit_kw):
    """End-to-end: :func:`build_problem` then run the constrained-Schur GN fit.

    Extra keyword args are forwarded to :func:`lucid.fitting.gauss_newton.fit`.
    Returns the ``fit`` result dict augmented with the ``problem`` and a ``dp_hat``
    DetectorParams reconstructed at the fitted optimiser vector + per-PMT factor.
    """
    prob = build_problem(sim, sources, dp_true, train, truth_k=truth_k, key=key,
                         n_sensors=n_sensors)
    res = _gn_fit(prob['source_models'], prob['truth_charge'], prob['theta0'],
                  prob['n_sensors'], **fit_kw)
    res['problem'] = prob
    # dp_hat carries the fitted per-PMT factor k (the recovered per-sensor correction),
    # which the forward applied on top of qe_corrections=1.
    res['dp_hat'] = prob['unravel'](res['log_theta'] if 'log_theta' in res else res['theta'],
                                    res.get('k', 1.0))
    return res
