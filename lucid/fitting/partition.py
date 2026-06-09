"""Per-case calibration setup by *partitioning* a DetectorParams — no role table.

A calibration is specified by listing which DetectorParams LEAVES you fit. You do not
build a separate parameter container or a name→behaviour table; you split the real
``DetectorParams`` pytree into a trained part and a frozen part, and the *structure of
the trained leaves* carries everything the fitter needs:

    theta, fixed = partition(dp, ['scatter_length', 'absorption_length', 'qe_corrections'])

``partition`` returns two ``DetectorParams`` of identical structure: trained leaves live
in ``theta`` (the rest are ``None``), frozen leaves live in ``fixed`` (trained ones
``None``). The forward merges them back structurally — *no path lookup table*:

    dp = combine(theta, fixed)            #  jax.tree_map(pick, …, is_leaf = x is None)

The ``is_leaf = (x is None)`` trick is the one correct way to make ``None`` a sentinel
leaf so the two complementary pytrees share a treedef (otherwise JAX treats ``None`` as
an *empty subtree* and the structures don't line up). ``ravel_pytree`` drops the ``None``
slots automatically, so the optimiser vector is exactly the trained leaves.

The fitter then routes each trained leaf by its **shape**, and reads its **space** from
the parameter bounds — both structural, no per-name dispatch:

==================  =================================  ===========================
trained leaf is…    routing                            why
==================  =================================  ===========================
scalar              one global column (dense FD)       single global value
shape ``(N_CTRL,)`` ``N_CTRL`` global columns          λ-deviation curve
shape ``(NS,)``     per-PMT diagonal → Schur block     one multiplicative k per PMT
positive-bounded    optimise in **log**                lengths, rates, qe, devs
signed-bounded      optimise in **linear**             t0, walk
non-scalar leaf     gauge its natural-space mean to 0  removes the rank-1 degeneracy
==================  =================================  ===========================

The single gauge rule (non-scalar ⇒ mean-zero in natural space) unifies what used to be
two special cases: ``mean(log k)=0`` for per-PMT k and ``geomean(dev)=1`` for a λ-curve
are the *same* removal of the overall scale that is degenerate with the global it
multiplies. A scalar has no such partner, so it is never gauged.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from lucid.detector_params import (
    DetectorParams, _SUBTUPLES, _FIELD_TO_SUBTUPLE, _flatten_detector_params,
    default_bounds,
)

_IS_NONE = lambda x: x is None


def _leaf(path):
    """Accept ``'scattering.scatter_length'`` or bare ``'scatter_length'`` → leaf name."""
    name = path.split('.')[-1]
    if name not in _FIELD_TO_SUBTUPLE:
        raise KeyError(f"unknown DetectorParams leaf {path!r}; "
                       f"known leaves: {sorted(_FIELD_TO_SUBTUPLE)}")
    return name


def partition(dp, train):
    """Split ``dp`` into ``(theta, fixed)`` DetectorParams by trained leaf names.

    ``train`` is an iterable of leaf names (``'scatter_length'``) or dotted paths
    (``'scattering.scatter_length'``). In ``theta`` the trained leaves keep their value
    and every other leaf is ``None``; in ``fixed`` it is the mirror image. ``combine``
    is their structural inverse.
    """
    trained = {_leaf(p) for p in train}
    theta_subs, fixed_subs = {}, {}
    for attr, cls in _SUBTUPLES:
        sub = getattr(dp, attr)
        tvals, fvals = {}, {}
        for f in cls._fields:
            v = getattr(sub, f)
            tvals[f], fvals[f] = (v, None) if f in trained else (None, v)
        theta_subs[attr] = cls(**tvals)
        fixed_subs[attr] = cls(**fvals)
    return DetectorParams(**theta_subs), DetectorParams(**fixed_subs)


def combine(theta, fixed):
    """Structural inverse of :func:`partition`: pick the non-``None`` leaf at each slot."""
    return jax.tree_util.tree_map(lambda t, f: f if t is None else t,
                                  theta, fixed, is_leaf=_IS_NONE)


def trained_leaves(theta):
    """Ordered ``[(name, value)]`` for the non-``None`` leaves of a ``theta`` partition."""
    out = []
    for attr, cls in _SUBTUPLES:
        sub = getattr(theta, attr)
        for f in cls._fields:
            v = getattr(sub, f)
            if v is not None:
                out.append((f, v))
    return out


def _space(name, ns=1):
    """``'linear'`` if the leaf's lower bound is negative (signed), else ``'log'``."""
    bmin, _ = default_bounds(ns)
    return 'linear' if float(np.min(np.asarray(_flatten_detector_params(bmin)[name]))) < 0.0 else 'log'


def classify(theta, n_sensors=None):
    """Read the routing/space/gauge of every trained leaf from its structure alone.

    Returns a dict with:
      ``globals`` : ``[(name, shape, space)]`` scalar & curve leaves (the dense FD block),
      ``per_pmt`` : ``[(name, shape, space)]`` shape-``(NS,)`` leaves (Schur blocks),
      ``n_sensors`` : inferred sensor count (or the passed value),
      ``gauge`` : ``{name: 'mean'|'geomean'}`` for every non-scalar trained leaf.
    """
    leaves = trained_leaves(theta)
    # infer NS from the largest 1-D leaf if not given
    ns = n_sensors
    if ns is None:
        sizes = [int(np.asarray(v).shape[0]) for _, v in leaves if np.asarray(v).ndim == 1]
        ns = max(sizes) if sizes else 1
    glob, ppmt, gauge = [], [], {}
    for name, v in leaves:
        v = np.asarray(v)
        sp = _space(name, ns)
        if v.ndim == 1 and v.shape[0] == ns:
            ppmt.append((name, v.shape, sp))
            gauge[name] = 'mean' if sp == 'linear' else 'geomean'
        else:
            glob.append((name, v.shape, sp))
            if v.ndim >= 1 and v.size > 1:                 # a curve: gauge the overall scale
                gauge[name] = 'mean' if sp == 'linear' else 'geomean'
    return dict(globals=glob, per_pmt=ppmt, n_sensors=ns, gauge=gauge)
