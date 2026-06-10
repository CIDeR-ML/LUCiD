"""Bridge: DetectorParams + setup_event_simulator → a Gauss-Newton/Fisher problem.

Turns a calibration setup (a sim callable ``sim(source, detector_params, key)`` plus a
truth ``DetectorParams`` and a list of sources) into the ``SourceModel`` forwards,
initial ``theta`` vector, and truth observables the generic fitter/Fisher consume.

The GLOBAL parameters being fitted are a chosen subset of DetectorParams LEAF fields
(scalars or λ-curve arrays); they are raveled into a single LOG-space vector ``theta``
(all calibration globals are positive). The per-PMT QE/gain factor is the multiplicative
Schur block ``k`` — handled by the fitter, not part of ``theta`` — so the forwards set
``qe_corrections=1`` and the truth bakes in the truth ``k``.
"""

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting.gauss_newton import SourceModel


def _ravel_fields(flat, fields):
    """Concatenate the named leaf values (flattened) into one vector; return (vec, shapes)."""
    parts, shapes = [], []
    for f in fields:
        v = np.asarray(flat[f], dtype=np.float64).ravel()
        shapes.append((f, np.asarray(flat[f]).shape))
        parts.append(v)
    return np.concatenate(parts) if parts else np.zeros(0), shapes


def _unravel_fields(vec, shapes):
    """Inverse of ``_ravel_fields``: split ``vec`` back into {field: array}."""
    out, i = {}, 0
    for f, shp in shapes:
        size = int(np.prod(shp)) if shp else 1
        out[f] = jnp.asarray(vec[i:i + size]).reshape(shp)
        i += size
    return out


def build_calibration_problem(sim, sources, dp_true, trainable_fields,
                              *, per_pmt_field='qe_corrections', truth_k=None,
                              key=None, num_sensors=None, eps=1e-8):
    """Build the fitter inputs for a charge calibration.

    Parameters
    ----------
    sim : callable
        Calibration simulator ``sim(source, detector_params, key) -> (charges, times)``
        (from ``setup_event_simulator(..., is_calibration=True)``; ``hit_mode='aggregated'``).
    sources : list
        Calibration source objects passed to ``sim``.
    dp_true : DetectorParams
        Truth detector parameters.
    trainable_fields : list[str]
        Flat leaf names of the GLOBAL parameters to fit (e.g. 'scatter_length',
        'absorption_length', 'mie_scatter_length', 'g', 'wall_reflection_rate',
        'sensor_reflection_rate', 'qe', or λ-curve arrays like 'abs_dev').
    per_pmt_field : str
        The per-PMT multiplicative factor (default 'qe_corrections'); the truth bakes
        in ``truth_k`` and the forwards use 1.
    truth_k : array (num_sensors,) or None
        Truth per-PMT factor (defaults ones).

    Returns
    -------
    dict with: ``source_models`` (list[SourceModel]), ``theta0`` (log truth globals),
    ``theta_true`` (== theta0), ``truth_charge`` (list per source), ``unravel`` (vec→dp),
    ``shapes``, ``lk_true`` (log truth_k), ``num_sensors``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    flat_true = {k: np.asarray(v) for k, v in _flatten_detector_params(dp_true).items()}
    ns = num_sensors if num_sensors is not None else int(np.asarray(flat_true[per_pmt_field]).shape[0])
    if truth_k is None:
        truth_k = np.ones(ns)

    vec_true, shapes = _ravel_fields(flat_true, trainable_fields)
    theta0 = np.log(np.clip(vec_true, 1e-12, None))

    flat_true_j = {k: jnp.asarray(v) for k, v in flat_true.items()}

    def unravel(theta_log, k_value=1.0):
        """Build a DetectorParams from the log-global vector; per-PMT field set to k_value.
        Fully JAX-traceable (called inside the jitted forward)."""
        overrides = _unravel_fields(jnp.exp(jnp.asarray(theta_log)), shapes)
        flat = dict(flat_true_j)
        flat.update(overrides)
        flat[per_pmt_field] = jnp.asarray(k_value) * jnp.ones(ns)
        return _nest_flat_kwargs(flat)

    # Truth observables: charge at the truth globals with the truth per-PMT k baked in.
    flat_k = {k: jnp.asarray(v) for k, v in flat_true.items()}
    flat_k[per_pmt_field] = jnp.asarray(truth_k)
    dp_truth_k = _nest_flat_kwargs(flat_k)
    truth_charge = [np.array(sim(src, dp_truth_k, key)[0]) for src in sources]

    # Forwards: per-sensor MEAN charge at k=1 for each source (ek used as the sim key).
    def make_forward(src):
        def forward(theta_log, ek, pk):
            return sim(src, unravel(theta_log, 1.0), ek)[0]
        return forward

    # eps is the √-residual offset: the default 1e-8 is the plain variance-stabilizing √-MSE;
    # eps=3/8 is the ANSCOMBE transform √(x+3/8), which removes the Poisson Jensen bias
    # (E[√Q]²≈μ−¼) that otherwise collapses the amplitude globals ~−1/(4μ) on low-occupancy
    # shot-noise data. Both the model m and the truth √ must use the same offset.
    source_models = [SourceModel(make_forward(src), eps=eps) for src in sources]

    return dict(source_models=source_models, theta0=theta0, theta_true=theta0,
                truth_charge=truth_charge, unravel=unravel, shapes=shapes,
                lk_true=np.log(np.clip(truth_k, 1e-6, None)), num_sensors=ns,
                trainable_fields=list(trainable_fields))
