"""Bridge: a truth ``DetectorParams`` + a simulator -> the inputs a fit or a CRB consumes.

Names a set of ``DetectorParams`` leaf fields as the free parameters and builds everything the
rest of the package needs from them: the parameterisation, one forward per source, the truth
observables, and the starting vector.

Two consumers:

* :func:`lucid.fitting.calibrate.fit` uses the forwards and the truth charge, fitting them with
  the Neyman residual and profiled gains, on the shared Gauss-Newton loop.
* :func:`lucid.fitting.crb` uses the same forwards wrapped as ``SourceModel``, because the bound
  is a property of the observable, not the estimator: it differentiates ``sqrt(k*M)`` and
  marginalises the per-PMT block by Schur complement instead of profiling it.

The parameterisation is :class:`lucid.fitting.params.FieldParams`: log space, unnamed fields held
at truth, and the per-PMT factor kept out of ``theta`` because there is one per sensor and
profiling it costs a division where fitting it would cost the fit.
"""

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting.schur_gn import SourceModel
from lucid.fitting.params import FieldParams


def build_calibration_problem(sim, sources, dp_true, trainable_fields,
                              *, per_pmt_field='qe_corrections', truth_k=None,
                              key=None, num_sensors=None, eps=1e-8):
    """Build the fitter and CRB inputs for a charge calibration.

    Parameters
    ----------
    sim : callable
        Calibration simulator ``sim(source, detector_params, key) -> (charges, times)``
        (from ``setup_event_simulator(..., is_calibration=True)``; ``hit_mode='aggregated'``).
    sources : list
        Calibration source objects passed to ``sim``.
    dp_true : DetectorParams
        Truth detector parameters. Everything not named in ``trainable_fields`` is held here.
    trainable_fields : list[str]
        Flat leaf names of the free parameters (e.g. 'scatter_length', 'absorption_length',
        'mie_scatter_length', 'g', 'wall_reflection_rate', 'sensor_reflection_rate', 'qe', or
        λ-curve arrays like 'abs_dev').
    per_pmt_field : str
        The per-PMT multiplicative factor (default 'qe_corrections'). The truth bakes in
        ``truth_k``; the forwards use 1, and the fit profiles it.
    truth_k : array (num_sensors,) or None
        Truth per-PMT factor (defaults ones).
    eps : float
        The ``sqrt``-residual offset carried by ``SourceModel``, and therefore by the CRB, whose
        Jacobian is of ``sqrt(k*M)``. It does not reach the fit, which uses the Neyman residual
        with its weight floored by ``q_floor``.

    Returns
    -------
    dict with: ``params`` (the :class:`~lucid.fitting.params.FieldParams`), ``source_models``
    (list[SourceModel], for the CRB), ``theta0`` / ``theta_true`` (log truth globals),
    ``truth_charge`` (list per source), ``unravel`` (vec -> DetectorParams), ``shapes``,
    ``lk_true`` (log truth_k), ``num_sensors``, ``trainable_fields``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    flat_true = {k: np.asarray(v) for k, v in _flatten_detector_params(dp_true).items()}
    ns = num_sensors if num_sensors is not None else int(
        np.asarray(flat_true[per_pmt_field]).shape[0])
    if truth_k is None:
        truth_k = np.ones(ns)

    params = FieldParams(dp_true, trainable_fields, per_pmt_field=per_pmt_field, n_sensors=ns)
    theta0 = params.theta_from_physical()

    def unravel(theta_log, k_value=1.0):
        """DetectorParams from the log-global vector, per-PMT field set to ``k_value``.

        Returned so analysis scripts can inspect a fitted point. Equivalent to
        ``params.to_dp`` with the wavelength index (ignored by ``FieldParams``) set to 0.
        """
        return params.to_dp(jnp.asarray(theta_log), 0, jnp.asarray(k_value))

    # Truth observables: charge at the truth globals with the truth per-PMT k baked in.
    flat_k = {k: jnp.asarray(v) for k, v in flat_true.items()}
    flat_k[per_pmt_field] = jnp.asarray(truth_k)
    dp_truth_k = _nest_flat_kwargs(flat_k)
    truth_charge = [np.array(sim(src, dp_truth_k, key)[0]) for src in sources]

    def make_forward(src):
        def forward(theta_log, ek, pk):
            return sim(src, unravel(theta_log, 1.0), ek)[0]
        return forward

    source_models = [SourceModel(make_forward(src), eps=eps) for src in sources]

    return dict(params=params, source_models=source_models, theta0=theta0, theta_true=theta0,
                truth_charge=truth_charge, unravel=unravel, shapes=params.shapes,
                lk_true=np.log(np.clip(truth_k, 1e-6, None)), num_sensors=ns,
                trainable_fields=list(trainable_fields))
