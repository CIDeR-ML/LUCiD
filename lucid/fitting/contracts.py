"""Executable contracts for the ``lucid.fitting`` closure surfaces (MAIN_BRANCH_PLAN B5).

The fitter is closure/duck-typed by design (no config dataclasses). The two opaque callables
below are the contracts that otherwise live only in docstrings and drift. Typing them as
``Protocol`` makes them grep/pyright/IDE-checkable at ZERO runtime cost — import and annotate,
nothing executes.

Entry points (all in ``lucid.fitting``):
  * calibration : ``calibrate(sim, sources, params, data, theta0)`` -> fitted parameters + the
                  profiled per-PMT gain map; ``closure(...)`` for the same fit against data you
                  generated. ``fit(sources, truth_list, theta0, n_sensors)`` is the field-bridge
                  form. The per-PMT gains are PROFILED in closed form, not carried as a Schur
                  nuisance — that changed when the estimator did.
                  ``crb(sources, theta_true, n_sensors)`` -> covariance at truth (×√12 honest),
                  and it DOES still marginalise the gains by Schur complement, because a bound
                  must integrate over a nuisance where a fit may profile it.
  * recon       : ``fit_track(model, oc, ot, start, **RECON_GN)`` or
                  ``fit_track_multistart(model, oc, ot, [seedA, seedB], margin=0.01)`` ->
                  the 9-vector ``[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]``.

The internal GN metric convention (what ``SourceModel``/``ChargeTimeModel``/``ReconModel`` feed
the assembler): each block is ``(r, J, W)`` with residual ``r`` (m,), Jacobian ``J`` (m, p), and
weight ``W`` either a per-row vector (lit-mask | 1/μ) or ``None``≡identity; the normal matrix is
``H = Σ (J·W)ᵀ J`` and gradient ``g = Σ (J·W)ᵀ r``. ``W`` is PRIVATE to each model's residual
form (√-MSE vs Poisson) — the GN loop never sees the form.

Recon recipe knobs (the validated Fisher-GN recipe, matching ``fit_track``'s current
defaults): ``nkeys=8, niters=150, lr=4.0, lr_final=1.5, ridge_i=0.1, lam=0.01, refresh=8,
readout='polyak'``, SCALE9-preconditioned, AMP_DETACH in the time term. Calibration runs the
same loop (:func:`lucid.fitting.gn.gauss_newton`) with ``lam=0.01, mu=0.1, max_step=0.5`` and a
Neyman residual.
"""
from __future__ import annotations
from typing import Protocol, Tuple, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:                                   # avoid importing JAX / params at module load
    from jax import Array
    from lucid.detector_params import ParticleParams


@runtime_checkable
class CalibForward(Protocol):
    """Calibration source forward: per-sensor MEAN charge at the per-PMT factor k=1.

    ``theta`` are the log-global params; ``ek``/``pk`` are the engine + photon forward-noise
    keys (the expected-value engine is deterministic GIVEN its keys, so CRN finite differences
    are clean). Wrapped by :class:`~lucid.fitting.schur_gn.SourceModel` into the √(k·M)
    residual + a CRN-FD Jacobian.
    """

    def __call__(self, theta, ek, pk) -> "Array": ...   # -> (n_sensors,)


@runtime_checkable
class PerPhotonPredictor(Protocol):
    """Recon per-photon track predictor: ``setup_event_simulator(..., hit_mode='per_photon')``.

    Returns the four per-photon arrays the recon loss consumes:
      ``log_w``        (n_photons,)  per-photon log survival weight (soft, temperature>0),
      ``flat_times``   (n_photons,)  predicted arrival time of each photon,
      ``flat_indices`` (n_photons,)  destination PMT index of each photon,
      ``total_charge`` (n_sensors,)  per-PMT expected charge μ (Σ weights).
    Wrapped by :class:`~lucid.fitting.recon.ReconModel` into the per-PMT (μ, time-NLL).
    """

    def __call__(self, track: "ParticleParams", key) -> Tuple["Array", "Array", "Array", "Array"]: ...
