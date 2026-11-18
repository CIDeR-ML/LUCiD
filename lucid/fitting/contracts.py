"""Executable contracts for the ``lucid.fitting`` closure surfaces (MAIN_BRANCH_PLAN B5).

The fitter is closure/duck-typed by design (no config dataclasses). The two opaque callables
below are the contracts that otherwise live only in docstrings and drift. Typing them as
``Protocol`` makes them grep/pyright/IDE-checkable at ZERO runtime cost — import and annotate,
nothing executes.

Entry points (all in ``lucid.fitting``):
  * calibration : ``calibrate(CalibrationProblem(CalibrationForward(sim, sources, params), data),
                  theta0)`` -> fitted parameters + the profiled per-PMT gain map; the data may be
                  ``closure_data(...)`` you generated. The fit PROFILES the per-PMT gains in
                  closed form. ``crb(model, theta)`` -> covariance at a point (×√12 honest);
                  it MARGINALISES the gains by Schur complement instead, because a bound must
                  integrate over a nuisance where a fit may profile it.
  * recon       : ``reconstruct(model, oc, ot, seed=, fit=, seed_key=, fit_key=)``
                  (seeds and fit from the event's own keys, pick, one fit) or
                  ``fit_track(model, oc, ot, start, fit, key=)`` from a start of your own -> the
                  9-vector ``[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]``.

The internal GN metric convention (what ``ChargeTimeModel``/``ReconModel`` feed
the assembler): each block is ``(r, J, W)`` with residual ``r`` (m,), Jacobian ``J`` (m, p), and
weight ``W`` either a per-row vector (lit-mask | 1/μ) or ``None``≡identity; the normal matrix is
``H = Σ (J·W)ᵀ J`` and gradient ``g = Σ (J·W)ᵀ r``. ``W`` is PRIVATE to each model's residual
form (√-MSE vs Poisson) — the GN loop never sees the form.

Recon recipe knobs (the validated Fisher-GN recipe, ``FIT_RECIPE``): ``nkeys=8, steps=150,
lr=4.0, lr_final=1.5, mu=0.1, lam=0.01, refresh=8``, Polyak readout, SCALE9-preconditioned,
AMP_DETACH in the time term. Calibration runs the same loop
(:func:`lucid.fitting.minimize.minimize`, by default with :func:`lucid.fitting.gn.gauss_newton_tx`
at ``lam=0.01, mu=0.1``, ``max_step=0.5``) and a Neyman residual.
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
    are clean).
    """

    def __call__(self, theta, ek, pk) -> "Array": ...   # -> (n_sensors,)


@runtime_checkable
class PerPhotonPredictor(Protocol):
    """Recon per-photon track predictor: ``setup_event_simulator(..., source='particle')``.

    Returns the four per-photon arrays the recon loss consumes:
      ``log_w``        (n_photons,)  per-photon log survival weight (soft, temperature>0),
      ``flat_times``   (n_photons,)  predicted arrival time of each photon,
      ``flat_indices`` (n_photons,)  destination PMT index of each photon,
      ``total_charge`` (n_sensors,)  per-PMT expected charge μ (Σ weights).
    Wrapped by :class:`~lucid.fitting.recon.ReconModel` into the per-PMT (μ, time-NLL).
    """

    def __call__(self, track: "ParticleParams", key) -> Tuple["Array", "Array", "Array", "Array"]: ...
