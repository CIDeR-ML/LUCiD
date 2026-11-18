"""Fitting: calibrate a detector, or reconstruct an event, on one Gauss-Newton loop.

The two problems look nothing alike from outside — calibration solves for a handful of optical
constants from thousands of sensors lit by known sources; reconstruction solves for one particle's
nine numbers from a single event. They meet at a narrow seam. A problem supplies

    grad_metric_loss(theta, step, refresh) -> (gradient, metric, loss)

and :func:`lucid.fitting.gn.gauss_newton` knows nothing else about it. That is deliberately all
they share: calibration is least squares and assembles ``JᵀJ``; reconstruction is a likelihood and
builds a Fisher metric separately, with no residual vector at all. An interface written in terms
of residuals could serve only one of them.

Calibrating a detector
----------------------
One model, one problem, any solver. Run the closure first — data generated from parameters you
chose, so a poor recovery is a fact about the configuration and not about the data::

    from lucid.fitting import (FieldParams, CalibrationForward, CalibrationProblem, calibrate,
                               closure_data, crb)
    params = FieldParams(dp, ['scatter_length', 'absorption_length', 'qe'])
    model = CalibrationForward(sim, sources, params)
    theta_true = params.theta_from_physical()
    data = closure_data(model, theta_true, k_true)       # or the observed (S, n_sensors) charge
    res = calibrate(CalibrationProblem(model, data), theta_true + 0.2)
    res['real'], res['gains']                            # parameters, and the per-PMT gain map

``calibrate(problem, theta0, tx=optax.adam(2e-2))`` runs another optimiser on the same problem.
:class:`FieldParams` fits any named ``DetectorParams`` leaves; :class:`CalibrationParams` is the
published estimand, per-wavelength optics plus shared reflection.

Reconstructing an event
-----------------------
:func:`reconstruct` runs the published method on a :class:`ReconModel`: data-driven seeds
(:func:`seed_track`, recipe :data:`SEED_RECIPE`), a pre-fit pick, then one :func:`fit_track`
(recipe :data:`FIT_RECIPE`; the model's is :data:`MODEL_RECIPE`). Call :func:`fit_track`
directly to fit from a start of your own.

Uncertainty
-----------
``crb(model, theta)`` is the Cramér-Rao bound at a point, from the model alone. It answers a
different question than the fit does — what the observable could support, rather than what one
estimator returned — and it marginalises the per-PMT gains by Schur complement rather than
profiling them, for that reason.
"""

from lucid.fitting import report
from lucid.fitting.analysis import (
    angular_error_deg,
    bootstrap,
    bootstrap_ci,
    resolution_stats,
    track_errors,
    vertex_residual,
)
from lucid.fitting.calib import (
    CalibrationForward,
    CalibrationProblem,
    neyman_residual,
    profile_gains,
)
from lucid.fitting.calibrate import calibrate, closure_data
from lucid.fitting.contracts import CalibForward, PerPhotonPredictor
from lucid.fitting.fisher import SQRT12, crb, make_constrained_schur
from lucid.fitting.gn import gauss_newton, gauss_newton_tx
from lucid.fitting.minimize import minimize
from lucid.fitting.params import CalibrationParams, FieldParams
from lucid.fitting.recon import (
    FIT_RECIPE,
    MODEL_RECIPE,
    SCALE9,
    FitRecipe,
    ModelRecipe,
    ReconModel,
    ReconProblem,
    fit_track,
    track_from_vec9,
    vec9_dir,
    vec9_from_track,
)
from lucid.fitting.schur_gn import ChargeTimeModel, fit_charge_time, ridge_inverse
from lucid.fitting.seed import (
    SEED_RECIPE,
    SeedRecipe,
    fuse_seeds,
    pick_by_margin,
    reconstruct,
    seed_track,
    seed_vertex_time,
)
from lucid.fitting.timing import calibrate_timing

# The optax transformations are re-exported; `damped_matrix` deliberately is not. It is the single
# implementation of the damping convention, so a fix lands in one place; import it explicitly with
# `from lucid.fitting.transforms import damped_matrix`.
from lucid.fitting.transforms import (
    annealed_learning_rate,
    damped_gauss_newton,
    scale_by_damped_gauss_newton,
    scale_by_driver_schedule,
)

__all__ = [
    # parameterisations
    "CalibrationParams",
    "FieldParams",
    # calibration: entry points, then the pieces
    "calibrate",
    "closure_data",
    "CalibrationForward",
    "CalibrationProblem",
    "profile_gains",
    "neyman_residual",
    # the shared optimiser: the damped Gauss-Newton configuration, its optax pieces, the driver
    "gauss_newton",
    "gauss_newton_tx",
    "damped_gauss_newton",
    "scale_by_damped_gauss_newton",
    "minimize",
    # reconstruction
    "ReconModel",
    "ReconProblem",
    "fit_track",
    "FitRecipe",
    "FIT_RECIPE",
    "ModelRecipe",
    "MODEL_RECIPE",
    "reconstruct",
    "seed_track",
    "SeedRecipe",
    "SEED_RECIPE",
    "seed_vertex_time",
    "fuse_seeds",
    "pick_by_margin",
    "track_from_vec9",
    "vec9_from_track",
    "vec9_dir",
    "SCALE9",
    # uncertainty
    "crb",
    "make_constrained_schur",
    "SQRT12",
    # not yet consolidated: the joint charge+time fit
    "ridge_inverse",
    "fit_charge_time",
    "ChargeTimeModel",
    "calibrate_timing",
    # reporting and contracts
    "report",
    "CalibForward",
    "PerPhotonPredictor",
    "bootstrap",
    "bootstrap_ci",
    "resolution_stats",
    "vertex_residual",
    "angular_error_deg",
    "track_errors",
]
