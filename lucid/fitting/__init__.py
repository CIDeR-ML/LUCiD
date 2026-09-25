"""Fitting: calibrate a detector, or reconstruct an event, on one Gauss-Newton loop.

The two problems look nothing alike from outside — calibration solves for a handful of optical
constants from thousands of sensors lit by known sources; reconstruction solves for one particle's
nine numbers from a single event. They meet at a narrow seam. A problem supplies

    grad_metric_loss(theta, step, refresh) -> (gradient, metric, loss)
    accumulate(theta, dtheta) -> theta

and :func:`lucid.fitting.gn.gauss_newton` knows nothing else about it. That is deliberately all
they share: calibration is least squares and assembles ``JᵀJ``; reconstruction is a likelihood and
builds a Fisher metric separately, with no residual vector at all. An interface written in terms
of residuals could serve only one of them.

Calibrating a detector
----------------------
Run the closure first — data generated from parameters you chose, so a poor recovery is a fact
about the configuration and not about the data::

    from lucid.fitting import CalibrationParams, closure, calibrate
    par = CalibrationParams(n_wavelengths=3)
    res = closure(sim, sources, par, theta_true, gains=k_true, perturbation=0.4)
    print(res['frac_error'])

Then the same fit against real data::

    res = calibrate(sim, sources, par, observed_charge, theta0)
    res['real'], res['gains']         # parameters, and the per-PMT gain map

:class:`CalibrationParams` is the published estimand: per-wavelength optics plus shared
reflection. To fit something else, name ``DetectorParams`` leaves with :class:`FieldParams`, or go
through :func:`build_calibration_problem` and :func:`fit`, which do that for you.

Reconstructing an event
-----------------------
:func:`fit_track` on a :class:`ReconModel`, or :func:`fit_track_multistart` when the basin is in
doubt; :func:`seed_vertex_time` supplies a starting point by time multilateration.
:class:`ProjectedReconProblem` is the variant that projects the time term off the measured
vertex-``t0`` degeneracy, so time-term noise cannot slide the fit along a ray it cannot resolve.

Uncertainty
-----------
:func:`crb` is the Cramér-Rao bound at truth. It answers a different question than the fit does —
what the observable could support, rather than what one estimator returned — and it marginalises
the per-PMT gains by Schur complement rather than profiling them, for that reason.
"""
from lucid.fitting.params import CalibrationParams, FieldParams, LogParams
from lucid.fitting.calib import (
    CalibrationForward, CalibrationJacobian, CalibrationProblem,
    profile_gains, neyman_residual,
)
from lucid.fitting.gn import gauss_newton
from lucid.fitting.calibrate import calibrate, closure, closure_data, fit
from lucid.fitting.schur_gn import (
    SourceModel, sqrt_residual, make_constrained_schur, ridge_inverse,
    fit_charge_time, ChargeTimeModel,
)
from lucid.fitting.fisher import crb, SQRT12
from lucid.fitting.problem import build_calibration_problem
from lucid.fitting.timing import calibrate_timing
from lucid.fitting.recon import (
    ReconModel, ReconProblem, ProjectedReconProblem, fit_track, fit_track_multistart,
    track_from_vec9, vec9_from_track, vec9_dir, SCALE9, seed_vertex_time,
    fuse_seeds, pick_by_margin, DEFAULT_RECIPE,
)
from lucid.fitting import report
from lucid.fitting.contracts import CalibForward, PerPhotonPredictor
from lucid.fitting.analysis import (
    bootstrap_ci, resolution_stats, vertex_residual, angular_error_deg,
)
# The TRANSFORMATIONS are re-exported; `damped_matrix` is not, and now there is only one of it.
# `gn.damped_matrix` was a second, numpy implementation of the same convention, kept so the
# delegation could be pinned bit-exactly against the loop it replaced. It had no production caller
# once `gauss_newton` moved onto the optax step, and a convention with two implementations is one
# that has to be fixed twice -- which it was, earlier in this work. Reach the surviving one
# explicitly: `from lucid.fitting.transforms import damped_matrix`.
from lucid.fitting.transforms import (damped_gauss_newton, scale_by_damped_gauss_newton,
                                      scale_by_driver_schedule, annealed_learning_rate)
from lucid.fitting.minimize import minimize
from lucid.fitting.scaled import ScaledProblem

__all__ = [
    # parameterisations
    'CalibrationParams', 'FieldParams', 'LogParams',
    # calibration: entry points, then the pieces
    'calibrate', 'closure', 'closure_data', 'fit', 'build_calibration_problem',
    'CalibrationForward', 'CalibrationJacobian', 'CalibrationProblem',
    'profile_gains', 'neyman_residual',
    # the shared optimiser: the damped Gauss-Newton configuration, its optax pieces, the driver
    'gauss_newton',
    'damped_gauss_newton', 'scale_by_damped_gauss_newton', 'minimize', 'ScaledProblem',
    # reconstruction
    'ReconModel', 'ReconProblem', 'ProjectedReconProblem', 'fit_track', 'DEFAULT_RECIPE',
    'fit_track_multistart', 'seed_vertex_time',
    'track_from_vec9', 'vec9_from_track', 'vec9_dir', 'SCALE9',
    'fuse_seeds', 'pick_by_margin',
    # uncertainty
    'crb', 'SQRT12',
    # not yet consolidated: the CRB source model and the joint charge+time fit
    'SourceModel', 'sqrt_residual', 'make_constrained_schur', 'ridge_inverse',
    'fit_charge_time', 'ChargeTimeModel', 'calibrate_timing',
    # reporting and contracts
    'report', 'CalibForward', 'PerPhotonPredictor',
    'bootstrap_ci', 'resolution_stats', 'vertex_residual', 'angular_error_deg',
]
