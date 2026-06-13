"""Unified calibration / reconstruction fitting.

A single Gauss-Newton machine (consistent fixed-dataset, cached-Jacobian, constrained
per-PMT Schur, median ridge) + the matching Fisher/CRB at truth (×√12 honesty), plus a
bridge that turns a DetectorParams + setup_event_simulator calibration setup into the
fitter's inputs.

    from lucid.fitting import build_calibration_problem, fit, crb
    prob = build_calibration_problem(sim, sources, dp_true, ['scatter_length', ...])
    res  = fit(prob['source_models'], prob['truth_charge'], prob['theta0'], prob['num_sensors'])
    cov  = crb(prob['source_models'], prob['theta_true'], prob['num_sensors'])
"""
from lucid.fitting.gauss_newton import (
    fit, SourceModel, sqrt_residual, make_constrained_schur, ridge_inverse,
    fit_charge_time, ChargeTimeModel,
)
from lucid.fitting.fisher import crb, SQRT12
from lucid.fitting.problem import build_calibration_problem
from lucid.fitting.timing import calibrate_timing
from lucid.fitting.recon import (
    ReconModel, fit_track, fit_track_multistart, track_from_vec9, vec9_from_track, vec9_dir,
    SCALE9, seed_vertex_time,
)
from lucid.fitting.contracts import CalibForward, PerPhotonPredictor
