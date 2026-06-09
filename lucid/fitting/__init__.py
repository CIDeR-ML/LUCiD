"""Unified calibration / reconstruction fitting.

A single Gauss-Newton machine (consistent fixed-dataset, cached-Jacobian, constrained
per-PMT Schur, median ridge) + the matching Fisher/CRB at truth (×√12 honesty). A
calibration is specified by *partitioning* a DetectorParams — you name the leaves you
fit and the fitter reads routing/space/gauge from their structure (no role table):

    from lucid.fitting import calibrate
    res = calibrate(sim, sources, dp_true,
                    train=['scatter_length', 'absorption_length', 'qe_corrections'])
    # res['dp_hat'] is the recovered DetectorParams; res['k'] the per-PMT factor.

Lower-level: ``build_problem`` returns the fitter inputs; ``partition``/``combine`` are
the structural split/merge; ``fit``/``crb`` are the generic optimiser + Fisher bound.
"""
from lucid.fitting.gauss_newton import (
    fit, SourceModel, sqrt_residual, make_constrained_schur, ridge_inverse,
)
from lucid.fitting.fisher import crb, SQRT12
from lucid.fitting.partition import partition, combine, classify, trained_leaves
from lucid.fitting.problem import build_problem, calibrate
from lucid.fitting.timing import calibrate_timing
