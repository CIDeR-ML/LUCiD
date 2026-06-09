"""Calibration / reconstruction fitting.

A fit is ``loss(params, key) -> scalar`` over a params pytree. The DEFAULT solver is the
canonical optax loop — reverse-mode ``grad`` over the whole pytree, no flatten/Schur/FD:

    from lucid.fitting import make_loss, fit, charge, gauge_mean_log
    loss   = make_loss(sim, sources, observations, terms=[charge])
    dp_hat = fit(loss, dp0, project=gauge_mean_log)

New observable → add a term; new parameter → a leaf in the pytree; reconstruction → pass a
track pytree; joint → pass ``(dp, track)``. The Gauss-Newton + Schur solver (``fit_gn``) and
the Fisher/CRB (``crb``) stay available as the opt-in advanced path on the same contract.
"""
# Default fitter + loss primitives (the simple, pytree-native path).
from lucid.fitting.optimize import fit, gauge_mean_log, gauge_mean
from lucid.fitting.loss import (
    make_loss, sqrt_mse, charge, charge_var, first_arrival,
)

# Advanced solver: Gauss-Newton + constrained per-PMT Schur + Fisher/CRB.
from lucid.fitting.gauss_newton import (
    fit as fit_gn, SourceModel, sqrt_residual, make_constrained_schur, ridge_inverse,
)
from lucid.fitting.fisher import crb, SQRT12
from lucid.fitting.problem import build_calibration_problem
from lucid.fitting.timing import calibrate_timing
