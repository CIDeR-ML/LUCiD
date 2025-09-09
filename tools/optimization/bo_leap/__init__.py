"""
BO-LEAP: Bayesian Optimization with Local search via Evolution And gradients with Patience.

This module provides an advanced optimization algorithm that combines:
- Bayesian Optimization for global exploration
- CMA-ES for population-based local search  
- Gradient descent for refined local optimization
"""

from .bo_leap import (
    setup_and_run_bo_leap,
    bo_leap_optimize,
    bo_leap_local_search,
    make_bounded_functions,
    make_cyclic_aware_clip,
    clip_to_bounds,
    bounded_gradient_step,
    bounded_gradient_descent
)

from .gp_predict import gp_predict_next_point

__all__ = [
    # Main high-level interface
    'setup_and_run_bo_leap',
    
    # Lower-level optimization function
    'bo_leap_optimize',
    
    # Local search component
    'bo_leap_local_search',
    
    # Helper functions for custom implementations
    'make_bounded_functions',
    'make_cyclic_aware_clip', 
    'clip_to_bounds',
    'bounded_gradient_step',
    'bounded_gradient_descent',
    
    # GP prediction
    'gp_predict_next_point'
]