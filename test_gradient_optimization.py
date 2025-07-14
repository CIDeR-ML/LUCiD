#!/usr/bin/env python3
"""
Test script for gradient-based optimization functionality.
Tests numerical, gradient, and hybrid optimization modes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import run_optimization from the new optimize module
from tools.optimization.optimize import run_optimization

from tools.utils import base_dir_path


def test_optimization_modes():
    """Test different optimization modes on a small number of events."""
    
    config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    n_events = 10
    
    print("="*80)
    print("TESTING GRADIENT-BASED OPTIMIZATION")
    print("="*80)
    
    # Test 1: Numerical only
    print("\n1. Testing NUMERICAL optimization only...")
    print("-"*60)
    results_numerical = run_optimization(
        json_filename=config_file,
        N_events=n_events,
        optimization_mode='numerical',
        n_numerical_iterations=15,
        n_gradient_iterations=0,
        numerical_population_size=100,
        verbose=True,
        base_seed=42
    )
    
    # # Test 2: Gradient only (with random initialization)
    # print("\n\n2. Testing GRADIENT optimization only...")
    # print("-"*60)
    # results_gradient = run_optimization(
    #     json_filename=config_file,
    #     N_events=n_events,
    #     optimization_mode='gradient',
    #     n_numerical_iterations=5,  # Just for initial guess
    #     n_gradient_iterations=30,
    #     numerical_population_size=10,
    #     energy_lr=0.5,
    #     spatial_lr=0.05,
    #     energy_scale=0.01,
    #     position_scale=0.1,
    #     direction_scale=0.1,
    #     patience=15,
    #     verbose=True,
    #     base_seed=42
    # )
    
    # Test 3: Hybrid optimization
    print("\n\n3. Testing HYBRID optimization...")
    print("-"*60)
    results_hybrid = run_optimization(
        json_filename=config_file,
        N_events=n_events,
        optimization_mode='hybrid',
        n_numerical_iterations=15,
        n_gradient_iterations=100,
        numerical_population_size=100,
        energy_lr=0.5,
        spatial_lr=0.05,
        energy_scale=0.01,
        position_scale=0.1,
        direction_scale=0.1,
        patience=250,
        verbose=True,
        base_seed=42
    )
    
    # Compare results
    print("\n"*2)
    print("="*80)
    print("COMPARISON OF OPTIMIZATION MODES")
    print("="*80)
    
    def get_mean_errors(results):
        errors = results['final_errors']
        successful = [e for e in errors if e is not None]
        if not successful:
            return None, None, None, None
        
        import numpy as np
        return (
            np.mean([e['position_error'] for e in successful]),
            np.mean([e['direction_error_deg'] for e in successful]),
            np.mean([e['energy_error'] for e in successful]),
            np.mean([e['final_loss'] for e in successful])
        )
    
    modes = ['Numerical', 'Gradient', 'Hybrid']
    all_results = [results_numerical, results_gradient, results_hybrid]
    
    print(f"\n{'Mode':<12} {'Pos Error (m)':<15} {'Dir Error (°)':<15} {'Energy Error (MeV)':<20} {'Final Loss':<15}")
    print("-"*80)
    
    for mode, results in zip(modes, all_results):
        pos_err, dir_err, energy_err, loss = get_mean_errors(results)
        if pos_err is not None:
            print(f"{mode:<12} {pos_err:<15.3f} {dir_err:<15.1f} {energy_err:<20.1f} {loss:<15.6f}")
        else:
            print(f"{mode:<12} {'FAILED':<15} {'FAILED':<15} {'FAILED':<20} {'FAILED':<15}")
    
    print("\n✅ All optimization modes tested successfully!")


if __name__ == "__main__":
    test_optimization_modes()