#!/usr/bin/env python3
"""
Simple test for gradient optimization with minimal iterations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from tools.utils import base_dir_path
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.optimization.algorithms import adaptive_search, hybrid_optimization
from tools.utils import generate_random_params


def test_gradient_simple():
    """Test gradient optimization with 1 event and minimal iterations."""
    
    config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    
    print("="*60)
    print("SIMPLE GRADIENT OPTIMIZATION TEST")
    print("="*60)
    
    # Setup
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    
    # Detector bounds
    detector_bounds = {
        'type': 'cylinder',
        'r': detector.r,
        'H': detector.H
    }
    
    # Sensor parameters
    sensor_params = (
        jnp.array(4.),
        jnp.array(0.2),
        jnp.array(6.),
        jnp.array(0.001)
    )
    
    # Setup simulator
    simulate_event = setup_event_simulator(
        config_file, 
        n_photons=100_000,  # Reduced for speed
        temperature=0.05, 
        K=2
    )
    
    # Generate test event
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    true_energy, true_position, true_direction_angles = generate_random_params(subkey)
    true_theta, true_phi = true_direction_angles
    true_direction = jnp.array([
        jnp.sin(true_theta) * jnp.cos(true_phi),
        jnp.sin(true_theta) * jnp.sin(true_phi),
        jnp.cos(true_theta)
    ])
    
    print(f"\nTrue parameters:")
    print(f"  Energy: {float(true_energy):.2f} MeV")
    print(f"  Position: [{float(true_position[0]):.3f}, {float(true_position[1]):.3f}, {float(true_position[2]):.3f}]")
    print(f"  Direction: θ={float(true_theta)*180/jnp.pi:.1f}°, φ={float(true_phi)*180/jnp.pi:.1f}°")
    
    # Simulate event
    true_params = (true_energy, true_position, true_direction_angles)
    true_charges, true_times = simulate_event(true_params, sensor_params, subkey)
    print(f"  Active sensors: {jnp.sum(true_charges > 0)}")
    
    # Test 1: Numerical only (very few iterations)
    print("\n" + "-"*60)
    print("Test 1: Numerical optimization (5 iterations)")
    print("-"*60)
    
    key, subkey = jax.random.split(key)
    result_num = adaptive_search(
        true_charges, true_times, simulate_event, sensor_params, sensor_positions,
        detector_bounds, true_position, true_direction, true_energy,
        n_iterations=5,
        population_size=10,
        random_seed=int(subkey[0]),
        verbose=True,
        optimization_type='numerical'
    )
    best_num, _ = result_num
    
    if best_num:
        pos_err = float(jnp.linalg.norm(best_num['position'] - true_position))
        dir_err = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(best_num['direction'], true_direction)), 0, 1)))
        energy_err = float(jnp.abs(best_num['energy'] - true_energy))
        
        print(f"\nNumerical results:")
        print(f"  Loss: {best_num['loss']:.6f}")
        print(f"  Position error: {pos_err:.3f} m")
        print(f"  Direction error: {jnp.degrees(dir_err):.1f}°")
        print(f"  Energy error: {energy_err:.1f} MeV")
    
    # Test 2: Hybrid optimization
    print("\n" + "-"*60)
    print("Test 2: Hybrid optimization (3 numerical + 10 gradient)")
    print("-"*60)
    
    gradient_kwargs = {
        'energy_lr': 0.5,
        'spatial_lr': 0.05,
        'energy_scale': 0.01,
        'position_scale': 0.1,
        'direction_scale': 0.1,
        'patience': 5,
        'patience_factor': 0.5
    }
    
    key, subkey = jax.random.split(key)
    result_hybrid = adaptive_search(
        true_charges, true_times, simulate_event, sensor_params, sensor_positions,
        detector_bounds, true_position, true_direction, true_energy,
        n_iterations=3,
        population_size=10,
        random_seed=int(subkey[0]),
        verbose=True,
        optimization_type='hybrid',
        gradient_iterations=10,
        gradient_kwargs=gradient_kwargs
    )
    best_hybrid, _ = result_hybrid
    
    if best_hybrid:
        pos_err = float(jnp.linalg.norm(best_hybrid['position'] - true_position))
        dir_err = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(best_hybrid['direction'], true_direction)), 0, 1)))
        energy_err = float(jnp.abs(best_hybrid['energy'] - true_energy))
        
        print(f"\nHybrid results:")
        print(f"  Loss: {best_hybrid['loss']:.6f}")
        print(f"  Position error: {pos_err:.3f} m")
        print(f"  Direction error: {jnp.degrees(dir_err):.1f}°")
        print(f"  Energy error: {energy_err:.1f} MeV")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_gradient_simple()