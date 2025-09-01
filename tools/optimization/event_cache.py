"""
Event generation and caching utilities for LUCiD optimization.

This module provides shared functions for generating and caching event databases
used by both the initial guess benchmark and the main optimization pipeline.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime

from ..utils import base_dir_path
from ..geometry import generate_detector
from ..simulation import setup_event_simulator


def get_cache_filepath(config_file, detector_type, K, cache_dir=None):
    """
    Get the cache file path based on detector configuration.
    
    Args:
        config_file: Path to detector configuration file
        detector_type: Type of detector (e.g., 'Cylinder')
        K: Number of nearest neighbors used in simulation
        cache_dir: Custom cache directory path (optional)
        
    Returns:
        tuple: (cache_dir, cache_filepath)
    """
    # Extract detector name from config file
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
    # Create cache directory
    if cache_dir is None:
        cache_dir = os.path.join(base_dir_path(), 'data', 'events_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename with detector name (not type)
    cache_filename = f'{detector_name}_events_K{K}.pkl'
    cache_filepath = os.path.join(cache_dir, cache_filename)
    
    return cache_dir, cache_filepath


def get_detector_bounds(detector):
    """Extract detector bounds based on detector type."""
    detector_type = detector.__class__.__name__
    
    if 'Cylinder' in detector_type:
        return {
            'type': 'cylinder',
            'r': detector.r,
            'H': detector.H
        }
    elif 'Sphere' in detector_type:
        return {
            'type': 'sphere',
            'r': detector.r
        }
    elif 'Box' in detector_type:
        return {
            'type': 'box',
            'x': detector.L,
            'y': detector.W,
            'z': detector.H
        }
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def generate_random_event_params(key, detector_bounds, fraction=0.7):
    """
    Generate random event parameters based on detector geometry.
    
    This is the same function from optimize.py, shared for consistency.
    """
    if detector_bounds['type'] == 'cylinder':
        r_vert = jax.random.uniform(key, shape=(), minval=0, maxval=detector_bounds['r'] * fraction)
        key, _ = jax.random.split(key)
        theta = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        key, _ = jax.random.split(key)
        z_vert = jax.random.uniform(key, shape=(), minval=-detector_bounds['H']/2 * fraction, 
                                   maxval=detector_bounds['H']/2 * fraction)
        position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
    elif detector_bounds['type'] == 'sphere':
        u = jax.random.uniform(key, shape=())
        key, _ = jax.random.split(key)
        cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
        key, _ = jax.random.split(key)
        phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        
        r = detector_bounds['r'] * fraction * jnp.cbrt(u)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        position = jnp.array([r * sin_theta * jnp.cos(phi), 
                             r * sin_theta * jnp.sin(phi), 
                             r * cos_theta])
        
    elif detector_bounds['type'] == 'box':
        position = jax.random.uniform(key, shape=(3,), 
                                    minval=jnp.array([-detector_bounds['x']/2, 
                                                     -detector_bounds['y']/2, 
                                                     -detector_bounds['z']/2]) * fraction,
                                    maxval=jnp.array([detector_bounds['x']/2, 
                                                     detector_bounds['y']/2, 
                                                     detector_bounds['z']/2]) * fraction)
    
    # Random direction
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
    
    # Random energy
    key, _ = jax.random.split(key)
    energy = jax.random.uniform(key, shape=(), minval=500.0, maxval=1500.0)
    
    return position, direction, energy


def generate_and_cache_events(config_file, detector_type, n_events=50000, n_photons=500000, 
                             K=6, seed=12345, verbose=True, cache_dir=None):
    """
    Generate or load cached event database.
    
    Args:
        config_file: Path to detector configuration file
        detector_type: Type of detector
        n_events: Number of events to generate (hardcoded to 50k)
        n_photons: Number of photons per event (hardcoded to 500k)
        K: Number of nearest neighbors
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        dict: Event database with charges, times, and metadata
    """
    # Hardcode event parameters as specified
    n_events = 50000
    n_photons = 1_000_000
    
    # Get cache filepath
    cache_dir, cache_filepath = get_cache_filepath(config_file, detector_type, K, cache_dir)
    
    # Check if cache exists
    if os.path.exists(cache_filepath):
        if verbose:
            print(f"Loading cached events from {cache_filepath}")
        
        with open(cache_filepath, 'rb') as f:
            events_data = pickle.load(f)
        
        # Validate cache
        if len(events_data['metadata']) != n_events:
            if verbose:
                print(f"Warning: Cache has {len(events_data['metadata'])} events, expected {n_events}")
                print("Regenerating cache...")
        else:
            if verbose:
                print(f"Loaded {n_events} cached events")
            return events_data
    
    # Generate new events
    if verbose:
        print(f"Generating {n_events} events with {n_photons} photons each...")
        print(f"Cache will be saved to: {cache_filepath}")
    
    # Setup detector
    detector = generate_detector(config_file)
    detector_bounds = get_detector_bounds(detector)
    sensor_positions = jnp.array(detector.all_points)
    
    # Setup simulation
    simulate_event = setup_event_simulator(
        json_filename=config_file,
        max_sensors_per_cell=4,
        n_photons=n_photons,
        temperature=0.05,
        K=K,
        detector_type=detector_type
    )
    
    # Sensor parameters
    sensor_params = (
        jnp.array(100.0),    # scatter_length
        jnp.array(0.1),      # reflection_rate
        jnp.array(100.0),    # absorption_length
        jnp.array(0.001)     # gumbel_softmax_temperature
    )
    
    # Generate events
    key = jax.random.PRNGKey(seed)
    all_charges = []
    all_times = []
    all_metadata = []
    
    # Progress bar
    event_iterator = tqdm(range(n_events), desc="Generating events") if verbose else range(n_events)
    
    for i in event_iterator:
        # Generate random event parameters
        event_key = jax.random.PRNGKey(seed + i)
        position, direction, energy = generate_random_event_params(event_key, detector_bounds)
        
        # Convert to simulation format
        theta = jnp.arccos(jnp.clip(direction[2], -1.0, 1.0))
        phi = jnp.arctan2(direction[1], direction[0])
        direction_angles = jnp.array([theta, phi])
        
        # Simulate event
        params = (energy, position, direction_angles)
        charges, times = simulate_event(params, sensor_params, event_key)
        
        # Store results
        all_charges.append(charges)
        all_times.append(times)
        all_metadata.append({
            'position': np.array(position),
            'direction': np.array(direction),
            'energy': float(energy),
            'event_id': i
        })
    
    # Stack arrays for efficient storage and loading
    if verbose:
        print("Stacking arrays...")
    
    all_charges_stacked = jnp.stack(all_charges)
    all_times_stacked = jnp.stack(all_times)
    
    events_data = {
        'all_charges': all_charges_stacked,
        'all_times': all_times_stacked,
        'metadata': all_metadata,
        'config': {
            'n_events': n_events,
            'n_photons': n_photons,
            'K': K,
            'seed': seed,
            'detector_type': detector_type,
            'detector_bounds': detector_bounds,
            'config_file': config_file,
            'generation_time': datetime.now().isoformat()
        }
    }
    
    # Save cache
    if verbose:
        print(f"Saving cache to {cache_filepath}")
    
    with open(cache_filepath, 'wb') as f:
        pickle.dump(events_data, f)
    
    if verbose:
        print(f"Cache saved successfully")
    
    return events_data


def load_event_cache(config_file, detector_type, K, verbose=True, cache_dir=None):
    """
    Load event cache, generating if necessary.
    
    This is a convenience function that always uses the hardcoded parameters.
    """
    return generate_and_cache_events(
        config_file=config_file,
        detector_type=detector_type,
        n_events=50000,      # Hardcoded
        n_photons=500000,    # Hardcoded
        K=K,
        seed=12345,
        verbose=verbose,
        cache_dir=cache_dir
    )