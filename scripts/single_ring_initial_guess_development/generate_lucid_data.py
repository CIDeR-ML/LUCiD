#!/usr/bin/env python3
"""
Generate simulation data using LUCiD
"""

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os
import sys
import json

# Add parent directories to path to access tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# LUCiD imports
from tools.utils import base_dir_path
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.visualization import create_detector_display

def generate_simulation_data(config_file=None, 
                           n_events=100,
                           output_dir=None,
                           random_seed=42):
    """
    Generate simulation data using LUCiD
    
    Parameters:
    -----------
    config_file : str
        Path to detector geometry configuration file
    n_events : int
        Number of events to generate
    output_dir : str
        Directory to save generated data
    """
    
    # Set default paths using base_dir_path
    if config_file is None:
        config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'generated_data')
    
    # Initialize random key
    key = jax.random.PRNGKey(random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup detector and simulation
    print(f"Loading detector configuration from {config_file}")
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    n_sensors = len(sensor_positions)
    
    print(f"Detector has {n_sensors} sensors")
    print(f"Sensor positions shape: {sensor_positions.shape}")
    
    # Setup simulation parameters
    sensor_params = (
        jnp.array(50.0),    # scatter_length
        jnp.array(0.1),    # reflection_rate
        jnp.array(100.0),    # absorption_length
        jnp.array(0.001)    # gumbel_softmax_temperature
    )
    
    # Setup event simulator
    print("Setting up event simulator...")
    simulate_event = setup_event_simulator(
        json_filename=config_file,
        max_sensors_per_cell=4,
        n_photons=1_000_000,  # Number of photons to simulate
        temperature=0.2,       # Temperature for differentiable simulation
        K=2,                   # Number of interactions
        detector_type='Cylinder'  # Detector type from config
    )
    
    print("Simulation setup complete")
    
    # Generate events
    events_data = []
    
    for i in range(n_events):
        # Generate random event parameters
        subkey, key = jax.random.split(key)
        
        # Random vertex position (within detector bounds)
        if hasattr(detector, 'r'):  # Cylinder
            r_vert = jax.random.uniform(subkey, shape=(), minval=0, maxval=detector.r * 0.8)
            subkey, _ = jax.random.split(subkey)
            theta = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
            subkey, _ = jax.random.split(subkey)
            z_vert = jax.random.uniform(subkey, shape=(), minval=-detector.H/2 * 0.8, maxval=detector.H/2 * 0.8)
            event_position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        else:  # Box or other geometries
            subkey, _ = jax.random.split(subkey)
            event_position = jax.random.uniform(subkey, shape=(3,), minval=-2.0, maxval=2.0)
        
        # Random direction
        subkey, _ = jax.random.split(subkey)
        phi = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
        subkey, _ = jax.random.split(subkey)
        cos_theta = jax.random.uniform(subkey, shape=(), minval=-1, maxval=1)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
        
        # Random energy (in MeV)
        subkey, _ = jax.random.split(subkey)
        energy = jax.random.uniform(subkey, shape=(), minval=250.0, maxval=900.0)
        
        # Event time
        event_time = jnp.array(0.0)
        
        # Prepare parameters in the format expected by simulate_event
        # Convert direction to spherical angles (theta, phi)
        theta = jnp.arccos(direction[2])  # z-component gives cos(theta)
        phi = jnp.arctan2(direction[1], direction[0])
        direction_angles = jnp.array([theta, phi])
        
        # Pack parameters as expected by the simulator
        particle_params = (energy, event_position, direction_angles)
        
        # Simulate event
        print(f"Generating event {i+1}/{n_events}...", end='\r')
        hit_charges, hit_times = simulate_event(particle_params, sensor_params, subkey)
        
        # Store event data
        event_dict = {
            'event_id': i,
            'vertex_position': event_position.tolist(),
            'direction': direction.tolist(),
            'energy': float(energy),
            'event_time': float(event_time),
            'hit_times': hit_times.tolist(),
            'hit_charges': hit_charges.tolist(),
            'sensor_params': {
                'scatter_length': float(sensor_params[0]),
                'reflection_rate': float(sensor_params[1]),
                'absorption_length': float(sensor_params[2]),
                'gumbel_softmax_temperature': float(sensor_params[3])
            }
        }
        events_data.append(event_dict)
    
    print(f"\nGenerated {n_events} events")
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'lucid_simulated_events_{timestamp}.json')
    
    # Save metadata along with events
    output_data = {
        'metadata': {
            'detector_config': config_file,
            'n_events': n_events,
            'n_sensors': n_sensors,
            'generation_time': timestamp,
            'detector_type': detector.__class__.__name__
        },
        'events': events_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Data saved to {output_file}")
    
    # Generate and save a sample visualization
    if n_events > 0 and False:  # Disabled for now - visualization function needs different approach
        print("\nCreating sample visualization...")
        # TODO: Implement proper visualization using the display function
        pass
    
    return output_file

if __name__ == "__main__":
    # Generate data with default parameters (paths will be set automatically using base_dir_path)
    output_file = generate_simulation_data(
        n_events=50  # Generate 5 events for testing
    )
    
    print(f"\nSimulation complete! Generated data saved to: {output_file}")