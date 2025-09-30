import json
import jax.numpy as jnp
from pathlib import Path


def load_optimization_config(config_path):
    """
    Load optimization configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def print_optimization_parameters(config, detector_r, detector_h, num_detectors, verbosity=None):
    """
    Print all optimization parameters in a formatted way.
    
    Args:
        config: Configuration dictionary loaded from JSON
        detector_r: Detector radius (from detector object)
        detector_h: Detector height (from detector object)
        num_detectors: Number of detectors (from detector object)
        verbosity: Verbosity level (0, 1, 2). If None, uses config value.
    """
    if verbosity is None:
        verbosity = config.get('verbosity', {}).get('level', 2)
    
    if verbosity >= 2:
        print(f"Number of sensors: {num_detectors}")
        print(f"Detector dimensions: R={detector_r:.1f}m, H={detector_h:.1f}m")
        print(f"Fixed temperature: {config['basic_config']['temperature']}")
        print(f"Speed of light in medium: {config['basic_config']['c_medium']}")
        print(f"Number of events for analysis: {config['basic_config']['n_events']}")
        print(f"POS grid parameters:")
        print(f"  Initial spacing L0: {config['position_grid_search']['pos_l0']}")
        print(f"  Refinement levels: {config['position_grid_search']['pos_levels']}")
        print(f"  Reduction factor: {config['position_grid_search']['pos_reduction']}")
        print(f"Hierarchical cone direction search parameters:")
        print(f"  Levels: {config['cone_direction_search']['cone_levels']}")
        print(f"  Initial divisions: {config['cone_direction_search']['cone_initial_div']}")
        print(f"  Max cone angle: {config['cone_direction_search']['cone_max_angle_deg']}°")
        print(f"  Cone reduction factor: {config['cone_direction_search']['cone_reduction']}")
        print(f"Energy optimization parameters:")
        print(f"  Energy delta: ±{config['energy_optimization']['energy_delta']}")
        print(f"  Energy scan steps: {config['energy_optimization']['energy_scan_steps']}")


def get_detector_params_from_config(config):
    """
    Convert detector parameters from config to JAX array format.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: JAX arrays for detector parameters
    """
    detector_config = config['detector_params']
    return (
        jnp.array(detector_config['scatter_length']),
        jnp.array(detector_config['reflection_rate']),
        jnp.array(detector_config['absorption_length']),
        jnp.array(detector_config['gumbel_softmax_temp'])
    )