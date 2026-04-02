#!/usr/bin/env python3
"""
Single Ring Track Optimization Script (Adam Optimizer)

Four-stage optimization pipeline:
1. Energy estimation from photon count
2. Hierarchical grid search for position + t0
3. Hierarchical cone-based direction search
4. Energy scan optimization
5. Adam optimization (position + direction + t0 + energy)

Usage:
    python single_track_optimization.py <config_file_path>

Example:
    python single_track_optimization.py ../../config/single_track_optimization_config.json
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import time
import math
import numpy as np
from functools import partial
import pickle
from tqdm import tqdm
from jax import grad, jit, vmap, value_and_grad
import uproot
from jax import jit
import optax  # JAX optimization library

from tools.geometry import generate_detector
from tools.utils import load_single_event, save_single_event, generate_random_params, print_particle_params
from tools.utils import load_range_params, check_track_endpoint_in_detector
from tools.simulation import setup_event_simulator
from tools.generate import read_photon_data_from_photonsim

from tools.optimization.utils.functions import estimate_muon_energy_from_photon_count, cone_points
from tools.optimization.utils.functions import hierarchical_direction_search_cone, energy_scan_optimization
from tools.optimization.utils.functions import cartesian_to_spherical, spherical_to_cartesian
from tools.optimization.utils.functions import performance_summary

from tools.optimization.optimize import load_optimization_config, print_optimization_parameters
from tools.optimization.optimize import get_detector_bounds, hierarchical_position_grid_search
from tools.detector_params import ParticleParams

from tools.optimization.losses import (
    energy_loss,
    counts_loss,
    origin_time_loss,
    cone_time_loss,
    smooth_pinball,
    first_arrival_nll,
    poisson_nll,
    get_optimal_tau_vtx,
    TAU_VTX_PARAM_A,
    TAU_VTX_PARAM_B,
    TAU_VTX_PARAM_C,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Single Ring Track Optimization with Adam Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python single_track_optimization.py ../../config/single_track_optimization_config.json
        """
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to optimization configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: project_root/output)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name for the output file (default: timestamp-based name)'
    )
    return parser.parse_args()


def load_config(config_path):
    """Load and validate configuration file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_optimization_config(str(config_path))

    # Add default values for new parameters if not present
    if 'optimization_params' not in config:
        config['optimization_params'] = {}

    optimization_params = config['optimization_params']
    optimization_params.setdefault('damping_factor', 0.998)

    # Add Adam optimizer parameters if not present
    if 'adam_optimizer' not in config:
        config['adam_optimizer'] = {}

    adam_params = config['adam_optimizer']
    adam_params.setdefault('learning_rate', 0.2)
    adam_params.setdefault('b1', 0.9)
    adam_params.setdefault('b2', 0.999)
    adam_params.setdefault('eps', 1e-8)

    # Add fixed_vertex config if not present (default: disabled)
    if 'fixed_vertex' not in config:
        config['fixed_vertex'] = {'enabled': False, 'position': [0.0, 0.0, 0.0]}

    return config

def create_combined_loss_function(vertex_weight_scale, counts_weight_scale,
                                   prediction_simulator, c_medium,
                                   nrays=10_000, num_detectors=10764,
                                   detector_positions=None):
    """Create combined loss function with likelihood-based losses and dynamic tau_vtx.

    Uses:
    - poisson_nll for charge loss
    - first_arrival_nll for time loss
    - origin_time_loss with dynamic tau_vtx for vertex loss
    - 3-term combined loss formula matching tau scan scripts

    tau_vtx is computed from nrays and current energy using the learned parametrization.
    """
    nrays_float = float(nrays)
    TAU_TIME = 0.15  # Fixed tau for first_arrival_nll (time likelihood)

    @jit
    def combined_product_loss(track, hit_detector_positions, observed_times, observed_counts,
                            true_data, key):
        """
        Combined loss function with likelihood-based losses.

        Args:
            track: ParticleParams with energy, position, theta, phi, t0

        Returns:
            combined_loss, (vertex_loss_val, charge_loss_val, time_loss_val)
        """
        position = track.position
        t0 = track.t0
        energy = track.energy

        # Create track with t0=0 for simulation (time shift applied to observations)
        track_sim = ParticleParams(
            energy=energy, position=position,
            theta=track.theta, phi=track.phi, t0=jnp.array(0.0)
        )

        # Simulator returns (log_w, flat_times, flat_indices, total_charge)
        log_w, flat_times, flat_indices, total_charge = prediction_simulator(track_sim, key)

        # Charge loss: Poisson NLL
        charge_loss_val = poisson_nll(observed_counts, total_charge)

        # Time loss: First-arrival NLL
        # Shift observed times by t0 (time treatment matching tau scans)
        t_obs_shifted = observed_times - t0
        time_nll = first_arrival_nll(
            log_w, flat_times, flat_indices,
            t_obs_shifted, TAU_TIME, num_detectors
        )
        # Mask for hit sensors and average
        hit_mask = observed_counts > 0
        n_hit = jnp.sum(hit_mask) + 1e-8
        time_loss_val = jnp.sum(jnp.where(hit_mask, time_nll, 0.0)) / n_hit

        # Vertex loss: origin_time_loss with DYNAMIC tau_vtx
        # tau_vtx computed from current energy with stop_gradient
        tau_vtx = jax.lax.stop_gradient(
            TAU_VTX_PARAM_A * nrays_float + TAU_VTX_PARAM_B * energy + TAU_VTX_PARAM_C
        )
        tau_vtx = jnp.clip(tau_vtx, 0.05, 0.95)

        vertex_loss_val = origin_time_loss(
            jax.lax.stop_gradient(position), hit_detector_positions, observed_times,
            observed_counts, t0, tau=tau_vtx
        )

        # 3-term combined loss (matching tau scan scripts and notebook)
        c = charge_loss_val
        t = time_loss_val
        v = vertex_loss_val
        s = 0.

        combined_loss = jnp.sqrt((c + s) * (t + s) * (v + s)) \
            + jnp.sqrt((c + s) * jax.lax.stop_gradient((t + s) * (v + s))) \
            + jnp.sqrt((v + s) * jax.lax.stop_gradient((t + s) * (c + s)))

        return combined_loss, (vertex_loss_val, charge_loss_val, time_loss_val)

    combined_grad_fn = jit(value_and_grad(combined_product_loss, has_aux=True))
    return combined_grad_fn


def run_complete_optimization_adam(initial_t0, hit_detector_positions, observed_times, observed_counts,
                                   true_data, true_energy, true_position, true_direction, TRUE_T0,
                                   config, detector_bounds, prediction_simulator,
                                   combined_grad_fn, qe):
    """
    Run complete optimization pipeline with Adam optimizer
    """

    # Extract configuration parameters
    VERTEX_WEIGHT_SCALE = config['optimization_weights']['vertex_weight_scale']
    COUNTS_WEIGHT_SCALE = config['optimization_weights']['counts_weight_scale']

    # Convert true direction to spherical coordinates for difference calculation
    true_theta, true_phi = cartesian_to_spherical(true_direction)

    # Adam optimizer parameters
    ADAM_LEARNING_RATE = config['adam_optimizer']['learning_rate']
    ADAM_B1 = config['adam_optimizer']['b1']
    ADAM_B2 = config['adam_optimizer']['b2']
    ADAM_EPS = config['adam_optimizer']['eps']

    POS_N_DIV = config['position_grid_search']['pos_n_div']
    POS_LEVELS = config['position_grid_search']['pos_levels']
    POS_FRACTION = config['position_grid_search']['pos_fraction']
    POS_MIN_L = config['position_grid_search']['pos_min_L']
    T0_N_DIV = config['position_grid_search']['t0_n_div']
    T0_MIN = config['position_grid_search']['t0_min']
    T0_MAX = config['position_grid_search']['t0_max']

    CONE_LEVELS = config['cone_direction_search']['cone_levels']
    CONE_INITIAL_DIV = config['cone_direction_search']['cone_initial_div']
    CONE_MAX_ANGLE_DEG = config['cone_direction_search']['cone_max_angle_deg']
    CONE_REDUCTION = config['cone_direction_search']['cone_reduction']

    ENERGY_DELTA = config['energy_optimization']['energy_delta']
    ENERGY_SCAN_STEPS = config['energy_optimization']['energy_scan_steps']

    MAX_ITERATIONS = config['gradient_descent']['max_iterations']

    DETECTOR_R = detector_bounds.get('r', None)
    DETECTOR_H = detector_bounds.get('H', None)

    # Define parameter-specific scaling factors
    # Parameter structure: [x, y, z, t0, theta, phi, energy]

    POS_LR_SCALE = config['learning_rates']['position_learning_rate']
    DIR_LR_SCALE = config['learning_rates']['direction_learning_rate']
    T0_LR_SCALE = config['learning_rates']['t0_learning_rate']
    ENE_LR_SCALE = config['learning_rates']['energy_learning_rate']

    damping_factor = config['optimization_params']['damping_factor']

    verbosity = config.get('verbosity', {}).get('level', 2)
    tolerance = 1e-6

    # Stage 0: Energy estimation from photon count
    if verbosity >= 2:
        print("  Stage 0: Energy estimation from photon count")


    # we do the calculation for a track at the origin with theta=jnp.arccos(1/jnp.sqrt(3)). phi=jnp.pi/4.. and t0=0. (t0 has no effect).
    # the choice of phi and theta correspond to [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)].
    energy_guess_scan = energy_scan_optimization(
        prediction_simulator, jnp.array([0.,0.,0.]), jnp.arccos(1/jnp.sqrt(3)), jnp.pi/4., 0.,
        hit_detector_positions, observed_times, observed_counts,
        true_data, energy_guess=1000+np.random.uniform(-50,50), energy_delta=700,
        n_steps=10, verbosity=verbosity
    )
    energy_guess = energy_guess_scan['best_energy']

    if verbosity >= 2:
        print(f"    Energy guess: {energy_guess:.1f} (true: {true_energy:.1f})")

    N_photons = jnp.sum(observed_counts)
    # energy_guess = estimate_muon_energy_from_photon_count(N_photons, qe=qe)
    # if verbosity >= 2:
    #     print(f"    Observed photons: {N_photons}")
    #     print(f"    Energy guess: {energy_guess:.1f} (true: {true_energy:.1f})")

    # Stage 1: Position+t0 grid search
    if verbosity >= 2:
        print("  Stage 1: Position+t0 grid search (t0-loop approach)")
    pos_results = hierarchical_position_grid_search(
        hit_detector_positions, observed_times, observed_counts,
        true_position, TRUE_T0, initial_t0, detector_bounds,
        n_div=POS_N_DIV, t0_n_div=T0_N_DIV, levels=POS_LEVELS, fraction=POS_FRACTION,
        t0_min=T0_MIN, t0_max=T0_MAX,
        min_L=POS_MIN_L, verbosity=verbosity)

    optimal_position = pos_results['best_position']
    optimal_t0 = pos_results['best_t0']

    if optimal_position is None or optimal_t0 is None:
        if verbosity >= 1:
            print("  ERROR: Position+t0 search failed to find valid position")
        return None

    # Stage 2: Hierarchical cone direction search
    if verbosity >= 2:
        print("  Stage 2: Hierarchical cone direction search at optimal position")
    cone_results = hierarchical_direction_search_cone(
        prediction_simulator, optimal_position, optimal_t0,
        hit_detector_positions, observed_times, observed_counts,
        true_data, energy_guess, levels=CONE_LEVELS, initial_div=CONE_INITIAL_DIV,
        max_angle_deg=CONE_MAX_ANGLE_DEG, reduction=CONE_REDUCTION, verbosity=verbosity
    )

    # Stage 3: Energy scan optimization
    if verbosity >= 2:
        print("  Stage 3: Energy scan optimization")
    energy_scan_results = energy_scan_optimization(
        prediction_simulator, optimal_position,
        cone_results['best_theta'], cone_results['best_phi'],
        optimal_t0, hit_detector_positions, observed_times, observed_counts,
        true_data, energy_guess, energy_delta=ENERGY_DELTA,
        n_steps=ENERGY_SCAN_STEPS, verbosity=verbosity
    )

    optimal_energy = energy_scan_results['best_energy']

    # Stage 4: Adam optimization
    if verbosity >= 2:
        print("  Stage 4: Adam optimization (position + direction + t0 + energy)")
        print("    Update scaling: position=1.0, direction=0.025, t0=10.0, energy=10.0")

    # Create initial ParticleParams
    track = ParticleParams(
        energy=jnp.asarray(optimal_energy),
        position=jnp.array([optimal_position[0], optimal_position[1], optimal_position[2]]),
        theta=jnp.asarray(cone_results['best_theta']),
        phi=jnp.asarray(cone_results['best_phi']),
        t0=jnp.asarray(optimal_t0),
    )
    initial_track = track

    # Initialize Adam optimizer
    optimizer = optax.adam(learning_rate=ADAM_LEARNING_RATE, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS)
    opt_state = optimizer.init(track)

    position_scale = POS_LR_SCALE
    direction_scale = DIR_LR_SCALE
    t0_scale = T0_LR_SCALE
    energy_scale = ENE_LR_SCALE

    history = {
        'parameters': [track],
        'combined_losses': [],
        'vertex_losses': [],
        'counts_losses': [],
        'energy_losses': [],
        'direction_losses': [],
        'position_errors': [],
        'direction_errors': [],
        't0_errors': [],
        'energy_errors': [],
        'position_differences': [],  # 3D: predicted - true
        'direction_differences': [],  # 2D: [delta_theta, delta_phi]
        't0_differences': [],  # 1D: predicted - true
        'energy_differences': []  # 1D: predicted - true
    }

    # Random key for loss evaluations
    opt_key = jax.random.PRNGKey(12345)

    if verbosity >= 2:
        print(f"    Starting Adam optimization...")

    adam_start_time = time.time()
    current_damping_w = 5.0
    for iteration in range(MAX_ITERATIONS):
        opt_key, _ = jax.random.split(opt_key)

        (combined_loss, (vertex_loss, counts_loss_val, energy_loss_val)), grads = combined_grad_fn(
            track, hit_detector_positions, observed_times, observed_counts,
            true_data, opt_key
        )

        # Handle NaN gradients
        has_nan = any(jnp.any(jnp.isnan(g)) for g in jax.tree.leaves(grads))
        if has_nan:
            if verbosity >= 2:
                print("      Warning: NaN gradient detected, replacing with zeros")
            grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0), grads)

        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        # Note: Convergence check removed to ensure consistent history length across all events

        # Adam update with parameter-specific scaling
        updates, opt_state = optimizer.update(grads, opt_state, track)

        current_damping_w *= damping_factor

        # Apply parameter-specific scaling to updates
        scaled_updates = ParticleParams(
            energy=updates.energy * energy_scale * current_damping_w,
            position=updates.position * position_scale * current_damping_w,
            theta=updates.theta * direction_scale * current_damping_w,
            phi=updates.phi * direction_scale * current_damping_w,
            t0=updates.t0 * t0_scale * current_damping_w,
        )
        if iteration < 25:
            scaled_updates = scaled_updates._replace(energy=jnp.zeros_like(scaled_updates.energy))
            #scaled_updates = scaled_updates._replace(t0=jnp.zeros_like(scaled_updates.t0))

        # Apply scaled updates to parameters
        track = optax.apply_updates(track, scaled_updates)

        # Apply constraints
        if DETECTOR_R is not None and DETECTOR_H is not None:
            track = track._replace(
                position=jnp.array([
                    jnp.clip(track.position[0], -DETECTOR_R * 0.95, DETECTOR_R * 0.95),
                    jnp.clip(track.position[1], -DETECTOR_R * 0.95, DETECTOR_R * 0.95),
                    jnp.clip(track.position[2], -DETECTOR_H/2 * 0.95, DETECTOR_H/2 * 0.95),
                ]),
                t0=jnp.clip(track.t0, -20.0, 20.0),
                energy=jnp.clip(track.energy, 300.0, 2000.0),
            )
        else:
            track = track._replace(
                t0=jnp.clip(track.t0, -20.0, 20.0),
                energy=jnp.clip(track.energy, 300.0, 2000.0),
            )

        # Calculate current parameters
        current_position = track.position
        current_t0 = track.t0
        current_theta = track.theta
        current_phi = track.phi
        current_energy = track.energy
        current_direction = spherical_to_cartesian(current_theta, current_phi)

        # Calculate differences (predicted - true)
        position_difference = current_position - true_position
        t0_difference = current_t0 - TRUE_T0
        energy_difference = current_energy - true_energy
        theta_difference = current_theta - true_theta
        phi_difference = current_phi - true_phi
        direction_difference = jnp.array([theta_difference, phi_difference])

        # Calculate error magnitudes for printing
        position_error = jnp.linalg.norm(position_difference)
        t0_error = abs(t0_difference)
        energy_error = abs(energy_difference)
        cos_angle = jnp.clip(jnp.dot(current_direction, true_direction), -1.0, 1.0)
        direction_error = np.degrees(np.arccos(cos_angle))

        if verbosity >= 2 and ((iteration+1) % 100 == 0 or iteration == 0):
            print(f"      Iter {iteration}: pos_err={position_error:.3f}m, t0_err={t0_error:.3f}, "
                  f"dir_err={direction_error:.3f}°, E_err={energy_error:.1f}")
            print(f"      Loss: {combined_loss:.6f}, grad_norm: {grad_norm:.6f}")

        # Store history
        history['parameters'].append(track)
        history['combined_losses'].append(float(combined_loss))
        history['vertex_losses'].append(float(vertex_loss))
        history['counts_losses'].append(float(counts_loss_val))
        history['energy_losses'].append(float(energy_loss_val))
        history['position_errors'].append(float(position_error))
        history['direction_errors'].append(float(direction_error))
        history['t0_errors'].append(float(t0_error))
        history['energy_errors'].append(float(energy_error))
        history['position_differences'].append(position_difference.tolist())
        history['direction_differences'].append(direction_difference.tolist())
        history['t0_differences'].append(float(t0_difference))
        history['energy_differences'].append(float(energy_difference))

    # Final calculations
    adam_end_time = time.time()
    adam_optimization_time = adam_end_time - adam_start_time

    final_position = track.position
    final_t0 = track.t0
    final_theta = track.theta
    final_phi = track.phi
    final_energy = track.energy
    final_direction = spherical_to_cartesian(final_theta, final_phi)

    # Calculate final differences (predicted - true)
    final_position_difference = final_position - true_position
    final_t0_difference = final_t0 - TRUE_T0
    final_energy_difference = final_energy - true_energy
    final_theta_difference = final_theta - true_theta
    final_phi_difference = final_phi - true_phi
    final_direction_difference = jnp.array([final_theta_difference, final_phi_difference])

    # Calculate final error magnitudes
    final_position_error = jnp.linalg.norm(final_position_difference)
    final_t0_error = abs(final_t0_difference)
    final_energy_error = abs(final_energy_difference)
    final_cos_angle = jnp.clip(jnp.dot(final_direction, true_direction), -1.0, 1.0)
    final_direction_error = np.degrees(np.arccos(final_cos_angle))

    return {
        'energy_estimation': {
            'n_photons': N_photons,
            'energy_guess': float(energy_guess),
            'energy_guess_error': float(abs(energy_guess - true_energy))
        },
        'grid_position_search': pos_results,
        'cone_direction_search': cone_results,
        'energy_scan_search': energy_scan_results,
        'initial_track': initial_track,
        'final_position': final_position,
        'final_direction': final_direction,
        'final_theta': final_theta,
        'final_phi': final_phi,
        'final_t0': final_t0,
        'final_energy': final_energy,
        'final_combined_loss': history['combined_losses'][-1] if history['combined_losses'] else float('inf'),
        'final_vertex_loss': history['vertex_losses'][-1] if history['vertex_losses'] else float('inf'),
        'final_counts_loss': history['counts_losses'][-1] if history['counts_losses'] else float('inf'),
        'final_energy_loss': history['energy_losses'][-1] if history['energy_losses'] else float('inf'),
        'final_direction_loss': history['direction_losses'][-1] if history['direction_losses'] else float('inf'),
        'final_position_error': float(final_position_error),
        'final_direction_error': float(final_direction_error),
        'final_t0_error': float(final_t0_error),
        'final_energy_error': float(final_energy_error),
        'final_position_difference': final_position_difference.tolist(),  # 3D: [dx, dy, dz]
        'final_direction_difference': final_direction_difference.tolist(),  # 2D: [dtheta, dphi]
        'final_t0_difference': float(final_t0_difference),
        'final_energy_difference': float(final_energy_difference),
        'total_iterations': len(history['parameters']) - 1,
        'converged': grad_norm < tolerance,
        'adam_optimization_time': adam_optimization_time,
        'history': history
    }


def generate_event_data(event_idx, random_key, data_dir, data_simulator,
                       detector_bounds, fraction = 0.6, fixed_vertex_config=None):
    """Generate a single event with random parameters within detector bounds

    Args:
        event_idx: Event index
        random_key: JAX random key
        data_dir: Directory containing .root files (will randomly select one each call)
        data_simulator: Data simulator function
        detector_bounds: Detector bounds dictionary
        fixed_vertex_config: Dict with 'enabled' and 'position' keys to fix vertex at specific location
    """
    import glob

    # Get all .root files in the directory
    root_files = sorted(glob.glob(os.path.join(data_dir, "*.root")))
    if not root_files:
        raise ValueError(f"No .root files found in directory: {data_dir}")

    # Randomly select a file using the random key
    file_select_key, random_key = jax.random.split(random_key)
    file_idx = jax.random.randint(file_select_key, shape=(), minval=0, maxval=len(root_files))
    data_file = root_files[int(file_idx)]

    # Get number of entries from the selected file
    with uproot.open(data_file) as file:
        tree = file['OpticalPhotons']
        n_entries = tree.num_entries

    entry_idx = event_idx % n_entries

    photon_data = read_photon_data_from_photonsim(data_file, entry_idx)
    #photon_data['N'] = len(photon_data['photon_origins'])

    # Process photon data
    photon_origins = photon_data['photon_origins']
    photon_directions = photon_data['photon_directions']
    photon_times = photon_data['photon_times']
    N = len(photon_origins)
    # the number 1_000_000 is hard coded also in _simulation_core
    padding_size = max(0, 1_000_000-N)

    # Pad the origins array (2D array with shape [N,3])
    photon_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)), 
                                        mode='constant', constant_values=0)

    # Pad the directions array with a default unit vector [0,0,1]
    default_direction = jnp.array([0.0, 0.0, 1.0])
    padding_directions = jnp.tile(default_direction, (padding_size, 1))
    if padding_size > 0:
        photon_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
    else:
        photon_data['photon_directions'] = photon_directions

    # Pad the times array (1D array with shape [N])
    photon_data['photon_times'] = jnp.pad(photon_times, (0, padding_size),
                                          mode='constant', constant_values=0)

    photon_data['N'] = N

    key = random_key

    DETECTOR_R = detector_bounds['r']
    DETECTOR_H = detector_bounds['H']

    # Check if vertex should be fixed at a specific position
    if fixed_vertex_config and fixed_vertex_config.get('enabled', False):
        true_position = jnp.array(fixed_vertex_config['position'])
    else:
        # Random vertex generation within detector bounds
        r_vert = jax.random.uniform(key, shape=(), minval=0, maxval=DETECTOR_R * fraction)
        key, _ = jax.random.split(key)
        theta = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        key, _ = jax.random.split(key)
        z_vert = jax.random.uniform(key, shape=(), minval=-DETECTOR_H/2 * fraction,
                                   maxval=DETECTOR_H/2 * fraction)
        true_position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])

    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    true_direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])

    true_energy = photon_data['energy']
    TRUE_T0 = jax.random.uniform(key, shape=(), minval=-15.0, maxval=15.0)

    true_track = ParticleParams.from_cartesian(energy=true_energy, position=true_position,
                                               direction=true_direction, t0=TRUE_T0)

    # Compute rotation to transform from original direction (0,0,1) to true_direction
    original_direction = jnp.array([0.0, 0.0, 1.0])
    true_direction_norm = true_direction / (jnp.linalg.norm(true_direction) + 1e-8)

    # Rotation axis = cross product of original and target directions
    rotation_axis = jnp.cross(original_direction, true_direction_norm)
    axis_norm = jnp.linalg.norm(rotation_axis)

    # Handle case where directions are parallel (axis_norm ~ 0)
    rotation_axis = jnp.where(
        axis_norm < 1e-6,
        jnp.array([1.0, 0.0, 0.0]),  # Arbitrary axis when parallel
        rotation_axis / (axis_norm + 1e-8)
    )

    # Rotation angle = arccos of dot product
    rotation_angle = jnp.arccos(jnp.clip(
        jnp.dot(original_direction, true_direction_norm), -1.0, 1.0
    ))

    # Set rotation parameters in photon_data
    photon_data['rotation_axis'] = rotation_axis
    photon_data['rotation_angle'] = rotation_angle
    photon_data['apply_rotation'] = jnp.array(True)

    # Set translation parameters to move from origin to true_position
    photon_data['apply_translation'] = jnp.array(True)
    photon_data['translation_vector'] = true_position


    key, _ = jax.random.split(key)
    true_data = jax.lax.stop_gradient(data_simulator(true_track, key, photon_data))

    hit_counts, hit_times_raw = true_data
    hit_times = hit_times_raw + TRUE_T0

    return {
        'event_idx': event_idx,
        'entry_idx': entry_idx,
        'true_energy': float(true_energy),
        'true_position': np.array(true_position),
        'true_direction': np.array(true_direction),
        'TRUE_T0': float(TRUE_T0),
        'true_data': true_data,
        'hit_times': hit_times,
        'hit_counts': hit_counts,
        'photon_data': photon_data
    }


def main():
    """Main execution function"""

    # Parse arguments
    args = parse_arguments()

    # Load configuration
    print(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Extract basic configuration
    default_json_filename = config['basic_config']['default_json_filename']
    data_dir = config['basic_config']['data_dir']
    TEMPERATURE = config['basic_config']['temperature']
    N_EVENTS = config['basic_config']['n_events']
    K = config['basic_config']['k']
    Nphot = config['basic_config']['nphot']
    C_MEDIUM = config['basic_config']['c_medium']

    # Extract optimization parameters
    VERTEX_WEIGHT_SCALE = config['optimization_weights']['vertex_weight_scale']
    COUNTS_WEIGHT_SCALE = config['optimization_weights']['counts_weight_scale']

    qe = config['detector_params']['qe']

    # Extract optimization parameters
    damping_factor = config['optimization_params']['damping_factor']

    # Extract Adam parameters
    adam_lr = config['adam_optimizer']['learning_rate']
    adam_b1 = config['adam_optimizer']['b1']
    adam_b2 = config['adam_optimizer']['b2']
    adam_eps = config['adam_optimizer']['eps']

    MAX_ITERATIONS = config['gradient_descent']['max_iterations']
    verbosity = config.get('verbosity', {}).get('level', 2)
    store_true_data = config.get('storage', {}).get('store_true_data', True)

    # Setup detector
    print(f"Setting up detector from: {default_json_filename}")
    detector = generate_detector(default_json_filename)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius
    NUM_DETECTORS = len(detector_points)

    # Load range parametrization for track endpoint validation
    # TODO: Extract particle and medium from config instead of hardcoding
    range_params = load_range_params('muon', 'water')
    print(f"Loaded range parametrization: {range_params['description']}")

    # Get detector bounds
    detector_bounds = get_detector_bounds(detector)
    DETECTOR_R = detector_bounds['r'] if detector_bounds['type'] == 'cylinder' else None
    DETECTOR_H = detector_bounds['H'] if detector_bounds['type'] == 'cylinder' else None

    # Setup simulators
    PHYSICS_CONFIG = config['basic_config'].get('physics_config', None)
    print("Setting up simulators...")
    prediction_simulator = setup_event_simulator(
        default_json_filename, Nphot, TEMPERATURE, max_sensors_per_cell=4, K=K, is_data=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True
    )

    data_simulator = setup_event_simulator(
        default_json_filename, Nphot, temperature=0.0, K=20, is_data=True, is_calibration=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True
    )

    # Print parameters
    print_optimization_parameters(config, detector_bounds['r'], detector_bounds.get('H', 0), NUM_DETECTORS)

    # Print optimization parameters
    print("\n" + "=" * 80)
    print("Additional Optimization Parameters:")
    print("=" * 80)
    print(f"Quantum efficiency:      {qe}")
    print(f"Damping factor:          {damping_factor}")

    # Print Adam parameters
    print("\n" + "=" * 80)
    print("Adam Optimizer Parameters:")
    print("=" * 80)
    print(f"Learning rate:           {adam_lr}")
    print(f"Beta1:                   {adam_b1}")
    print(f"Beta2:                   {adam_b2}")
    print(f"Epsilon:                 {adam_eps}")

    # Check data directory
    print(f"\nData directory: {data_dir}")
    import glob
    root_files = glob.glob(os.path.join(data_dir, "*.root"))
    print(f"Found {len(root_files)} .root files in directory")

    # Create combined gradient function
    combined_grad_fn = create_combined_loss_function(
        VERTEX_WEIGHT_SCALE, COUNTS_WEIGHT_SCALE,
        prediction_simulator, C_MEDIUM, nrays=Nphot, num_detectors=NUM_DETECTORS
    )

    # Warm-up: Pre-compile JIT functions
    print("\nPre-compiling JIT functions...")
    main_key = jax.random.PRNGKey(42)
    warmup_key = jax.random.split(main_key, 1)[0]

    # Generate warmup event with endpoint validation
    warmup_attempts = 0
    warmup_endpoint_valid = False
    while not warmup_endpoint_valid and warmup_attempts < 10:
        warmup_attempts += 1
        warmup_event_data = generate_event_data(
            0, warmup_key, data_dir, data_simulator, detector_bounds, fraction=0.9,
            fixed_vertex_config=config.get('fixed_vertex')
        )
        warmup_endpoint_valid = check_track_endpoint_in_detector(
            warmup_event_data['true_position'],
            warmup_event_data['true_direction'],
            warmup_event_data['true_energy'],
            range_params, detector_bounds, fraction=0.9
        )
        if not warmup_endpoint_valid:
            warmup_key, _ = jax.random.split(warmup_key)

    if not warmup_endpoint_valid:
        print("ERROR: Could not generate valid warmup event. Exiting.")
        sys.exit(1)

    hit_mask = warmup_event_data['hit_counts'] > -999
    hit_detector_positions = detector_points[hit_mask]
    observed_times = warmup_event_data['hit_times'][hit_mask]
    observed_counts = warmup_event_data['hit_counts'][hit_mask]

    _ = run_complete_optimization_adam(
        initial_t0=0.0,
        hit_detector_positions=hit_detector_positions,
        observed_times=observed_times,
        observed_counts=observed_counts,
        true_data=warmup_event_data['true_data'],
        true_energy=warmup_event_data['true_energy'],
        true_position=warmup_event_data['true_position'],
        true_direction=warmup_event_data['true_direction'],
        TRUE_T0=warmup_event_data['TRUE_T0'],
        config=config,
        detector_bounds=detector_bounds,
        prediction_simulator=prediction_simulator,
        combined_grad_fn=combined_grad_fn,
        qe=qe
    )
    print("JIT compilation complete.\n")

    # Process events
    print(f"Starting optimization for {N_EVENTS} events...")
    if verbosity >= 2:
        print(f"Storage mode: store_true_data = {store_true_data}")
        print("=" * 80)

    # Storage for results
    all_event_results = []

    # Performance tracking
    energy_guess_errors = []
    grid_position_errors = []
    grid_t0_errors = []
    cone_direction_errors = []
    energy_scan_improvements = []
    final_position_errors = []
    final_direction_errors = []
    final_t0_errors = []
    final_energy_errors = []
    final_position_differences = []  # 3D: [dx, dy, dz]
    final_direction_differences = []  # 2D: [dtheta, dphi]
    final_t0_differences = []
    final_energy_differences = []
    final_combined_losses = []
    final_vertex_losses = []
    final_counts_losses = []
    final_energy_losses = []
    final_direction_losses = []
    convergence_rates = []

    # Generate random keys
    event_keys = jax.random.split(main_key, N_EVENTS)

    # Process each event
    progress_bar = tqdm(range(N_EVENTS), desc="Processing events") if verbosity == 0 else range(N_EVENTS)
    for event_idx in progress_bar:
        event_start_time = time.time()
        try:
            if verbosity >= 2:
                print(f"\n--- Processing Event {event_idx} ---")

            # Generate event data with endpoint validation
            # Try up to 10 times to generate an event with endpoint within detector
            event_generation_attempts = 0
            max_attempts = 10
            event_key = event_keys[event_idx]
            endpoint_valid = False

            while not endpoint_valid and event_generation_attempts < max_attempts:
                event_generation_attempts += 1

                # Generate event data
                event_data = generate_event_data(
                    event_idx, event_key, data_dir,
                    data_simulator, detector_bounds, fraction=0.9,
                    fixed_vertex_config=config.get('fixed_vertex')
                )

                # Extract event parameters
                true_position = event_data['true_position']
                true_direction = event_data['true_direction']
                true_energy = event_data['true_energy']

                # Check if track endpoint is within detector bounds
                endpoint_valid = check_track_endpoint_in_detector(
                    true_position, true_direction, true_energy,
                    range_params, detector_bounds, fraction=0.9
                )

                if not endpoint_valid:
                    if event_generation_attempts == 5:
                        print(f"  WARNING: Event {event_idx} - 5 failed attempts to generate event with endpoint in detector")

                    # Generate new random key for next attempt
                    event_key, _ = jax.random.split(event_key)

            # Check if we exceeded max attempts
            if not endpoint_valid:
                print(f"  ERROR: Event {event_idx} - Failed to generate event with endpoint in detector after {max_attempts} attempts")
                print(f"  Last attempt: position={true_position}, energy={true_energy:.1f} MeV")
                print(f"  Exiting program.")
                sys.exit(1)

            if verbosity >= 2 and event_generation_attempts > 1:
                print(f"  Generated valid event after {event_generation_attempts} attempts")

            # Continue with already extracted parameters
            TRUE_T0 = event_data['TRUE_T0']
            true_data = event_data['true_data']

            # Get hit information
            hit_mask = event_data['hit_counts'] > -999
            hit_detector_positions = detector_points[hit_mask]
            observed_times = event_data['hit_times'][hit_mask]
            observed_counts = event_data['hit_counts'][hit_mask]

            # Convert true direction to spherical coordinates
            true_theta, true_phi = cartesian_to_spherical(true_direction)

            initial_t0 = 0.0

            if verbosity >= 2:
                print(f"  True position: {true_position}")
                print(f"  True direction: {true_direction}")
                print(f"  True energy: {true_energy:.1f}")
                print(f"  True t0: {TRUE_T0:.3f}")
                print(f"  Initial t0 guess: {initial_t0:.3f}")

            # Run optimization
            if verbosity >= 2:
                print("  Running Adam optimization...")
            results = run_complete_optimization_adam(
                initial_t0=initial_t0,
                hit_detector_positions=hit_detector_positions,
                observed_times=observed_times,
                observed_counts=observed_counts,
                true_data=true_data,
                true_energy=true_energy,
                true_position=true_position,
                true_direction=true_direction,
                TRUE_T0=TRUE_T0,
                config=config,
                detector_bounds=detector_bounds,
                prediction_simulator=prediction_simulator,
                combined_grad_fn=combined_grad_fn,
                qe=qe
            )

            if results is None:
                if verbosity >= 1:
                    print(f"  ERROR: Optimization failed for event {event_idx}")
                continue

            # Calculate improvements
            energy_guess_error = results['energy_estimation']['energy_guess_error']
            grid_position_error = results['grid_position_search']['position_error']
            grid_t0_error = results['grid_position_search']['t0_error']

            cone_direction = results['cone_direction_search']['best_direction']
            cone_cos_angle = np.clip(np.dot(cone_direction, true_direction), -1.0, 1.0)
            cone_direction_error = np.degrees(np.arccos(cone_cos_angle))

            energy_scan_improvement = results['energy_scan_search']['energy_improvement']

            # Create event data for storage
            event_data_to_store = {
                'event_idx': event_data['event_idx'],
                'entry_idx': event_data['entry_idx'],
                'true_energy': event_data['true_energy'],
                'true_position': event_data['true_position'],
                'true_direction': event_data['true_direction'],
                'TRUE_T0': event_data['TRUE_T0'],
                'hit_detector_positions': hit_detector_positions,
                'observed_times': observed_times,
                'observed_counts': observed_counts,
                'n_hits': int(jnp.sum(hit_mask))
            }

            if store_true_data:
                event_data_to_store['true_data'] = event_data['true_data']

            # Calculate event timing
            event_end_time = time.time()
            total_event_time = event_end_time - event_start_time

            # Store results
            event_result = {
                'event_data': event_data_to_store,
                'initial_t0': initial_t0,
                'true_theta': float(true_theta),
                'true_phi': float(true_phi),
                'energy_guess_error': energy_guess_error,
                'grid_position_error': grid_position_error,
                'grid_t0_error': grid_t0_error,
                'cone_direction_error': cone_direction_error,
                'energy_scan_improvement': energy_scan_improvement,
                'total_event_time': total_event_time,
                'adam_optimization_time': results['adam_optimization_time'],
                'optimization_results': results
            }
            all_event_results.append(event_result)

            # Track metrics
            energy_guess_errors.append(energy_guess_error)
            grid_position_errors.append(grid_position_error)
            grid_t0_errors.append(grid_t0_error)
            cone_direction_errors.append(cone_direction_error)
            energy_scan_improvements.append(energy_scan_improvement)
            final_position_errors.append(results['final_position_error'])
            final_direction_errors.append(results['final_direction_error'])
            final_t0_errors.append(results['final_t0_error'])
            final_energy_errors.append(results['final_energy_error'])
            final_position_differences.append(results['final_position_difference'])
            final_direction_differences.append(results['final_direction_difference'])
            final_t0_differences.append(results['final_t0_difference'])
            final_energy_differences.append(results['final_energy_difference'])
            final_combined_losses.append(results['final_combined_loss'])
            final_vertex_losses.append(results['final_vertex_loss'])
            final_counts_losses.append(results['final_counts_loss'])
            final_energy_losses.append(results['final_energy_loss'])
            final_direction_losses.append(results['final_direction_loss'])
            convergence_rates.append(1.0 if results['converged'] else 0.0)

            # Print results
            if verbosity == 1:
                print(f"Event {event_idx}: pos_err={results['final_position_error']*100:.1f}cm, "
                      f"dir_err={results['final_direction_error']:.2f}°, t0_err={results['final_t0_error']:.3f}, "
                      f"E_err={results['final_energy_error']:.1f}")
            elif verbosity >= 2:
                print(f"  Energy guess error: {energy_guess_error:.1f}")
                print(f"  Grid search - Position error: {grid_position_error*100:.1f}cm, t0 error: {grid_t0_error:.3f}")
                print(f"  Cone direction error: {cone_direction_error:.2f}°")
                print(f"  Energy scan improvement: {energy_scan_improvement:.1f}")
                print(f"  Final errors - Position: {results['final_position_error']:.3f}m, "
                      f"Direction: {results['final_direction_error']:.2f}°, t0: {results['final_t0_error']:.3f}, "
                      f"Energy: {results['final_energy_error']:.1f}")
                print(f"  Converged: {results['converged']}, Iterations: {results['total_iterations']}")

        except Exception as e:
            if verbosity >= 1:
                print(f"Error processing event {event_idx}: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()
            continue

    if verbosity >= 1:
        print(f"\nCompleted processing {len(all_event_results)} events successfully")

    # Performance summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION PERFORMANCE SUMMARY")
    print("=" * 80)

    performance_summary(
        energy_guess_errors,
        grid_position_errors,
        cone_direction_errors,
        energy_scan_improvements,
        final_position_errors,
        final_direction_errors,
        final_t0_errors,
        final_energy_errors,
        final_combined_losses,
        final_vertex_losses,
        final_counts_losses,
        final_energy_losses,
        convergence_rates,
    )

    # Print t0 grid search statistics
    print("\n" + "=" * 80)
    print("4D Grid Search t0 Performance:")
    print("=" * 80)
    grid_t0_errors_arr = np.array(grid_t0_errors)
    print(f"Mean t0 error:   {np.mean(grid_t0_errors_arr):.4f}")
    print(f"Median t0 error: {np.median(grid_t0_errors_arr):.4f}")
    print(f"Std t0 error:    {np.std(grid_t0_errors_arr):.4f}")
    print(f"Min t0 error:    {np.min(grid_t0_errors_arr):.4f}")
    print(f"Max t0 error:    {np.max(grid_t0_errors_arr):.4f}")

    # Timing summary
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    total_event_times = [event['total_event_time'] for event in all_event_results]
    adam_times = [event['adam_optimization_time'] for event in all_event_results]

    print(f"Total Event Processing Time:")
    print(f"  Mean:   {np.mean(total_event_times):.2f}s")
    print(f"  Median: {np.median(total_event_times):.2f}s")
    print(f"  Min:    {np.min(total_event_times):.2f}s")
    print(f"  Max:    {np.max(total_event_times):.2f}s")
    print(f"  Total:  {np.sum(total_event_times):.2f}s")

    print(f"\nAdam Optimization Time:")
    print(f"  Mean:   {np.mean(adam_times):.2f}s")
    print(f"  Median: {np.median(adam_times):.2f}s")
    print(f"  Min:    {np.min(adam_times):.2f}s")
    print(f"  Max:    {np.max(adam_times):.2f}s")
    print(f"  Total:  {np.sum(adam_times):.2f}s")

    print(f"\nAdam optimization as % of total event time: {100 * np.sum(adam_times) / np.sum(total_event_times):.1f}%")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert arrays
    energy_guess_errors = np.array(energy_guess_errors)
    grid_position_errors = np.array(grid_position_errors)
    grid_t0_errors = np.array(grid_t0_errors)
    cone_direction_errors = np.array(cone_direction_errors)
    energy_scan_improvements = np.array(energy_scan_improvements)
    final_position_errors = np.array(final_position_errors)
    final_direction_errors = np.array(final_direction_errors)
    final_t0_errors = np.array(final_t0_errors)
    final_energy_errors = np.array(final_energy_errors)
    final_position_differences = np.array(final_position_differences)  # Shape: (N_events, 3)
    final_direction_differences = np.array(final_direction_differences)  # Shape: (N_events, 2)
    final_t0_differences = np.array(final_t0_differences)
    final_energy_differences = np.array(final_energy_differences)
    final_combined_losses = np.array(final_combined_losses)
    final_vertex_losses = np.array(final_vertex_losses)
    final_counts_losses = np.array(final_counts_losses)
    final_energy_losses = np.array(final_energy_losses)
    final_direction_losses = np.array(final_direction_losses)
    convergence_rates = np.array(convergence_rates)

    # Prepare results summary with config
    results_summary = {
        'timestamp': timestamp,
        'config_file': args.config_file,
        'config': {
            'optimizer': 'Adam',
            'nphot': Nphot,
            'adam_learning_rate': adam_lr,
            'adam_b1': adam_b1,
            'adam_b2': adam_b2,
            'adam_eps': adam_eps,
            'n_events': N_EVENTS,
            'temperature': TEMPERATURE,
            'vertex_weight_scale': VERTEX_WEIGHT_SCALE,
            'counts_weight_scale': COUNTS_WEIGHT_SCALE,
            'pos_n_div': config['position_grid_search']['pos_n_div'],
            'pos_levels': config['position_grid_search']['pos_levels'],
            'pos_fraction': config['position_grid_search']['pos_fraction'],
            'pos_min_L': config['position_grid_search']['pos_min_L'],
            't0_n_div': config['position_grid_search']['t0_n_div'],
            't0_min': config['position_grid_search']['t0_min'],
            't0_max': config['position_grid_search']['t0_max'],
            'cone_levels': config['cone_direction_search']['cone_levels'],
            'cone_initial_div': config['cone_direction_search']['cone_initial_div'],
            'cone_max_angle_deg': config['cone_direction_search']['cone_max_angle_deg'],
            'cone_reduction': config['cone_direction_search']['cone_reduction'],
            'energy_delta': config['energy_optimization']['energy_delta'],
            'energy_scan_steps': config['energy_optimization']['energy_scan_steps'],
            'qe': qe,
            'damping_factor': damping_factor,
            'detector_r': float(DETECTOR_R) if DETECTOR_R is not None else None,
            'detector_h': float(DETECTOR_H) if DETECTOR_H is not None else None,
            'detector_bounds': detector_bounds,
            'data_dir': data_dir,
            'detector_file': default_json_filename
        },
        'raw_data': {
            'energy_guess_errors': energy_guess_errors.tolist(),
            'grid_position_errors': grid_position_errors.tolist(),
            'grid_t0_errors': grid_t0_errors.tolist(),
            'cone_direction_errors': cone_direction_errors.tolist(),
            'energy_scan_improvements': energy_scan_improvements.tolist(),
            'final_position_errors': final_position_errors.tolist(),
            'final_direction_errors': final_direction_errors.tolist(),
            'final_t0_errors': final_t0_errors.tolist(),
            'final_energy_errors': final_energy_errors.tolist(),
            'final_position_differences': final_position_differences.tolist(),  # (N_events, 3)
            'final_direction_differences': final_direction_differences.tolist(),  # (N_events, 2)
            'final_t0_differences': final_t0_differences.tolist(),
            'final_energy_differences': final_energy_differences.tolist(),
            'final_combined_losses': final_combined_losses.tolist(),
            'final_vertex_losses': final_vertex_losses.tolist(),
            'final_counts_losses': final_counts_losses.tolist(),
            'final_energy_losses': final_energy_losses.tolist(),
            'final_direction_losses': final_direction_losses.tolist(),
            'convergence_rates': convergence_rates.tolist()
        },
        'all_event_results': all_event_results
    }

    # Save results
    if args.name:
        output_file = output_dir / f'{args.name}.pkl'
    else:
        output_file = output_dir / f'single_track_optimization_adam_{timestamp}_{N_EVENTS}_events.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results_summary, f)

    print(f"\n{'=' * 80}")
    print("Results saved:")
    print(f"  {output_file}")
    print(f"{'=' * 80}")

    print("\nOptimization complete!")


if __name__ == '__main__':
    main()
