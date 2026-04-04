#!/usr/bin/env python3
"""
Optimization pipeline functions for single-ring track reconstruction.

Contains the combined loss function factory, the main Adam optimization loop,
and event data generation used by the CLI entry point (run.py) and notebooks.
"""

import os
import sys
import time
import glob
import math
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import optax
import uproot

from lucid.geometry import generate_detector
from lucid.utils import load_single_event, save_single_event, generate_random_params, print_particle_params
from lucid.utils import load_range_params, check_track_endpoint_in_detector
from lucid.simulation import setup_event_simulator
from lucid.sources.event_io import read_photon_data_from_photonsim

from lucid.optimization.utils.functions import estimate_muon_energy_from_photon_count, cone_points
from lucid.optimization.utils.functions import hierarchical_direction_search_cone, energy_scan_optimization
from lucid.optimization.utils.functions import cartesian_to_spherical, spherical_to_cartesian
from lucid.optimization.utils.functions import performance_summary

from lucid.optimization.grid_search import load_optimization_config, print_optimization_parameters
from lucid.optimization.grid_search import get_detector_bounds, hierarchical_position_grid_search
from lucid.detector_params import ParticleParams

from lucid.optimization.losses import (
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
