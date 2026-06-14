#!/usr/bin/env python
"""
Evaluate reconstruction performance using learned tau_vtx parametrization.

This script uses the parametrization derived from combined tau scans:
    tau_vtx = a * Nrays + b * Energy + c

During gradient-based optimization (Stage 4), tau_vtx is computed dynamically
from the current reconstructed energy, with stop_gradient applied to prevent
gradients from flowing through the parametrization.

Grid: Nrays = [50k, 150k, 250k], Energy = [500, 1000, 1500] MeV
Total: 9 combinations x 50 events = 450 events

Usage:
    python run_eval_with_parametrization.py --output OUTPUT_DIR [--n-events N]
"""

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

import argparse
import csv
import os
import time
import glob
from datetime import datetime
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import uproot
import optax
from jax import jit, value_and_grad
from jax.scipy.special import gammaln

# LUCiD imports
from lucid.geometry import generate_detector
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.simulation import setup_event_simulator
from lucid.utils import load_range_params, check_track_endpoint_in_detector
from lucid.detector_params import ParticleParams, load_detector_params
from lucid.wavelength import DEFAULT_WAVELENGTH_NM
from lucid.optimization.grid_search import (
    load_optimization_config,
    get_detector_bounds,
    hierarchical_position_grid_search
)
from lucid.optimization.utils.functions import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)
from lucid.losses import (
    first_arrival_nll,
    get_optimal_tau_vtx,
    TAU_VTX_PARAM_A,
    TAU_VTX_PARAM_B,
    TAU_VTX_PARAM_C,
)


# =============================================================================
# EVALUATION GRID (no tau_vtx dimension - it's computed dynamically)
# =============================================================================
# Format: (nrays, energy_MeV)
EVAL_GRID = [
    # Nrays=50k
    (50_000,  500),
    (50_000,  1000),
    (50_000,  1500),
    # Nrays=150k
    (150_000, 500),
    (150_000, 1000),
    (150_000, 1500),
    # Nrays=250k
    (250_000, 500),
    (250_000, 1000),
    (250_000, 1500),
]

# Total combinations: 9
# Events per combination: 50
# Total events: 450

ENERGY_BASE_PATH = '/sdf/data/neutrino/cjesus/photonsim_output/water/monoenergetic/event_by_event/mu-/'
PHYSICS_CONFIG = str(project_root / 'config' / 'SK_physics_config.json')
GEOM_CONFIG = str(project_root / 'config' / 'SK_geom_config.json')
OPT_CONFIG = str(project_root / 's3df_jobs' / 'nrays_config' / 'opt_config_5.json')

# Fixed parameters
TAU_TIME = 0.15  # tau for first-arrival NLL (time likelihood)
TEMPERATURE = 0.1
K = 7
N_BOOTSTRAP = 100  # Number of bootstrap samples for CI estimation


# =============================================================================
# BOOTSTRAPPING
# =============================================================================
def bootstrap_percentile_ci(data, percentile=68, n_bootstrap=N_BOOTSTRAP, ci_level=68, rng=None):
    """Bootstrap CI for a given percentile."""
    data = np.asarray(data)
    n = data.size
    if n == 0:
        return None
    if rng is None:
        rng = np.random.default_rng()

    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boots[i] = np.percentile(sample, percentile)

    alpha = (100 - ci_level) / 2
    ci_lower = np.percentile(boots, alpha)
    ci_upper = np.percentile(boots, 100 - alpha)
    est = np.percentile(data, percentile)

    return {
        "estimate": float(est),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_error": float(np.std(boots, ddof=1)),
    }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def poisson_nll(true, pred, eps=1e-8):
    """Poisson negative log-likelihood, normalized by total true counts."""
    nll = pred - true * jnp.log(pred + eps) + gammaln(true + 1.0)
    return jnp.sum(nll) / (jnp.sum(true) + eps)


def softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.)


def smooth_pinball(r, tau=0.1, sigma=0.5):
    """Smooth version of pinball/quantile loss."""
    pos = softplus(r / sigma) * sigma
    neg = softplus(-r / sigma) * sigma
    return tau * pos + (1.0 - tau) * neg


@jit
def origin_time_loss_configurable(origin, detector_positions, true_times, true_q, t0,
                                   tau_vtx, photosensor_radius=0.25, c_medium=(0.299792/1.33)):
    """Origin time loss with configurable tau_vtx parameter."""
    d = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected = (d - photosensor_radius) / c_medium
    r = true_times - expected - t0
    w = jnp.where(true_q > 0., 1., 0.)
    wsum = jnp.sum(w) + 1e-8
    main = jnp.sum(w * smooth_pinball(r, tau=tau_vtx, sigma=0.25)) / wsum
    return main


def create_combined_loss_function_dynamic_tau(prediction_simulator, num_detectors,
                                               tau_time, nrays, hit_detector_positions):
    """
    Create likelihood-based combined loss function with DYNAMIC tau_vtx.

    tau_vtx is computed from the current reconstructed energy using the
    learned parametrization, with stop_gradient applied.
    """
    # Convert nrays to float for JAX compatibility
    nrays_float = float(nrays)

    @jit
    def combined_product_loss(params, observed_times, observed_counts, key):
        position = params[:3]
        t0 = params[3]
        theta = params[4]
        phi = params[5]
        energy = params[6]

        track = ParticleParams(energy=energy, position=position, theta=theta, phi=phi, t0=jnp.array(0.0))
        log_w, flat_times, flat_indices, total_charge = prediction_simulator(track, key)

        charge_loss = poisson_nll(observed_counts, total_charge)
        t_obs_shifted = observed_times - t0

        time_nll = first_arrival_nll(
            log_w, flat_times, flat_indices,
            t_obs_shifted, tau_time, num_detectors)

        hit_mask = observed_counts > 0
        n_hit = jnp.sum(hit_mask) + 1e-8
        time_loss = jnp.sum(jnp.where(hit_mask, time_nll, 0.0)) / n_hit

        # DYNAMIC tau_vtx: compute from current energy with stop_gradient
        # This prevents gradients from flowing through the parametrization
        tau_vtx = jax.lax.stop_gradient(
            TAU_VTX_PARAM_A * nrays_float + TAU_VTX_PARAM_B * energy + TAU_VTX_PARAM_C
        )

        # Clamp tau_vtx to reasonable range [0.05, 0.95]
        tau_vtx = jnp.clip(tau_vtx, 0.05, 0.95)

        vertex_loss_val = origin_time_loss_configurable(
            jax.lax.stop_gradient(position), hit_detector_positions, observed_times,
            observed_counts, t0, tau_vtx=tau_vtx)

        c = charge_loss
        t = time_loss
        v = vertex_loss_val
        s = 0.

        combined = jnp.sqrt((c + s) * (t + s) * (v + s))\
            + jnp.sqrt((c + s) * jax.lax.stop_gradient((t + s) * (v + s)))\
            + jnp.sqrt((v + s) * jax.lax.stop_gradient((t + s) * (c + s)))

        sqrt_ct = jnp.sqrt(c * t)

        return combined, (charge_loss, time_loss, vertex_loss_val, sqrt_ct, tau_vtx)

    combined_grad_fn = jit(value_and_grad(combined_product_loss, has_aux=True))
    return combined_grad_fn, combined_product_loss


# =============================================================================
# EVENT GENERATION
# =============================================================================
def generate_event_data(event_idx, random_key, data_dir, data_simulator,
                        detector_bounds, detector_points, fraction=0.9):
    """Generate a single event with random parameters within detector bounds."""
    root_files = sorted(glob.glob(os.path.join(data_dir, "*.root")))
    if not root_files:
        raise ValueError(f"No .root files found in directory: {data_dir}")

    file_select_key, random_key = jax.random.split(random_key)
    file_idx = jax.random.randint(file_select_key, shape=(), minval=0, maxval=len(root_files))
    data_file = root_files[int(file_idx)]

    with uproot.open(data_file) as file:
        tree = file['OpticalPhotons']
        n_entries = tree.num_entries

    entry_idx = event_idx % n_entries
    photon_data = read_photon_data_from_photonsim(data_file, entry_idx)

    photon_origins = photon_data['photon_origins']
    photon_directions = photon_data['photon_directions']
    photon_times = photon_data['photon_times']
    N = len(photon_origins)

    padding_size = max(0, 1_000_000 - N)
    photon_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)),
                                             mode='constant', constant_values=0)

    default_direction = jnp.array([0.0, 0.0, 1.0])
    padding_directions = jnp.tile(default_direction, (padding_size, 1))
    if padding_size > 0:
        photon_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
    else:
        photon_data['photon_directions'] = photon_directions

    photon_data['photon_times'] = jnp.pad(photon_times, (0, padding_size),
                                          mode='constant', constant_values=0)

    if 'wavelengths' in photon_data:
        photon_data['wavelengths'] = jnp.pad(
            photon_data['wavelengths'], (0, padding_size),
            mode='constant', constant_values=DEFAULT_WAVELENGTH_NM)

    photon_data['N'] = N

    key = random_key
    DETECTOR_R = detector_bounds['r']
    DETECTOR_H = detector_bounds['H']

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

    true_track = ParticleParams.from_cartesian(
        energy=true_energy, position=true_position, direction=true_direction, t0=0.0
    )

    original_direction = jnp.array([0.0, 0.0, 1.0])
    true_direction_norm = true_direction / (jnp.linalg.norm(true_direction) + 1e-8)
    rotation_axis = jnp.cross(original_direction, true_direction_norm)
    axis_norm = jnp.linalg.norm(rotation_axis)
    rotation_axis = jnp.where(
        axis_norm < 1e-6,
        jnp.array([1.0, 0.0, 0.0]),
        rotation_axis / (axis_norm + 1e-8)
    )
    rotation_angle = jnp.arccos(jnp.clip(
        jnp.dot(original_direction, true_direction_norm), -1.0, 1.0
    ))

    photon_data['rotation_axis'] = rotation_axis
    photon_data['rotation_angle'] = rotation_angle
    photon_data['apply_rotation'] = jnp.array(True)
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


# =============================================================================
# OPTIMIZATION STAGES
# =============================================================================
def run_stage_1_position_search(hit_detector_positions, observed_times, observed_counts,
                                 true_position, TRUE_T0, initial_t0, detector_bounds, config):
    """Stage 1: Hierarchical grid search for position + t0"""
    pos_results = hierarchical_position_grid_search(
        hit_detector_positions, observed_times, observed_counts,
        true_position, TRUE_T0, initial_t0, detector_bounds,
        n_div=config['position_grid_search']['pos_n_div'],
        t0_n_div=config['position_grid_search']['t0_n_div'],
        levels=config['position_grid_search']['pos_levels'],
        fraction=config['position_grid_search']['pos_fraction'],
        t0_min=config['position_grid_search']['t0_min'],
        t0_max=config['position_grid_search']['t0_max'],
        min_L=config['position_grid_search']['pos_min_L'],
        verbosity=0
    )
    return pos_results


def run_stage_2_direction_search(optimal_position, optimal_t0, energy_guess,
                                  observed_counts, true_direction, config, prediction_simulator):
    """Stage 2: Hierarchical cone-based direction search"""
    levels = config['cone_direction_search']['cone_levels']
    initial_div = config['cone_direction_search']['cone_initial_div']
    max_angle_deg = config['cone_direction_search']['cone_max_angle_deg']
    reduction = config['cone_direction_search']['cone_reduction']

    best_direction = np.array([0., 0., 1.])
    best_theta = 0.
    best_phi = 0.
    best_loss = float('inf')

    cone_key = jax.random.PRNGKey(42)
    current_max_angle = np.radians(max_angle_deg)

    def rotation_matrix_from_vectors(vec1, vec2):
        a = vec1 / (np.linalg.norm(vec1) + 1e-8)
        b = vec2 / (np.linalg.norm(vec2) + 1e-8)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.linalg.norm(v) < 1e-8:
            if c > 0:
                return np.eye(3)
            else:
                perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
                perp = perp - np.dot(perp, a) * a
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                return 2 * np.outer(perp, perp) - np.eye(3)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s + 1e-8))

    for level in range(levels):
        n_theta = initial_div
        n_phi = initial_div * 2

        if level == 0:
            for i in range(n_theta):
                theta_val = np.pi * (i / max(n_theta - 1, 1))
                for j in range(n_phi):
                    phi_val = 2 * np.pi * (j / n_phi)
                    direction = spherical_to_cartesian(theta_val, phi_val)
                    track = ParticleParams(energy=energy_guess, position=optimal_position,
                                           theta=theta_val, phi=phi_val, t0=optimal_t0)
                    _, _, _, total_charge = prediction_simulator(track, cone_key)
                    loss = poisson_nll(observed_counts, total_charge)
                    if loss < best_loss:
                        best_loss = loss
                        best_direction = np.array(direction)
                        best_theta = theta_val
                        best_phi = phi_val
        else:
            z_axis = np.array([0., 0., 1.])
            R = rotation_matrix_from_vectors(z_axis, best_direction)
            for i in range(n_theta):
                cone_theta = current_max_angle * (i / max(n_theta - 1, 1))
                for j in range(n_phi):
                    cone_phi = 2 * np.pi * (j / n_phi)
                    local_dir = np.array([
                        np.sin(cone_theta) * np.cos(cone_phi),
                        np.sin(cone_theta) * np.sin(cone_phi),
                        np.cos(cone_theta)
                    ])
                    direction = R @ local_dir
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    theta_val, phi_val = cartesian_to_spherical(direction)
                    track = ParticleParams(energy=energy_guess, position=optimal_position,
                                           theta=theta_val, phi=phi_val, t0=optimal_t0)
                    _, _, _, total_charge = prediction_simulator(track, cone_key)
                    loss = poisson_nll(observed_counts, total_charge)
                    if loss < best_loss:
                        best_loss = loss
                        best_direction = np.array(direction)
                        best_theta = theta_val
                        best_phi = phi_val

        current_max_angle *= reduction

    cos_angle = np.clip(np.dot(best_direction, true_direction), -1.0, 1.0)
    direction_error = np.degrees(np.arccos(cos_angle))

    return {
        'best_direction': best_direction,
        'best_theta': float(best_theta),
        'best_phi': float(best_phi),
        'best_loss': float(best_loss),
        'direction_error': direction_error
    }


def run_stage_3_energy_scan(optimal_position, best_theta, best_phi, optimal_t0,
                             energy_guess, observed_counts, true_energy, config, prediction_simulator):
    """Stage 3: Energy scan at optimal position/direction"""
    energy_delta = config['energy_optimization']['energy_delta']
    n_steps = config['energy_optimization']['energy_scan_steps']

    energies = jnp.linspace(energy_guess - energy_delta, energy_guess + energy_delta, n_steps)

    best_loss = float('inf')
    best_energy = energy_guess
    scan_key = jax.random.PRNGKey(42)

    for energy in energies:
        track = ParticleParams(energy=energy, position=optimal_position,
                               theta=best_theta, phi=best_phi, t0=optimal_t0)
        _, _, _, total_charge = prediction_simulator(track, scan_key)
        loss = poisson_nll(observed_counts, total_charge)

        if loss < best_loss:
            best_loss = loss
            best_energy = energy

    return {
        'best_energy': float(best_energy),
        'best_loss': float(best_loss)
    }


def run_stage_4_adam_optimization(initial_params, observed_times, observed_counts,
                                   true_energy, true_position, true_direction, TRUE_T0,
                                   combined_grad_fn, config, detector_bounds):
    """Stage 4: Adam optimizer refinement with dynamic tau_vtx"""
    DETECTOR_R = detector_bounds['r']
    DETECTOR_H = detector_bounds['H']

    ADAM_LEARNING_RATE = config['adam_optimizer']['learning_rate']
    ADAM_B1 = config['adam_optimizer']['b1']
    ADAM_B2 = config['adam_optimizer']['b2']
    ADAM_EPS = config['adam_optimizer']['eps']
    MAX_ITERATIONS = 400
    damping_factor = config['optimization_params']['damping_factor']
    tolerance = 1e-6

    POS_LR_SCALE = config['learning_rates']['position_learning_rate'] * 2.
    DIR_LR_SCALE = config['learning_rates']['direction_learning_rate'] * 5.
    T0_LR_SCALE = config['learning_rates']['t0_learning_rate']
    ENE_LR_SCALE = config['learning_rates']['energy_learning_rate']

    optimizer = optax.adam(learning_rate=ADAM_LEARNING_RATE, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS)
    opt_state = optimizer.init(initial_params)
    current_params = jnp.array(initial_params)

    opt_key = jax.random.PRNGKey(12345)
    grad_norm = float('inf')

    final_combined_loss = float('inf')
    final_sqrt_ct = float('inf')
    final_tau_vtx = 0.0

    for iteration in range(MAX_ITERATIONS):
        opt_key, _ = jax.random.split(opt_key)

        (combined_loss, (charge_loss_val, time_loss_val, vtx_loss_val, sqrt_ct_val, tau_vtx_val)), grad = combined_grad_fn(
            current_params, observed_times, observed_counts, opt_key
        )

        final_combined_loss = float(combined_loss)
        final_sqrt_ct = float(sqrt_ct_val)
        final_tau_vtx = float(tau_vtx_val)

        if jnp.any(jnp.isnan(grad)):
            grad = jnp.nan_to_num(grad, nan=0.0)

        grad_norm = jnp.linalg.norm(grad)
        if grad_norm < tolerance:
            break

        if iteration < 25:
            update_scales = jnp.array([0, 0, 0, 0, DIR_LR_SCALE, DIR_LR_SCALE, 0.])
        else:
            update_scales = jnp.array([
                POS_LR_SCALE, POS_LR_SCALE, POS_LR_SCALE,
                T0_LR_SCALE, DIR_LR_SCALE, DIR_LR_SCALE, ENE_LR_SCALE
            ])

        updates, opt_state = optimizer.update(grad, opt_state, current_params)
        scaled_updates = updates * update_scales
        current_params = optax.apply_updates(current_params, scaled_updates)

        current_params = jnp.array([
            jnp.clip(current_params[0], -DETECTOR_R * 0.95, DETECTOR_R * 0.95),
            jnp.clip(current_params[1], -DETECTOR_R * 0.95, DETECTOR_R * 0.95),
            jnp.clip(current_params[2], -DETECTOR_H/2 * 0.95, DETECTOR_H/2 * 0.95),
            jnp.clip(current_params[3], -20.0, 20.0),
            current_params[4],
            current_params[5],
            jnp.clip(current_params[6], 300.0, 2000.0)
        ])

    # Calculate final errors
    final_position = current_params[:3]
    final_t0 = current_params[3]
    final_theta = current_params[4]
    final_phi = current_params[5]
    final_energy = current_params[6]
    final_direction = spherical_to_cartesian(final_theta, final_phi)

    final_position_error = float(jnp.linalg.norm(final_position - true_position))
    final_t0_error = float(abs(final_t0 - TRUE_T0))
    final_energy_error = float(abs(final_energy - true_energy))
    cos_angle = np.clip(np.dot(np.array(final_direction), np.array(true_direction)), -1.0, 1.0)
    final_direction_error = float(np.degrees(np.arccos(cos_angle)))

    return {
        'final_position_error': final_position_error,
        'final_direction_error': final_direction_error,
        'final_t0_error': final_t0_error,
        'final_energy_error': final_energy_error,
        'final_combined_loss': final_combined_loss,
        'final_sqrt_ct': final_sqrt_ct,
        'final_tau_vtx': final_tau_vtx,
        'final_energy': float(final_energy)
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction with learned tau_vtx parametrization')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--n-events', type=int, default=50, help='Number of events per combination (default: 50)')
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_events = args.n_events
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = output_dir / f'eval_tau_param_{timestamp}.csv'

    # Count total combinations
    total_combinations = len(EVAL_GRID)

    print("=" * 80)
    print("EVALUATION WITH LEARNED TAU_VTX PARAMETRIZATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print(f"CSV file: {csv_file}")
    print(f"N events per combination: {n_events}")
    print(f"N bootstrap samples: {N_BOOTSTRAP}")
    print()
    print("Parametrization: tau_vtx = a * Nrays + b * E + c")
    print(f"  a = {TAU_VTX_PARAM_A:.6e}")
    print(f"  b = {TAU_VTX_PARAM_B:.6e}")
    print(f"  c = {TAU_VTX_PARAM_C:.4f}")
    print()
    print(f"Evaluation grid:")
    for nrays, energy in EVAL_GRID:
        expected_tau = get_optimal_tau_vtx(nrays, energy)
        print(f"  Nrays={nrays:,}, E={energy}MeV -> tau_vtx ~ {expected_tau:.3f}")
    print()
    print(f"Total combinations: {total_combinations}")
    print(f"Total events to process: {total_combinations * n_events}")
    print("=" * 80)
    print()

    # Load optimization config
    config = load_optimization_config(OPT_CONFIG)

    # Add default values
    if 'optimization_params' not in config:
        config['optimization_params'] = {}
    config['optimization_params'].setdefault('damping_factor', 0.998)

    if 'adam_optimizer' not in config:
        config['adam_optimizer'] = {}
    config['adam_optimizer'].setdefault('learning_rate', 0.2)
    config['adam_optimizer'].setdefault('b1', 0.9)
    config['adam_optimizer'].setdefault('b2', 0.999)
    config['adam_optimizer'].setdefault('eps', 1e-8)

    # Setup detector (only once, doesn't depend on Nrays)
    print("Setting up detector...")
    detector = generate_detector(GEOM_CONFIG)
    detector_points = jnp.array(detector.all_points)
    NUM_DETECTORS = len(detector_points)
    detector_bounds = get_detector_bounds(detector)

    print(f"Detector: {detector_bounds['type']}, R={detector_bounds['r']:.2f}m, H={detector_bounds['H']:.2f}m")
    print(f"Number of sensors: {NUM_DETECTORS}")
    print()

    # Load range parametrization
    range_params = load_range_params('muon', 'water')

    # Initialize CSV with headers
    csv_headers = [
        'nrays', 'energy_MeV', 'n_events',
        'pos_err_68', 'pos_err_68_lo', 'pos_err_68_hi', 'pos_err_68_se',
        't0_err_68', 't0_err_68_lo', 't0_err_68_hi', 't0_err_68_se',
        'dir_err_68', 'dir_err_68_lo', 'dir_err_68_hi', 'dir_err_68_se',
        'E_err_68', 'E_err_68_lo', 'E_err_68_hi', 'E_err_68_se',
        'loss_combined', 'loss_sqrt_ct',
        'tau_vtx_mean', 'tau_vtx_std',  # Track what tau values were actually used
        'reco_energy_mean', 'reco_energy_std'
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    print(f"Initialized CSV with headers")
    print()

    # Bootstrap RNG
    bootstrap_rng = np.random.default_rng(42)

    # Main evaluation loop - group by Nrays to minimize simulator reloading
    combination_idx = 0
    current_nrays = None
    prediction_simulator = None
    data_simulator = None

    for nrays, energy_mev in EVAL_GRID:
        # Setup simulators when Nrays changes
        if nrays != current_nrays:
            print(f"Setting up simulators for Nrays={nrays:,}...")
            prediction_simulator = setup_event_simulator(
                GEOM_CONFIG, nrays, TEMPERATURE,
                max_candidates_per_ray=4, K=K, is_data=False,
                physics_config=PHYSICS_CONFIG, default_detector_params=True
            )
            data_simulator = setup_event_simulator(
                GEOM_CONFIG, nrays, temperature=0.0,
                K=12, is_data=True, is_calibration=False,
                physics_config=PHYSICS_CONFIG, default_detector_params=True
            )
            current_nrays = nrays
            print(f"Simulators ready for Nrays={nrays:,}")
            print()

        data_dir = f"{ENERGY_BASE_PATH}{energy_mev}MeV/"

        if not os.path.exists(data_dir):
            print(f"WARNING: Directory not found: {data_dir}, skipping...")
            continue

        combination_idx += 1
        expected_tau = get_optimal_tau_vtx(nrays, energy_mev)

        print(f"\n[{combination_idx}/{total_combinations}] "
              f"Nrays={nrays:,}, E={energy_mev}MeV (expected tau ~ {expected_tau:.3f})")
        print("-" * 60)

        # Storage for this combination
        position_errors = []
        t0_errors = []
        direction_errors = []
        energy_errors = []
        combined_losses = []
        sqrt_ct_losses = []
        tau_vtx_values = []
        reco_energies = []

        # Generate random keys
        main_key = jax.random.PRNGKey(42 + combination_idx * 2000)  # Different seed than scan scripts
        event_keys = jax.random.split(main_key, n_events)

        successful_events = 0

        for event_idx in tqdm(range(n_events), desc="Events", leave=False):
            try:
                # Generate event
                event_key = event_keys[event_idx]
                max_attempts = 10
                endpoint_valid = False

                for attempt in range(max_attempts):
                    event_data = generate_event_data(
                        event_idx, event_key, data_dir,
                        data_simulator, detector_bounds, detector_points, fraction=0.9
                    )

                    endpoint_valid = check_track_endpoint_in_detector(
                        event_data['true_position'],
                        event_data['true_direction'],
                        event_data['true_energy'],
                        range_params, detector_bounds, fraction=0.9
                    )

                    if endpoint_valid:
                        break
                    event_key, _ = jax.random.split(event_key)

                if not endpoint_valid:
                    continue

                # Extract event data
                true_position = event_data['true_position']
                true_direction = event_data['true_direction']
                true_energy = event_data['true_energy']
                TRUE_T0 = event_data['TRUE_T0']

                hit_mask = event_data['hit_counts'] > -999
                hit_detector_positions = detector_points[hit_mask]
                observed_times = event_data['hit_times'][hit_mask]
                observed_counts = event_data['hit_counts'][hit_mask]

                all_observed_times = event_data['hit_times']
                all_observed_counts = event_data['hit_counts']

                # Create loss function with DYNAMIC tau_vtx
                combined_grad_fn, _ = create_combined_loss_function_dynamic_tau(
                    prediction_simulator, NUM_DETECTORS, TAU_TIME, nrays, hit_detector_positions
                )

                # Stage 1: Position search
                stage1_results = run_stage_1_position_search(
                    hit_detector_positions, observed_times, observed_counts,
                    true_position, TRUE_T0, initial_t0=0.0, detector_bounds=detector_bounds, config=config
                )

                if stage1_results['best_position'] is None:
                    continue

                # Stage 2: Direction search
                stage2_results = run_stage_2_direction_search(
                    stage1_results['best_position'], stage1_results['best_t0'],
                    true_energy, all_observed_counts, true_direction, config, prediction_simulator
                )

                # Stage 3: Energy scan
                stage3_results = run_stage_3_energy_scan(
                    stage1_results['best_position'], stage2_results['best_theta'],
                    stage2_results['best_phi'], stage1_results['best_t0'],
                    true_energy, all_observed_counts, true_energy, config, prediction_simulator
                )

                # Stage 4: Adam optimization with dynamic tau
                initial_params = jnp.array([
                    stage1_results['best_position'][0],
                    stage1_results['best_position'][1],
                    stage1_results['best_position'][2],
                    stage1_results['best_t0'],
                    stage2_results['best_theta'],
                    stage2_results['best_phi'],
                    stage3_results['best_energy']
                ])

                stage4_results = run_stage_4_adam_optimization(
                    initial_params, all_observed_times, all_observed_counts,
                    true_energy, true_position, true_direction, TRUE_T0,
                    combined_grad_fn, config, detector_bounds
                )

                # Store results
                position_errors.append(stage4_results['final_position_error'])
                t0_errors.append(stage4_results['final_t0_error'])
                direction_errors.append(stage4_results['final_direction_error'])
                energy_errors.append(stage4_results['final_energy_error'])
                combined_losses.append(stage4_results['final_combined_loss'])
                sqrt_ct_losses.append(stage4_results['final_sqrt_ct'])
                tau_vtx_values.append(stage4_results['final_tau_vtx'])
                reco_energies.append(stage4_results['final_energy'])

                successful_events += 1

            except Exception as e:
                print(f"    Error in event {event_idx}: {e}")
                continue

        # Calculate 68% quantiles with bootstrap CIs
        if successful_events > 0:
            pos_boot = bootstrap_percentile_ci(position_errors, rng=bootstrap_rng)
            t0_boot = bootstrap_percentile_ci(t0_errors, rng=bootstrap_rng)
            dir_boot = bootstrap_percentile_ci(direction_errors, rng=bootstrap_rng)
            E_boot = bootstrap_percentile_ci(energy_errors, rng=bootstrap_rng)

            loss_combined_mean = np.mean(combined_losses)
            loss_sqrt_ct_mean = np.mean(sqrt_ct_losses)
            tau_vtx_mean = np.mean(tau_vtx_values)
            tau_vtx_std = np.std(tau_vtx_values)
            reco_energy_mean = np.mean(reco_energies)
            reco_energy_std = np.std(reco_energies)

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    nrays,
                    energy_mev,
                    successful_events,
                    f"{pos_boot['estimate']:.4f}",
                    f"{pos_boot['ci_lower']:.4f}",
                    f"{pos_boot['ci_upper']:.4f}",
                    f"{pos_boot['std_error']:.4f}",
                    f"{t0_boot['estimate']:.4f}",
                    f"{t0_boot['ci_lower']:.4f}",
                    f"{t0_boot['ci_upper']:.4f}",
                    f"{t0_boot['std_error']:.4f}",
                    f"{dir_boot['estimate']:.4f}",
                    f"{dir_boot['ci_lower']:.4f}",
                    f"{dir_boot['ci_upper']:.4f}",
                    f"{dir_boot['std_error']:.4f}",
                    f"{E_boot['estimate']:.4f}",
                    f"{E_boot['ci_lower']:.4f}",
                    f"{E_boot['ci_upper']:.4f}",
                    f"{E_boot['std_error']:.4f}",
                    f"{loss_combined_mean:.6f}",
                    f"{loss_sqrt_ct_mean:.6f}",
                    f"{tau_vtx_mean:.4f}",
                    f"{tau_vtx_std:.4f}",
                    f"{reco_energy_mean:.2f}",
                    f"{reco_energy_std:.2f}"
                ])

            print(f"  Results: {successful_events} events")
            print(f"    pos_68 = {pos_boot['estimate']:.3f} [{pos_boot['ci_lower']:.3f}, {pos_boot['ci_upper']:.3f}] m")
            print(f"    t0_68  = {t0_boot['estimate']:.3f} [{t0_boot['ci_lower']:.3f}, {t0_boot['ci_upper']:.3f}]")
            print(f"    dir_68 = {dir_boot['estimate']:.2f} [{dir_boot['ci_lower']:.2f}, {dir_boot['ci_upper']:.2f}] deg")
            print(f"    E_68   = {E_boot['estimate']:.1f} [{E_boot['ci_lower']:.1f}, {E_boot['ci_upper']:.1f}] MeV")
            print(f"    tau_vtx used: {tau_vtx_mean:.3f} +/- {tau_vtx_std:.3f}")
        else:
            print(f"  WARNING: No successful events for this combination")

    print()
    print("=" * 80)
    print(f"EVALUATION COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {csv_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
