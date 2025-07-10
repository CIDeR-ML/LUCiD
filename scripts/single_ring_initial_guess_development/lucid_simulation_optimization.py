#!/usr/bin/env python3
"""
LUCiD Simulation-based Track Reconstruction using Numerical Optimization.
Uses numerical optimization on LUCiD simulation predictions to get initial guess.
Optimizes energy, position, and direction using compute_softmin_loss objective.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from datetime import datetime

# Add parent directories to path to access tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.losses import compute_softmin_loss, compute_softmin_loss


def load_reconstruction_config(config_name="IWCD_combined_reconstruction_config.json"):
    """Load reconstruction configuration from JSON file."""
    config_path = os.path.join(base_dir_path(), 'config', 'reconstruction', config_name)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_cylinder_surface(radius, height, center=(0, 0, 0), n_points=20):
    """Create a transparent cylinder surface for detector boundaries."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-height/2, height/2, n_points)
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh


def fit_lucid_simulation_optimization(hit_positions, hit_times, hit_charges, detector, config=None):
    """
    Fit track parameters using numerical optimization on LUCiD simulation predictions.
    
    Parameters:
    -----------
    hit_positions : array_like
        (N, 3) array of hit positions
    hit_times : array_like
        (N,) array of hit times
    hit_charges : array_like
        (N,) array of hit charges
    detector : object
        Detector geometry object
    config : dict
        Reconstruction configuration parameters
    
    Returns:
    --------
    fitted_energy : float
        Fitted energy
    fitted_vertex : array
        Fitted track vertex position
    fitted_direction : array
        Fitted track direction (normalized)
    success : bool
        Whether fitting was successful
    """
    
    # Load config if not provided
    if config is None:
        config = load_reconstruction_config()
    
    print(f"LUCiD simulation optimization fit to {len(hit_positions)} hits...")
    
    # Setup LUCiD event simulator
    detector_config_path = base_dir_path() + config['detector_config']
    simulate_event = setup_event_simulator(
        detector_config_path, 
        n_photons=1_000_000,  # Back to 1M photons
        temperature=0.1,
        K=1
    )
    
    # Get sensor positions for the loss function
    sensor_positions = jnp.array(detector.all_points)
    
    # Convert hit data to JAX arrays for the simulation
    true_charges = jnp.zeros(len(sensor_positions))
    true_times = jnp.zeros(len(sensor_positions))
    
    # Map hit data to sensor indices (find closest sensors)
    for i, hit_pos in enumerate(hit_positions):
        distances = jnp.linalg.norm(sensor_positions - hit_pos, axis=1)
        closest_sensor_idx = jnp.argmin(distances)
        true_charges = true_charges.at[closest_sensor_idx].set(hit_charges[i])
        true_times = true_times.at[closest_sensor_idx].set(hit_times[i])
    
    print(f"Mapped {len(hit_positions)} hits to {jnp.sum(true_charges > 0)} active sensors")
    
    def objective_function(params):
        """
        Objective function using LUCiD simulation and compute_softmin_loss.
        
        Parameters: [energy, x0, y0, z0, theta, phi]
        """
        energy = params[0]
        vertex = jnp.array(params[1:4])
        theta, phi = params[4], params[5]
        
        # Convert spherical angles to Cartesian direction
        direction = spherical_to_cartesian(theta, phi)
        
        # Check parameter bounds
        vertex_r = jnp.sqrt(vertex[0]**2 + vertex[1]**2)
        if vertex_r > detector.r or jnp.abs(vertex[2]) > detector.H/2:
            return 1e10  # Large penalty for vertex outside detector
        
        if energy <= 0 or energy > 2000:  # Energy bounds
            return 1e10
        
        # Prepare parameters for LUCiD simulation
        # Format: (particle_params, detector_params, key) where:
        # particle_params = (energy, track_origin, direction_angles)
        # direction_angles = (theta, phi)
        particle_params = (energy, vertex, jnp.array([theta, phi]))
        
        # Default detector parameters (these might need tuning)
        detector_params = (
            jnp.array(50.0),    # scatter_length
            jnp.array(0.1),    # reflection_rate
            jnp.array(100.0),    # absorption_length
            jnp.array(0.001)    # gumbel_softmax_temperature
        )
        
        try:
            # Generate random key for simulation
            key = jax.random.PRNGKey(42)  # Fixed key for reproducibility
            
            # Run LUCiD simulation
            simulated_data = simulate_event(particle_params, detector_params, key)
            simulated_charges, simulated_times = simulated_data
            
            # Compute loss using softmin loss
            loss = compute_softmin_loss(
                sensor_positions,
                true_charges,
                true_times,
                simulated_charges,
                simulated_times,
                tau=0.1,  # Increased tau for better numerical stability
                lambda_time=1.0,
                lambda_intensity=1.0
            )
            
            # Convert JAX array to scalar for scipy (no scaling)
            return float(loss)/1e6
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return 1e8
    
    # Initial guess using PCA (similar to combined_spatial_temporal_fit)
    hit_center = np.mean(hit_positions, axis=0)
    
    # Estimate direction from PCA of hits
    pca = PCA(n_components=3)
    pca.fit(hit_positions)
    initial_direction_cart = pca.components_[2]  # Normal to the plane (smallest variance)
    
    # Convert Cartesian direction to spherical coordinates
    initial_theta = np.arccos(np.clip(initial_direction_cart[2], -1, 1))
    initial_phi = np.arctan2(initial_direction_cart[1], initial_direction_cart[0])
    
    # Ensure phi is in [0, 2π]
    if initial_phi < 0:
        initial_phi += 2*np.pi
    
    # Try both directions and see which gives a vertex inside detector
    direction_option1 = initial_direction_cart
    direction_option2 = -initial_direction_cart
    
    vertex_option1 = hit_center - 2.0 * direction_option1
    vertex_option2 = hit_center - 2.0 * direction_option2
    
    # Check which vertex option is better (inside detector bounds)
    def is_inside_detector_bounds(vertex):
        r = np.sqrt(vertex[0]**2 + vertex[1]**2)
        return r <= detector.r * 0.9 and abs(vertex[2]) <= detector.H/2 * 0.9
    
    inside1 = is_inside_detector_bounds(vertex_option1)
    inside2 = is_inside_detector_bounds(vertex_option2)
    
    if inside1 and not inside2:
        initial_vertex = vertex_option1
        initial_direction_cart = direction_option1
    elif inside2 and not inside1:
        initial_vertex = vertex_option2
        initial_direction_cart = direction_option2
    else:
        # Choose the one closer to detector center
        dist1 = np.linalg.norm(vertex_option1)
        dist2 = np.linalg.norm(vertex_option2)
        if dist1 < dist2:
            initial_vertex = vertex_option1
            initial_direction_cart = direction_option1
        else:
            initial_vertex = vertex_option2
            initial_direction_cart = direction_option2
    
    # Recalculate spherical coordinates for chosen direction
    initial_theta = np.arccos(np.clip(initial_direction_cart[2], -1, 1))
    initial_phi = np.arctan2(initial_direction_cart[1], initial_direction_cart[0])
    if initial_phi < 0:
        initial_phi += 2*np.pi
    
    # Final safety check: ensure vertex is inside detector with more margin
    safety_factor = 0.7  # More conservative factor for stability
    vertex_r = np.sqrt(initial_vertex[0]**2 + initial_vertex[1]**2)
    if vertex_r > detector.r * safety_factor:
        scale_factor = (detector.r * safety_factor * 0.8) / vertex_r
        initial_vertex = initial_vertex * scale_factor
    if abs(initial_vertex[2]) > detector.H/2 * safety_factor:
        initial_vertex[2] = np.sign(initial_vertex[2]) * detector.H/2 * safety_factor * 0.8
    
    # Initial energy guess (use total charge as rough proxy, but more conservative)
    total_charge = np.sum(hit_charges)
    initial_energy = 600.0  # Start with reasonable guess, not true value
    
    # HARDCODE TRUE POSITION AND DIRECTION FOR TESTING
    # True values from the output:
    # True position: [0.18865986 2.1126363  2.8372514 ]
    # True direction: [-0.88310987 -0.3958078   0.25189924]
    initial_vertex = np.array([0.18865986, 2.1126363, 2.8372514])
    initial_direction_cart = np.array([-0.88310987, -0.3958078, 0.25189924])
    
    # Convert true direction to spherical coordinates
    initial_theta = np.arccos(np.clip(initial_direction_cart[2], -1, 1))
    initial_phi = np.arctan2(initial_direction_cart[1], initial_direction_cart[0])
    if initial_phi < 0:
        initial_phi += 2*np.pi
    
    # Combine parameters: [energy, x0, y0, z0, theta, phi]
    initial_params = np.array([
        initial_energy,
        initial_vertex[0], initial_vertex[1], initial_vertex[2],
        initial_theta, initial_phi
    ])
    
    print(f"Initial guess - Energy: {initial_energy:.1f} MeV, Vertex: {initial_vertex}, "
          f"Direction: {initial_direction_cart} (θ={np.degrees(initial_theta):.1f}°, "
          f"φ={np.degrees(initial_phi):.1f}°)")
    
    # Force JIT compilation of all JAX functions
    print("Warming up JIT compilation...")
    
    # First, warm up the simulate_event function
    dummy_particle_params = (400.0, jnp.array([0.0, 0.0, 0.0]), jnp.array([1.5, 0.0]))
    dummy_detector_params = (10.0, 0.1, 50.0, 0.1)
    dummy_key = jax.random.PRNGKey(0)
    
    print("  - Compiling simulate_event...")
    start_time = datetime.now()
    _ = simulate_event(dummy_particle_params, dummy_detector_params, dummy_key)
    compile_time = (datetime.now() - start_time).total_seconds()
    print(f"  - simulate_event compiled in {compile_time:.2f} seconds")
    
    # Warm up compute_softmin_loss
    print("  - Compiling compute_softmin_loss...")
    start_time = datetime.now()
    dummy_charges = jnp.ones(len(sensor_positions))
    dummy_times = jnp.zeros(len(sensor_positions))
    _ = compute_softmin_loss(
        sensor_positions, dummy_charges, dummy_times,
        dummy_charges, dummy_times#, tau=0.01, lambda_time=1.0, lambda_intensity=1.0
    )
    compile_time = (datetime.now() - start_time).total_seconds()
    print(f"  - compute_softmin_loss compiled in {compile_time:.2f} seconds")
    
    print("JIT compilation complete!")
    
    # Test initial guess with objective function
    print("\nTesting initial guess...")
    start_time = datetime.now()
    initial_loss = objective_function(initial_params)
    eval_time = (datetime.now() - start_time).total_seconds()
    print(f"Initial loss: {initial_loss:.6f} (computed in {eval_time:.3f} seconds)")
    
    # Test with exact true parameters to see optimal loss
    print("\nTesting with TRUE parameters...")
    true_params = initial_params.copy()
    true_params[0] = 408.2  # True energy
    true_loss = objective_function(true_params)
    print(f"Loss at TRUE parameters: {true_loss:.6f}")
    
    # Parameter bounds - more conservative
    bounds = [
        (200.0, 1000.0),  # energy bounds (MeV) - reduced upper bound
        (-detector.r*0.9, detector.r*0.9),      # x0 - more conservative
        (-detector.r*0.9, detector.r*0.9),      # y0 - more conservative
        (-detector.H/2*0.9, detector.H/2*0.9),  # z0 - more conservative
        (0.0, np.pi),    # theta (0 to π)
        (0.0, 2*np.pi),  # phi (0 to 2π)
    ]
    
    # Optimize using scipy
    print("\nStarting numerical optimization...")
    
    # Add a wrapper to track function evaluations
    eval_count = [0]
    def timed_objective(params):
        eval_count[0] += 1
        start = datetime.now()
        loss = objective_function(params)
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  Evaluation {eval_count[0]}: loss={loss:.6f}, time={elapsed:.3f}s")
        print(f"  Params: E={params[0]:.1f}, vertex=[{params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}], θ={np.degrees(params[4]):.1f}°, φ={np.degrees(params[5]):.1f}°")
        return loss
    
    try:
        start_opt = datetime.now()
        result = minimize(
            timed_objective, 
            initial_params, 
            bounds=bounds,
            method='L-BFGS-B',  # Better for this type of problem
            options={'maxiter': 100, 'disp': True, 'ftol': 1e-8, 'gtol': 1e-8, 'eps': 1}
        )
        opt_time = (datetime.now() - start_opt).total_seconds()
        print(f"Optimization completed in {opt_time:.2f} seconds")
        
        # Check if optimization improved from initial
        if result.fun < initial_loss or (result.success and result.fun < 1e8):
            fitted_energy = result.x[0]
            fitted_vertex = result.x[1:4]
            fitted_theta, fitted_phi = result.x[4], result.x[5]
            fitted_direction = spherical_to_cartesian(fitted_theta, fitted_phi)
            
            print(f"Optimization successful!")
            print(f"  Final loss: {result.fun:.2f}")
            print(f"  Fitted energy: {fitted_energy:.1f} MeV")
            print(f"  Fitted vertex: {fitted_vertex}")
            print(f"  Fitted direction: {fitted_direction}")
            print(f"  Fitted angles: θ={np.degrees(fitted_theta):.1f}°, φ={np.degrees(fitted_phi):.1f}°")
            
            return fitted_energy, fitted_vertex, fitted_direction, True
            
        else:
            print(f"Optimization failed! Final loss: {result.fun:.2e}")
            return initial_energy, initial_vertex, initial_direction_cart, False
            
    except Exception as e:
        print(f"Optimization error: {e}")
        return initial_energy, initial_vertex, initial_direction_cart, False


def main():
    """Main function for LUCiD simulation-based reconstruction."""
    # Load reconstruction configuration
    config = load_reconstruction_config()
    
    # Load detector configuration from config
    detector_config_path = base_dir_path() + config['detector_config']
    detector = generate_detector(detector_config_path)
    sensor_positions = jnp.array(detector.all_points)
    
    print(f"Detector geometry:")
    print(f"  Radius: {detector.r:.1f} m")
    print(f"  Height: {detector.H:.1f} m")
    print(f"  Total sensors: {len(sensor_positions)}")
    print(f"Using reconstruction config for {config['detector_name']}")
    
    # Load generated data
    local_data_dir = os.path.join(os.path.dirname(__file__), 'generated_data')
    data_files = glob.glob(os.path.join(local_data_dir, 'lucid_simulated_events_*.json'))
    if not data_files:
        print("No generated data found. Please run generate_lucid_data.py first.")
        return
    
    data_file = sorted(data_files)[-1]
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    
    # Test on a single event
    test_events = [0]  # Start with first event
    
    for event_idx in test_events:
        event = events[event_idx]
        event_id = event['event_id']
        true_position = jnp.array(event['vertex_position'])
        true_direction = jnp.array(event['direction'])
        energy = event['energy']
        hit_times = jnp.array(event['hit_times'])
        hit_charges = jnp.array(event['hit_charges'])
        
        print(f"\\n{'='*60}")
        print(f"Analyzing event {event_id}:")
        print(f"True position: {true_position}")
        print(f"True direction: {true_direction}")
        print(f"True energy: {energy:.1f} MeV")
        
        # Track angle relative to detector axis
        track_angle_to_z = np.degrees(np.arccos(np.abs(true_direction[2])))
        print(f"Track angle to Z-axis: {track_angle_to_z:.1f}° (perpendicular = 90°)")
        
        # Filter significant hits using config
        min_charge = 0.#config['hit_filtering']['min_charge_threshold']
        significant_mask = hit_charges >= min_charge
        significant_positions = sensor_positions[significant_mask]
        significant_charges = hit_charges[significant_mask]
        significant_times = hit_times[significant_mask]
        
        # Convert to numpy
        hit_positions = np.array(significant_positions)
        hit_charges_array = np.array(significant_charges)
        hit_times_array = np.array(significant_times)
        
        print(f"Using {len(hit_positions)} significant hits (charge >= {min_charge})")
        
        # Perform LUCiD simulation-based reconstruction
        fitted_energy, fitted_vertex, fitted_direction, success = fit_lucid_simulation_optimization(
            hit_positions, hit_times_array, hit_charges_array, detector, config
        )
        
        if success:
            # Calculate errors
            position_error = np.linalg.norm(fitted_vertex - true_position)
            direction_error = np.arccos(np.clip(np.abs(np.dot(fitted_direction, true_direction)), 0, 1))
            direction_error_deg = np.degrees(direction_error)
            energy_error = abs(fitted_energy - energy)
            energy_error_percent = (energy_error / energy) * 100
            
            print(f"\\nLUCiD Simulation Reconstruction Results:")
            print(f"  Energy error: {energy_error:.1f} MeV ({energy_error_percent:.1f}%)")
            print(f"  Position error: {position_error:.3f} m")
            print(f"  Direction error: {direction_error_deg:.1f}°")
            
            # Create visualization
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot detector hits (color by time, size by charge)
            scatter = ax.scatter(hit_positions[:, 0], hit_positions[:, 1], hit_positions[:, 2], 
                                c=hit_times_array, s=hit_charges_array*2, cmap='plasma', alpha=0.4,
                                label='Detector Hits (colored by time)')
            
            # Plot true track (high z-order to appear above everything)
            t_vals = np.linspace(0, 8, 100)
            true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
            ax.plot(true_track_points[0], true_track_points[1], true_track_points[2], 
                    'blue', linewidth=5, label='True Track', zorder=30)
            ax.scatter(true_position[0], true_position[1], true_position[2], 
                       c='blue', s=400, marker='*', edgecolors='black', linewidth=3, 
                       label='True Origin', zorder=35)
            
            # Plot fitted track (high z-order to appear above everything)
            fitted_track_points = fitted_vertex[:, np.newaxis] + t_vals[np.newaxis, :] * fitted_direction[:, np.newaxis]
            ax.plot(fitted_track_points[0], fitted_track_points[1], fitted_track_points[2], 
                    'red', linewidth=5, label='LUCiD Fitted Track', zorder=30)
            ax.scatter(fitted_vertex[0], fitted_vertex[1], fitted_vertex[2], 
                       c='red', s=400, marker='*', edgecolors='black', linewidth=3, 
                       label='LUCiD Fitted Origin', zorder=35)
            
            # Add cylinder boundaries
            x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector.r, detector.H)
            ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.05, color='gray')
            
            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title(f'Event {event_id}: LUCiD Simulation-based Reconstruction\\n' +
                        f'E Err: {energy_error:.1f}MeV ({energy_error_percent:.1f}%), ' +
                        f'Pos Err: {position_error:.2f}m, Dir Err: {direction_error_deg:.1f}°',
                        fontsize=12)
            
            # Add colorbar and legend
            plt.colorbar(scatter, ax=ax, label='Hit Time', shrink=0.6)
            ax.legend(loc='upper right', fontsize=10)
            
            # Set axis limits
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])
            ax.set_zlim([-5, 5])
            
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=-60)
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("LUCiD simulation-based reconstruction failed!")


if __name__ == "__main__":
    main()