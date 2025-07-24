#!/usr/bin/env python3
"""
Data-like vs Prediction-like Event 3D Visualization Script

This script generates 3D visualizations comparing data-like events (using real photon 
trajectories from ROOT files) with prediction-like events (generated from physics simulation) 
using the same track parameters.

It provides two types of visualizations:
1. Scatter-based plots with track visualization
2. Disc-based plots using the detector's native visualization method

Usage:
    python data_vs_prediction_3D_visualization.py [options]

Examples:
    # Basic usage with default settings
    python data_vs_prediction_3D_visualization.py
    
    # Specify detector and ROOT file
    python data_vs_prediction_3D_visualization.py -c config/HK_geom_config.json -d data/water/muon/events.root
    
    # Choose specific entry and save figures
    python data_vs_prediction_3D_visualization.py --entry 5 --save-figs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime
import argparse

# LUCiD imports
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.generate import read_photon_data_from_photonsim
from tools.utils import spherical_to_cartesian
from tools.optimization.event_cache import get_detector_bounds, generate_random_event_params


def create_3d_event_plot(charges, times, sensor_positions, track_position, track_direction, 
                        track_energy, detector_bounds, title, color_by='charge', min_charge=5.0):
    """
    Create a 3D visualization of an event using scatter plots.
    
    Args:
        charges: Sensor charge array
        times: Sensor time array  
        sensor_positions: Sensor position array
        track_position: Track origin position
        track_direction: Track direction vector
        track_energy: Track energy
        detector_bounds: Detector boundary information
        title: Plot title
        color_by: Color hits by 'charge' or 'time'
        min_charge: Minimum charge threshold for display
        
    Returns:
        plotly figure object
    """
    # Filter hits with significant charge
    significant_mask = charges > min_charge
    hit_positions = sensor_positions[significant_mask]
    hit_charges = charges[significant_mask]
    hit_times = times[significant_mask]
    
    if len(hit_positions) == 0:
        print(f"Warning: No hits above threshold {min_charge} for {title}")
        return None
    
    # Setup color data
    if color_by == 'charge':
        color_data = hit_charges
        color_label = 'Charge'
        colorscale = 'viridis'
    else:
        color_data = hit_times
        color_label = 'Time [ns]'
        colorscale = 'plasma'
    
    # Create figure
    fig = go.Figure()
    
    # Add sensor hits
    fig.add_trace(go.Scatter3d(
        x=hit_positions[:, 0],
        y=hit_positions[:, 1], 
        z=hit_positions[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=color_data,
            colorscale=colorscale,
            opacity=0.8,
            colorbar=dict(title=color_label)
        ),
        name=f'Sensor Hits ({len(hit_positions)})',
        text=[f'Charge: {c:.1f}<br>Time: {t:.1f} ns' for c, t in zip(hit_charges, hit_times)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add track origin
    fig.add_trace(go.Scatter3d(
        x=[track_position[0]],
        y=[track_position[1]],
        z=[track_position[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Track Origin',
        text=f'Position: [{track_position[0]:.2f}, {track_position[1]:.2f}, {track_position[2]:.2f}]',
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add track direction line
    track_length = 8.0  # meters
    track_end = track_position + track_length * track_direction
    
    fig.add_trace(go.Scatter3d(
        x=[track_position[0], track_end[0]],
        y=[track_position[1], track_end[1]],
        z=[track_position[2], track_end[2]],
        mode='lines',
        line=dict(color='red', width=6),
        name='Track Direction',
        text=f'Direction: [{track_direction[0]:.3f}, {track_direction[1]:.3f}, {track_direction[2]:.3f}]',
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add detector boundary (simplified)
    if detector_bounds['type'] == 'cylinder':
        # Create cylinder outline
        theta_vals = np.linspace(0, 2*np.pi, 50)
        r = detector_bounds['r']
        h = detector_bounds['H']
        
        # Top and bottom circles
        for z_val in [-h/2, h/2]:
            fig.add_trace(go.Scatter3d(
                x=r * np.cos(theta_vals),
                y=r * np.sin(theta_vals),
                z=np.full_like(theta_vals, z_val),
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Set layout
    max_extent = max(detector_bounds.get('r', 10), detector_bounds.get('H', 20)/2) * 1.2
    
    fig.update_layout(
        title=dict(
            text=f'{title}<br>Energy: {track_energy:.1f} MeV, Active Sensors: {len(hit_positions)}',
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title='X [m]', range=[-max_extent, max_extent]),
            yaxis=dict(title='Y [m]', range=[-max_extent, max_extent]),
            zaxis=dict(title='Z [m]', range=[-max_extent, max_extent]),
            aspectmode='cube'
        ),
        width=800,
        height=700,
        showlegend=True
    )
    
    return fig


def create_comparison_plot(prediction_charges, prediction_times, data_charges, data_times,
                          sensor_positions, track_position, track_direction, track_energy,
                          detector_bounds, color_by='charge', min_charge=5.0):
    """
    Create side-by-side comparison of prediction-like and data-like events.
    """
    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Prediction-like Event', 'Data-like Event'),
        horizontal_spacing=0.05
    )
    
    # Helper function to add event to subplot
    def add_event_to_subplot(charges, times, col, event_type):
        # Filter hits
        significant_mask = charges > min_charge
        hit_positions = sensor_positions[significant_mask]
        hit_charges = charges[significant_mask]
        hit_times = times[significant_mask]
        
        if len(hit_positions) == 0:
            return
        
        # Color data
        color_data = hit_charges if color_by == 'charge' else hit_times
        colorscale = 'viridis' if color_by == 'charge' else 'plasma'
        
        # Add sensor hits
        fig.add_trace(go.Scatter3d(
            x=hit_positions[:, 0],
            y=hit_positions[:, 1],
            z=hit_positions[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_data,
                colorscale=colorscale,
                opacity=0.7
            ),
            name=f'{event_type} Hits',
            showlegend=True,
            text=[f'Charge: {c:.1f}<br>Time: {t:.1f} ns' for c, t in zip(hit_charges, hit_times)],
            hovertemplate='%{text}<extra></extra>'
        ), row=1, col=col)
        
        # Add track origin
        fig.add_trace(go.Scatter3d(
            x=[track_position[0]],
            y=[track_position[1]],
            z=[track_position[2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name=f'{event_type} Origin',
            showlegend=False
        ), row=1, col=col)
        
        # Add track direction
        track_end = track_position + 6.0 * track_direction
        fig.add_trace(go.Scatter3d(
            x=[track_position[0], track_end[0]],
            y=[track_position[1], track_end[1]],
            z=[track_position[2], track_end[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name=f'{event_type} Track',
            showlegend=False
        ), row=1, col=col)
    
    # Add both events
    add_event_to_subplot(prediction_charges, prediction_times, 1, 'Prediction')
    add_event_to_subplot(data_charges, data_times, 2, 'Data')
    
    # Update layout
    max_extent = max(detector_bounds.get('r', 10), detector_bounds.get('H', 20)/2) * 1.2
    
    scene_dict = dict(
        xaxis=dict(title='X [m]', range=[-max_extent, max_extent]),
        yaxis=dict(title='Y [m]', range=[-max_extent, max_extent]),
        zaxis=dict(title='Z [m]', range=[-max_extent, max_extent]),
        aspectmode='cube'
    )
    
    fig.update_layout(
        title=dict(
            text=f'Event Comparison - Energy: {track_energy:.1f} MeV',
            x=0.5
        ),
        scene=scene_dict,
        scene2=scene_dict,
        width=1600,
        height=700,
        showlegend=True
    )
    
    return fig


def analyze_events(prediction_charges, prediction_times, data_charges, data_times, 
                  track_position, track_direction, track_energy, min_charge=5.0):
    """
    Perform quantitative analysis of the two event types.
    """
    # Filter active sensors
    pred_active = prediction_charges > min_charge
    data_active = data_charges > min_charge
    
    pred_charges_active = prediction_charges[pred_active]
    pred_times_active = prediction_times[pred_active]
    data_charges_active = data_charges[data_active]
    data_times_active = data_times[data_active]
    
    print("\nEvent Analysis Summary")
    print("=" * 50)
    print(f"Track Parameters:")
    print(f"  Energy: {track_energy:.1f} MeV")
    print(f"  Position: [{track_position[0]:.3f}, {track_position[1]:.3f}, {track_position[2]:.3f}] m")
    print(f"  Direction: [{track_direction[0]:.3f}, {track_direction[1]:.3f}, {track_direction[2]:.3f}]")
    print()
    
    print(f"Prediction-like Event:")
    print(f"  Active sensors: {len(pred_charges_active):,}")
    print(f"  Total charge: {np.sum(pred_charges_active):.1f}")
    print(f"  Mean charge: {np.mean(pred_charges_active):.2f} ± {np.std(pred_charges_active):.2f}")
    print(f"  Charge range: [{np.min(pred_charges_active):.1f}, {np.max(pred_charges_active):.1f}]")
    print(f"  Mean time: {np.mean(pred_times_active):.1f} ± {np.std(pred_times_active):.1f} ns")
    print(f"  Time range: [{np.min(pred_times_active):.1f}, {np.max(pred_times_active):.1f}] ns")
    print()
    
    print(f"Data-like Event:")
    print(f"  Active sensors: {len(data_charges_active):,}")
    print(f"  Total charge: {np.sum(data_charges_active):.1f}")
    print(f"  Mean charge: {np.mean(data_charges_active):.2f} ± {np.std(data_charges_active):.2f}")
    print(f"  Charge range: [{np.min(data_charges_active):.1f}, {np.max(data_charges_active):.1f}]")
    print(f"  Mean time: {np.mean(data_times_active):.1f} ± {np.std(data_times_active):.1f} ns")
    print(f"  Time range: [{np.min(data_times_active):.1f}, {np.max(data_times_active):.1f}] ns")
    print()
    
    print(f"Comparison:")
    sensor_ratio = len(data_charges_active) / len(pred_charges_active) if len(pred_charges_active) > 0 else 0
    charge_ratio = np.sum(data_charges_active) / np.sum(pred_charges_active) if np.sum(pred_charges_active) > 0 else 0
    print(f"  Active sensor ratio (data/prediction): {sensor_ratio:.2f}")
    print(f"  Total charge ratio (data/prediction): {charge_ratio:.2f}")
    
    # Common sensors
    pred_indices = np.where(pred_active)[0]
    data_indices = np.where(data_active)[0]
    common_active = set(pred_indices) & set(data_indices)
    n_common = len(common_active)
    print(f"  Sensors active in both: {n_common:,}")
    print(f"  Overlap fraction: {n_common / max(len(pred_charges_active), len(data_charges_active)):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare data-like and prediction-like events with 3D visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration parameters
    parser.add_argument('-c', '--config', type=str, 
                       default='config/IWCD_geom_config.json',
                       help='Path to detector configuration file')
    parser.add_argument('-d', '--data-file', type=str,
                       default='data/water/muon/50_data_like_events.root',
                       help='Path to ROOT file with reference photons')
    parser.add_argument('--detector-type', type=str, default='Cylinder',
                       choices=['Cylinder', 'Sphere', 'Box'],
                       help='Detector geometry type')
    parser.add_argument('--entry', type=int, default=0,
                       help='Which entry to use from ROOT file')
    
    # Simulation parameters
    parser.add_argument('--photons', type=int, default=500_000,
                       help='Number of photons to simulate')
    parser.add_argument('-K', type=int, default=6,
                       help='Number of scattering iterations')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed')
    
    # Visualization options
    parser.add_argument('--min-charge', type=float, default=5.0,
                       help='Minimum charge threshold for display')
    parser.add_argument('--color-by', type=str, default='charge',
                       choices=['charge', 'time'],
                       help='Color sensor hits by charge or time')
    parser.add_argument('--save-figs', action='store_true',
                       help='Save figures to files')
    parser.add_argument('--show-scatter', action='store_true', default=True,
                       help='Show scatter plot visualizations')
    parser.add_argument('--show-discs', action='store_true', default=True,
                       help='Show disc-based visualizations')
    parser.add_argument('--dark-theme', action='store_true',
                       help='Use dark theme for disc visualizations')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use log scale for disc visualizations')
    
    args = parser.parse_args()
    
    print("Data-like vs Prediction-like Event 3D Visualization")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Detector config: {args.config}")
    print(f"  ROOT data file: {args.data_file}")
    print(f"  Entry index: {args.entry}")
    print(f"  Photons: {args.photons:,}")
    print(f"  Random seed: {args.seed}")
    
    # Setup detector
    print("\nSetting up detector...")
    detector = generate_detector(args.config)
    sensor_positions = jnp.array(detector.all_points)
    detector_bounds = get_detector_bounds(detector)
    n_sensors = len(sensor_positions)
    
    print(f"  Type: {args.detector_type}")
    print(f"  Sensors: {n_sensors:,}")
    print(f"  Bounds: {detector_bounds}")
    
    # Sensor parameters
    sensor_params = (
        jnp.array(50.0),    # scatter_length
        jnp.array(0.1),     # reflection_rate
        jnp.array(100.0),   # absorption_length
        jnp.array(0.001)    # gumbel_softmax_temperature
    )
    
    # Setup simulators
    print("\nSetting up simulators...")
    
    # Prediction simulator (regular physics simulation)
    prediction_simulator = setup_event_simulator(
        json_filename=args.config,
        max_sensors_per_cell=4,
        n_photons=args.photons,
        temperature=0.05,
        K=args.K,
        detector_type=args.detector_type,
        is_data=False
    )
    
    # Data simulator (transforms reference photons)
    data_simulator = setup_event_simulator(
        json_filename=args.config,
        max_sensors_per_cell=4,
        n_photons=args.photons,
        temperature=0.0,  # Zero temperature for data mode
        K=args.K,
        detector_type=args.detector_type,
        is_data=True
    )
    
    # Load photon data from ROOT file
    print(f"\nLoading reference photons from ROOT file...")
    photon_data = read_photon_data_from_photonsim(args.data_file, args.entry)
    photon_data['N'] = len(photon_data['photon_origins'])
    
    print(f"  Number of photons: {photon_data['N']:,}")
    print(f"  Primary energy: {photon_data['energy']:.1f} MeV")
    
    # Generate track parameters
    print("\nGenerating track parameters...")
    key = jax.random.PRNGKey(args.seed)
    track_position, track_direction, _ = generate_random_event_params(key, detector_bounds)
    track_energy = photon_data['energy']
    
    print(f"  Position: [{track_position[0]:.3f}, {track_position[1]:.3f}, {track_position[2]:.3f}] m")
    print(f"  Direction: [{track_direction[0]:.3f}, {track_direction[1]:.3f}, {track_direction[2]:.3f}]")
    print(f"  Energy: {track_energy:.1f} MeV")
    
    # Convert direction for prediction simulator
    theta = jnp.arccos(jnp.clip(track_direction[2], -1.0, 1.0))
    phi = jnp.arctan2(track_direction[1], track_direction[0])
    direction_angles = jnp.array([theta, phi])
    
    # Simulate events
    print("\nSimulating events...")
    event_key = jax.random.PRNGKey(args.seed + 1000)
    
    # Prediction-like event
    print("  Generating prediction-like event...")
    prediction_params = (track_energy, track_position, direction_angles)
    prediction_charges, prediction_times = prediction_simulator(prediction_params, sensor_params, event_key)
    
    # Data-like event
    print("  Generating data-like event...")
    data_params = (track_energy, track_position, track_direction)
    data_charges, data_times = data_simulator(data_params, sensor_params, event_key, photon_data)
    
    # Analysis
    analyze_events(prediction_charges, prediction_times, data_charges, data_times,
                  track_position, track_direction, track_energy, args.min_charge)
    
    # Create output directory (always needed since we always save)
    output_dir = Path('figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detector_name = Path(args.config).stem.replace('_geom_config', '')
    
    # Scatter plot visualizations
    if args.show_scatter:
        print("\nCreating scatter plot visualizations...")
        
        # Individual plots
        fig_pred = create_3d_event_plot(
            prediction_charges, prediction_times, sensor_positions,
            track_position, track_direction, track_energy, detector_bounds,
            title="Prediction-like Event (Physics Simulation)",
            color_by=args.color_by, min_charge=args.min_charge
        )
        
        fig_data = create_3d_event_plot(
            data_charges, data_times, sensor_positions,
            track_position, track_direction, track_energy, detector_bounds,
            title="Data-like Event (Reference Photon Transformation)",
            color_by=args.color_by, min_charge=args.min_charge
        )
        
        # Comparison plot
        fig_comp = create_comparison_plot(
            prediction_charges, prediction_times, data_charges, data_times,
            sensor_positions, track_position, track_direction, track_energy,
            detector_bounds, color_by=args.color_by, min_charge=args.min_charge
        )
        
        # Always save scatter plots as HTML files
        if fig_pred:
            filename = output_dir / f'{detector_name}_prediction_scatter_{timestamp}.html'
            fig_pred.write_html(str(filename))
            print(f"  Saved: {filename}")
        
        if fig_data:
            filename = output_dir / f'{detector_name}_data_scatter_{timestamp}.html'
            fig_data.write_html(str(filename))
            print(f"  Saved: {filename}")
        
        filename = output_dir / f'{detector_name}_comparison_scatter_{timestamp}.html'
        fig_comp.write_html(str(filename))
        print(f"  Saved: {filename}")
        
        if not args.save_figs:
            print("  Note: Scatter plots saved as HTML files (plotly doesn't display in terminal)")
            print("  Open the HTML files in your browser to view the plots")
    
    # Disc-based visualizations
    if args.show_discs:
        print("\nCreating disc-based visualizations...")
        
        # Get active sensor indices
        prediction_indices = np.where(prediction_charges > 0)[0]
        data_indices = np.where(data_charges > 0)[0]
        
        # Visualization parameters
        surface_color = 'black' if args.dark_theme else 'gray'
        colorscale_charge = 'inferno' if args.dark_theme else 'viridis'
        colorscale_time = 'plasma'
        
        # Charge-based visualizations
        print("  Creating charge-based disc visualizations...")
        
        # Note: The detector's visualize_event_data_plotly_discs method doesn't properly save files
        # It always calls fig.show() which outputs HTML. We'll skip disc plots for now.
        print("  Skipping disc visualizations (detector method outputs HTML to console)")
        print("  Use the notebook version for disc visualizations if needed")
    
    print("\nVisualization complete!")
    if args.save_figs:
        print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()