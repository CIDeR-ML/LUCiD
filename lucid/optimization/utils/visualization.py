"""
Visualization functions for LUCiD optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os
import jax.numpy as jnp

from .geometry import (
    create_cylinder_surface, create_sphere_surface, create_box_surface,
    compute_cone_cylinder_intersection, compute_cone_sphere_intersection, 
    compute_cone_box_intersection, get_cherenkov_angle
)
from ...visualization import create_detector_display

import os
import numpy as np
import plotly.graph_objects as go

from lucid.optimization.utils.geometry import (
    create_cylinder_surface, create_sphere_surface, create_box_surface,
    compute_cone_cylinder_intersection, compute_cone_sphere_intersection, 
    compute_cone_box_intersection, get_cherenkov_angle
)

from lucid.utils import spherical_to_cartesian  # noqa: F401 — canonical location


def _params_to_array(params_list):
    """Convert a list of parameters to a 2D numpy array.

    Handles both ParticleParams namedtuples and flat arrays.
    ParticleParams layout: [pos_x, pos_y, pos_z, t0, theta, phi, energy]
    """
    if hasattr(params_list[0], 'position'):
        # ParticleParams namedtuple
        return np.array([
            [*np.asarray(p.position), float(p.t0), float(p.theta), float(p.phi), float(p.energy)]
            for p in params_list
        ])
    return np.array(params_list)


def create_event_3D_visualization(
    event_ID,
    all_event_results,
    sensor_positions,
    true_charges,
    true_times,
    detector_bounds,
    color_by='time',
    min_charge=10.0,
    cherenkov_angle_rad=None,
    figures_dir=None,
    detector_name="Unknown"
):
    """
    Simplified interactive 3D visualization showing only:
      - Observed hits (filtered by min_charge) colored by 'time' or 'charge'
      - True track & origin (blue)
      - Fitted track & origin (red)
      - Final Cherenkov cone ring intersection with detector wall
      - Detector shading (cylinder/sphere/box)
    
    Parameters:
    -----------
    event_ID : int
        Index into all_event_results.
    all_event_results : list/dict
        Container of event results with optimization data.
    sensor_positions : np.ndarray (N,3)
        Positions of the sensors (m).
    true_charges : np.ndarray (N,)
    true_times   : np.ndarray (N,)
    detector_bounds : dict
        Should include 'type' key: 'cylinder'|'sphere'|'box' and geometry params:
          - cylinder: {'type':'cylinder', 'r':..., 'H':...}
          - sphere: {'type':'sphere', 'r':...}
          - box: {'type':'box', 'x':..., 'y':..., 'z':...}
    color_by : 'time'|'charge'
    min_charge : float
        threshold to display hits
    cherenkov_angle_rad : float or None
        If None, function will compute using get_cherenkov_angle(1.33) if available.
    figures_dir : str or None
        directory to save HTML
    detector_name : str
        used in file name / title
    """
    # Basic checks
    if event_ID >= len(all_event_results):
        raise IndexError(f"Event {event_ID} not available. Max event index: {len(all_event_results)-1}")

    # Extract event result structure
    event_result = all_event_results[event_ID]
    event_data = event_result['event_data']
    optimization_results = event_result['optimization_results']

    true_position = np.asarray(event_data['true_position'], dtype=float)
    true_direction = np.asarray(event_data['true_direction'], dtype=float)  # unit vector expected
    true_energy = event_data.get('true_energy', np.nan)

    # Get final fitted parameters (last item in trajectory)
    trajectory_params = _params_to_array(optimization_results['history']['parameters'])
    final_pos = trajectory_params[-1, :3]  # x,y,z
    final_theta = trajectory_params[-1, 4]  # theta
    final_phi = trajectory_params[-1, 5]    # phi

    # Compute cherenkov angle if not provided
    if cherenkov_angle_rad is None:
        try:
            cherenkov_angle_rad = get_cherenkov_angle(1.33)
        except Exception:
            cherenkov_angle_rad = np.radians(41.2)

    # Setup hits (filter by min_charge)
    significant_mask = np.asarray(true_charges) > min_charge
    hit_positions = sensor_positions[significant_mask]
    hit_charges_vis = np.asarray(true_charges)[significant_mask]
    hit_times_vis = np.asarray(true_times)[significant_mask]

    # Choose coloring for hits
    if color_by == 'time':
        color_data = hit_times_vis
        color_label = 'Hit Time (ns)'
        colorscale = 'plasma'
        cmin, cmax = 0, 50
    elif color_by == 'charge':
        color_data = hit_charges_vis
        color_label = 'Hit Charge (p.e.)'
        colorscale = 'viridis'
        cmin, cmax = 0, 50
    else:
        raise ValueError("color_by must be 'time' or 'charge'")

    # --- Build figure ---
    fig = go.Figure()

    # 1. Detector shaded surfaces / boundaries
    det_type = detector_bounds.get('type', 'cylinder')
    try:
        if det_type == 'cylinder':
            x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector_bounds['r'], detector_bounds['H'])
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                surfacecolor=np.ones_like(x_cyl),
                colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                opacity=0.15,
                showscale=False,
                name='Detector Surface',
                hoverinfo='skip'
            ))
            # top/bottom edges
            theta = np.linspace(0, 2*np.pi, 80)
            for z_val in [-detector_bounds['H']/2, detector_bounds['H']/2]:
                x_circle = detector_bounds['r'] * np.cos(theta)
                y_circle = detector_bounds['r'] * np.sin(theta)
                z_circle = np.full_like(theta, z_val)
                fig.add_trace(go.Scatter3d(
                    x=x_circle, y=y_circle, z=z_circle,
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False, hoverinfo='skip'
                ))

        elif det_type == 'sphere':
            x_sph, y_sph, z_sph = create_sphere_surface(detector_bounds['r'])
            fig.add_trace(go.Surface(
                x=x_sph, y=y_sph, z=z_sph,
                surfacecolor=np.ones_like(x_sph),
                colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                opacity=0.12,
                showscale=False,
                name='Detector Surface',
                hoverinfo='skip'
            ))

        elif det_type == 'box':
            vertices, edges = create_box_surface(detector_bounds['x'], detector_bounds['y'], detector_bounds['z'])
            vertices = np.asarray(vertices)
            # wireframe edges
            for edge in edges:
                v = vertices[list(edge)]
                fig.add_trace(go.Scatter3d(
                    x=v[:,0], y=v[:,1], z=v[:,2],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False, hoverinfo='skip'
                ))
            # semi-transparent faces
            if vertices.shape[0] == 8:
                faces = [
                    (0,1,3,2), (4,5,7,6), (0,1,5,4),
                    (2,3,7,6), (0,2,6,4), (1,3,7,5)
                ]
                for face in faces:
                    pts = vertices[list(face)]
                    fig.add_trace(go.Mesh3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        i=[0,0], j=[1,2], k=[2,3],
                        color='lightgrey',
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    except Exception as e:
        print(f"Warning: could not render detector surface: {e}")

    # 2. Observed hits (colored by time or charge)
    fig.add_trace(go.Scatter3d(
        x=hit_positions[:, 0],
        y=hit_positions[:, 1],
        z=hit_positions[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=color_data,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=0.5,
            colorbar=dict(title=color_label, x=1.02),
            showscale=True,
            line=dict(color='black', width=0.5)
        ),
        name='Hits',
        text=[f'Q: {q:.1f}<br>T: {t:.1f} ns' for q, t in zip(hit_charges_vis, hit_times_vis)],
        hovertemplate='<b>Hit</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>%{text}<extra></extra>'
    ))

    # 3. True track & origin (blue)
    t_vals = np.linspace(0, 8, 100)
    true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
    fig.add_trace(go.Scatter3d(
        x=true_track_points[0], y=true_track_points[1], z=true_track_points[2],
        mode='lines',
        line=dict(color='blue', width=8),
        name='True Track',
        hovertemplate='<b>True Track</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[true_position[0]], y=[true_position[1]], z=[true_position[2]],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='diamond', line=dict(color='black', width=2)),
        name='True Origin',
        hovertemplate='<b>True Origin</b><br>' + 
                     f'Position: ({true_position[0]:.2f}, {true_position[1]:.2f}, {true_position[2]:.2f})<extra></extra>'
    ))

    # 4. Fitted track & origin (red)
    fitted_direction = spherical_to_cartesian(final_theta, final_phi)
    fitted_track_points = final_pos[:, np.newaxis] + t_vals[np.newaxis, :] * np.asarray(fitted_direction)[:, np.newaxis]
    fig.add_trace(go.Scatter3d(
        x=fitted_track_points[0], y=fitted_track_points[1], z=fitted_track_points[2],
        mode='lines',
        line=dict(color='red', width=8, dash='dash'),
        name='Fitted Track',
        hovertemplate='<b>Fitted Track</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[final_pos[0]], y=[final_pos[1]], z=[final_pos[2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond', line=dict(color='black', width=2)),
        name='Fitted Origin',
        hovertemplate='<b>Fitted Origin</b><br>' + 
                     f'Position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})<extra></extra>'
    ))

    # 5. True Cherenkov ring intersection (solid yellow, thick)
    width = 5
    try:
        if det_type == 'cylinder':
            pts_true = compute_cone_cylinder_intersection(
                true_position, true_direction, cherenkov_angle_rad,
                detector_bounds['r'], detector_bounds['H']
            )
            if len(pts_true) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts_true[:,0], y=pts_true[:,1], z=pts_true[:,2],
                    mode='lines',
                    line=dict(color='blue', width=width),
                    name='True Ring',
                    hovertemplate='<b>True Cherenkov Ring</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                ))
        elif det_type == 'sphere':
            pts_true = compute_cone_sphere_intersection(
                true_position, true_direction, cherenkov_angle_rad,
                detector_bounds['r']
            )
            if len(pts_true) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts_true[:,0], y=pts_true[:,1], z=pts_true[:,2],
                    mode='lines',
                    line=dict(color='blue', width=width),
                    name='True Ring',
                    hovertemplate='<b>True Cherenkov Ring</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                ))
        elif det_type == 'box':
            segments_true = compute_cone_box_intersection(
                true_position, true_direction, cherenkov_angle_rad,
                detector_bounds['x'], detector_bounds['y'], detector_bounds['z']
            )
            if len(segments_true) > 0:
                for i, seg in enumerate(segments_true):
                    seg = np.asarray(seg)
                    if seg.ndim == 2 and seg.shape[0] > 0:
                        fig.add_trace(go.Scatter3d(
                            x=seg[:,0], y=seg[:,1], z=seg[:,2],
                            mode='lines',
                            line=dict(color='blue', width=width),
                            name='True Ring' if i == 0 else None,
                            showlegend=(i == 0),
                            hovertemplate='<b>True Cherenkov Ring</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                        ))
    except Exception as e:
        print(f"Warning: could not compute/plot true Cherenkov ring intersection: {e}")

    # 6. Reconstructed Cherenkov ring intersection (dashed red, thick)
    try:
        if det_type == 'cylinder':
            pts_fitted = compute_cone_cylinder_intersection(
                final_pos, fitted_direction, cherenkov_angle_rad,
                detector_bounds['r'], detector_bounds['H']
            )
            if len(pts_fitted) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts_fitted[:,0], y=pts_fitted[:,1], z=pts_fitted[:,2],
                    mode='lines',
                    line=dict(color='red', width=width, dash='dash'),
                    name='Reco Ring',
                    hovertemplate='<b>Fitted Cherenkov Ring</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                ))
        elif det_type == 'sphere':
            pts_fitted = compute_cone_sphere_intersection(
                final_pos, fitted_direction, cherenkov_angle_rad,
                detector_bounds['r']
            )
            if len(pts_fitted) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts_fitted[:,0], y=pts_fitted[:,1], z=pts_fitted[:,2],
                    mode='lines',
                    line=dict(color='red', width=width, dash='dash'),
                    name='Reco Ring',
                    hovertemplate='<b>Fitted Cherenkov Ring</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                ))
        elif det_type == 'box':
            segments_fitted = compute_cone_box_intersection(
                final_pos, fitted_direction, cherenkov_angle_rad,
                detector_bounds['x'], detector_bounds['y'], detector_bounds['z']
            )
            if len(segments_fitted) > 0:
                for i, seg in enumerate(segments_fitted):
                    seg = np.asarray(seg)
                    if seg.ndim == 2 and seg.shape[0] > 0:
                        fig.add_trace(go.Scatter3d(
                            x=seg[:,0], y=seg[:,1], z=seg[:,2],
                            mode='lines',
                            line=dict(color='red', width=width, dash='dash'),
                            name='Reco Ring' if i == 0 else None,
                            showlegend=(i == 0),
                            hovertemplate='<b>Fitted Cherenkov Ring</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                        ))
    except Exception as e:
        print(f"Warning: could not compute/plot fitted Cherenkov ring intersection: {e}")

    # Title with reconstruction errors
    pos_err = optimization_results.get('final_position_error', np.nan)
    dir_err = optimization_results.get('final_direction_error', np.nan)
    energy_guess_err = event_result.get('energy_guess_error', np.nan)
    E_err = optimization_results.get('final_energy_error', np.nan)
    dir_guess_err = event_result.get('cone_direction_error', np.nan)
    pos_guess_err = event_result.get('grid_position_error', np.nan)

    # Compute percentage errors for energy
    E_percent_guess = (100*energy_guess_err/true_energy) if (true_energy and not np.isnan(true_energy) and true_energy != 0) else np.nan
    E_percent_final = (100*E_err/true_energy) if (true_energy and not np.isnan(true_energy) and true_energy != 0) else np.nan

    title_text = (f"Event {event_ID}<br>"
                  f"Pos Guess Error: {pos_guess_err:.3f}m → Final Pos Error: {pos_err:.3f}m<br>"
                  f"Dir Guess Error: {dir_guess_err:.1f}° → Final Dir Error: {dir_err:.1f}°<br>"
                  f"E Guess Error: {E_percent_guess:.1f}% → Final E Error: {E_percent_final:.1f}%")

    # Layout
    center_point = true_position
    detector_size = 40
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            xaxis=dict(range=[center_point[0]-detector_size, center_point[0]+detector_size]),
            yaxis=dict(range=[center_point[1]-detector_size, center_point[1]+detector_size]),
            zaxis=dict(range=[center_point[2]-detector_size, center_point[2]+detector_size]),
            aspectmode='cube'
        ),
        width=1200,
        height=900,
        title=title_text,
        font=dict(size=10),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)"
        )
    )

    # Save HTML
    filename = f'{detector_name}_event_{event_ID:03d}_3D_interactive.html'
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, filename)
    fig.write_html(filename)
    print(f"3D visualization saved to {filename}")

    return fig


def create_optimization_path_3d_visualization(event_ID, all_event_results, arrow_every_n=100, figures_dir='figures/', detector_name='Unknown'):
    """
    Create 3D visualization with multiple direction arrows and arrow heads.
    Title shows both direction and position error improvements.
    """
    if event_ID >= len(all_event_results):
        print(f"Event {event_ID} not available. Max event index: {len(all_event_results)-1}")
        return

    fig = go.Figure()

    # Extract event data
    event_result = all_event_results[event_ID]
    event_data = event_result['event_data']
    optimization_results = event_result['optimization_results']

    # True parameters
    true_position = event_data['true_position']
    true_direction = event_data['true_direction']

    # Extract optimization trajectory
    trajectory_params = _params_to_array(optimization_results['history']['parameters'])
    trajectory_positions = trajectory_params[:, :3]  # [x, y, z]
    trajectory_thetas = trajectory_params[:, 4]
    trajectory_phis = trajectory_params[:, 5]
    combined_losses = np.array(optimization_results['history']['combined_losses'])
    vertex_losses = np.array(optimization_results['history']['vertex_losses'])
    counts_losses = np.array(optimization_results['history']['counts_losses'])

    x, y, z = trajectory_positions[:, 0], trajectory_positions[:, 1], trajectory_positions[:, 2]

    # Plot optimization trajectory colored by combined loss
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+lines",
        marker=dict(
            size=4,
            color=combined_losses,
            colorscale='Viridis_r',
            colorbar=dict(title="Loss", x=1.02),
            showscale=True,
            opacity=0.8
        ),
        line=dict(color='gray', width=2),
        text=[f'Iteration: {i}<br>Loss: {loss:.6f}<br>Vertex Loss: {v_loss:.6f}<br>WC Loss: {w_loss:.6f}<br>θ: {theta:.3f}, φ: {phi:.3f}'
              for i, (loss, v_loss, w_loss, theta, phi) in enumerate(zip(combined_losses, vertex_losses, counts_losses, trajectory_thetas, trajectory_phis))],
        hovertemplate='%{text}<extra></extra>',
        name="Optimization Trajectory"
    ))

    # Mark starting point
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        marker=dict(size=12, symbol="circle", color='blue', line=dict(width=3, color='darkblue')),
        name="Start Point"
    ))

    # Mark final point
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(size=12, symbol="square", color='green', line=dict(width=3, color='darkgreen')),
        name="Final Point"
    ))

    # Add multiple direction arrows along trajectory
    arrow_len = 1.5
    arrow_indices = list(range(0, len(trajectory_positions), arrow_every_n))
    if len(trajectory_positions) - 1 not in arrow_indices:
        arrow_indices.append(len(trajectory_positions) - 1)

    arrow_colors = ['darkblue', 'purple', 'orange', 'darkgreen', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, idx in enumerate(arrow_indices):
        if idx >= len(trajectory_positions):
            continue

        pos = trajectory_positions[idx]
        theta = trajectory_thetas[idx]
        phi = trajectory_phis[idx]
        direction = spherical_to_cartesian(theta, phi)

        color = arrow_colors[i % len(arrow_colors)]
        arrow_name = f"Direction at iter {idx}" if idx < len(trajectory_positions) - 1 else "Final Direction"

        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + arrow_len * direction[0]],
            y=[pos[1], pos[1] + arrow_len * direction[1]],
            z=[pos[2], pos[2] + arrow_len * direction[2]],
            mode="lines",
            line=dict(color=color, width=6),
            name=arrow_name,
            showlegend=(i < 5)
        ))

    # True position
    fig.add_trace(go.Scatter3d(
        x=[true_position[0]], y=[true_position[1]], z=[true_position[2]],
        mode="markers",
        marker=dict(size=18, symbol="diamond", color="red", line=dict(width=3, color='darkred')),
        name="True Position"
    ))

    # True direction
    true_arrow_len = 3.0
    fig.add_trace(go.Scatter3d(
        x=[true_position[0], true_position[0] + true_arrow_len * true_direction[0]],
        y=[true_position[1], true_position[1] + true_arrow_len * true_direction[1]],
        z=[true_position[2], true_position[2] + true_arrow_len * true_direction[2]],
        mode="lines",
        line=dict(color="red", width=10),
        name="True Direction"
    ))

    # Errors for title
    pos_err = optimization_results['final_position_error']
    dir_err = optimization_results['final_direction_error']
    t0_err = optimization_results['final_t0_error']
    dir_guess_err = event_result['cone_direction_error']
    pos_guess_err = event_result['grid_position_error']
    energy_guess_err = event_result['energy_guess_error']
    E_err = event_result['optimization_results']['final_energy_error']
    true_energy = event_result['event_data']['true_energy']

    combined_loss = optimization_results['final_combined_loss']


    title_text = (f"Event {event_ID}<br>"
                  f"Pos Guess Error: {pos_guess_err:.3f}m → Final Pos Error: {pos_err:.3f}m<br>"
                  f"Dir Guess Error: {dir_guess_err:.1f}° → Final Dir Error: {dir_err:.1f}°<br>"
                  f"E Guess Error: {100*energy_guess_err/true_energy:.1f} % → Final E Error: {100*E_err/true_energy:.1f} % <br>")


    # Layout
    center_point = true_position
    detector_size = 8

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            xaxis=dict(range=[center_point[0] - detector_size, center_point[0] + detector_size]),
            yaxis=dict(range=[center_point[1] - detector_size, center_point[1] + detector_size]),
            zaxis=dict(range=[center_point[2] - detector_size, center_point[2] + detector_size]),
            aspectmode='cube'
        ),
        font=dict(size=10),
        width=1200,
        height=800,
        title=title_text,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    # Save HTML
    filename = f'{detector_name}_event_{event_ID:03d}_path_3D_interactive.html'
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, filename)
    fig.write_html(filename)
    print(f"3D path visualization saved to {filename}")

    return fig
