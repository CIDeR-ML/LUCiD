"""
Visualization functions for LUCiD optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os

from .geometry import (
    create_cylinder_surface, create_sphere_surface, create_box_surface,
    compute_cone_cylinder_intersection, compute_cone_sphere_intersection, 
    compute_cone_box_intersection, get_cherenkov_angle
)
from ...visualization import create_detector_display

import os
import numpy as np
import plotly.graph_objects as go

from tools.optimization.utils.geometry import (
    create_cylinder_surface, create_sphere_surface, create_box_surface,
    compute_cone_cylinder_intersection, compute_cone_sphere_intersection, 
    compute_cone_box_intersection, get_cherenkov_angle
)

def spherical_to_cartesian(theta, phi):
    """Convert spherical angles to Cartesian direction vector"""
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    
    return jnp.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])

def create_full_event_3D_visualization(
    event_ID,
    all_event_results,
    sensor_positions,
    true_charges,
    true_times,
    detector_bounds,
    arrow_every_n=10,
    color_by='time',
    min_charge=10.0,
    cherenkov_angle_rad=None,
    figures_dir=None,
    detector_name="Unknown",
    show_legend_limit=8
):
    """
    Combined interactive 3D visualization:
      - optimization trajectory (markers + lines)
      - arrows at selected iterations (arrow_every_n)
      - for each arrow: compute Cherenkov cone intersection with detector geometry and plot ring/segments
      - detector shading (cylinder/sphere/box)
      - observed hits (filtered by min_charge) colored by 'time' or 'charge'
      - true & fitted tracks & origins
      - saves HTML and returns (fig, output_file_html)
    
    Parameters:
    -----------
    event_ID : int
        Index into all_event_results (same structure used by your create_3d_visualization).
    all_event_results : list/dict
        Container of event results with same keys/structure as in your posted functions.
    sensor_positions : np.ndarray (N,3)
        Positions of the sensors (m).
    true_charges : np.ndarray (N,)
    true_times   : np.ndarray (N,)
    detector_bounds : dict
        Should include 'type' key: 'cylinder'|'sphere'|'box' and geometry params:
          - cylinder: {'type':'cylinder', 'r':..., 'H':...}
          - sphere: {'type':'sphere', 'r':...}
          - box: {'type':'box', 'x':..., 'y':..., 'z':...}
    arrow_every_n : int
        Plot arrows every N trajectory steps (you said you will tune not to be crowded).
    color_by : 'time'|'charge'
    min_charge : float
        threshold to display hits
    cherenkov_angle_rad : float or None
        If None, function will compute using get_cherenkov_angle(1.33) if available.
    figures_dir : str or None
        directory to save HTML
    detector_name : str
        used in file name / title
    show_legend_limit : int
        Only show legend entries for the first N arrows (avoid huge legends).
    """
    # Basic checks
    if event_ID >= len(all_event_results):
        raise IndexError(f"Event {event_ID} not available. Max event index: {len(all_event_results)-1}")

    # Extract event result structure (compatible with your create_3d_visualization)
    event_result = all_event_results[event_ID]
    event_data = event_result['event_data']
    optimization_results = event_result['optimization_results']

    true_position = np.asarray(event_data['true_position'], dtype=float)
    true_direction = np.asarray(event_data['true_direction'], dtype=float)  # unit vector expected
    true_energy = event_data.get('true_energy', np.nan)

    # Trajectory / optimization history
    trajectory_params = np.array(optimization_results['history']['parameters'])
    trajectory_positions = trajectory_params[:, :3]            # x,y,z
    trajectory_thetas = trajectory_params[:, 4]               # theta
    trajectory_phis = trajectory_params[:, 5]                 # phi
    combined_losses = np.array(optimization_results['history']['combined_losses'])
    vertex_losses = np.array(optimization_results['history']['vertex_losses'])
    counts_losses = np.array(optimization_results['history']['counts_losses'])

    # Final / fitted result (use last item in trajectory)
    final_pos = trajectory_positions[-1]
    final_theta = trajectory_thetas[-1]
    final_phi = trajectory_phis[-1]

    # If helper available, compute cherenkov angle (fallback to 41.2 deg in water if not given)
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
        color_label = 'Hit T'
        colorscale = 'plasma'
        cmin, cmax = 0, 50
    elif color_by == 'charge':
        color_data = hit_charges_vis
        color_label = 'Hit Q'
        colorscale = 'viridis'
        cmin, cmax = 0, 50
    else:
        raise ValueError("color_by must be 'time' or 'charge'")

    # --- Build figure ---
    fig = go.Figure()

# Detector shaded surfaces / boundaries (fixed homogeneous grey)
    det_type = detector_bounds.get('type', 'cylinder')
    try:
        if det_type == 'cylinder':
            x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector_bounds['r'], detector_bounds['H'])
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                surfacecolor=np.ones_like(x_cyl),  # uniform surface
                colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                opacity=0.15,
                showscale=False,
                name='Cylinder Surface',
                hoverinfo='skip'
            ))
            # top/bottom edges ...
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
                name='Sphere Surface',
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
                i_tri, j_tri, k_tri = [], [], []
                for f in faces:
                    a,b,c,d = f
                    i_tri += [a, a]; j_tri += [b, c]; k_tri += [c, d]
                fig.add_trace(go.Mesh3d(
                    x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                    i=i_tri, j=j_tri, k=k_tri,
                    opacity=0.12,
                    color='lightgrey',
                    name='Box Surface',
                    showscale=False
                ))
    except Exception as e:
        print(f"Warning: could not add detector shading: {e}")

    # Observed hits
    fig.add_trace(go.Scatter3d(
        x=hit_positions[:, 0],
        y=hit_positions[:, 1],
        z=hit_positions[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_data,
            colorscale=colorscale,
            colorbar=dict(
                title=color_label,
                len=0.5,
                x=1.02,
                thickness=15,
            ),
            cmin=cmin, cmax=cmax,
            opacity=0.9
        ),
        name='Detector Hits',
        hovertemplate='<b>Hit Info</b><br>'
                      'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>'
                      f'{color_label}: %{{marker.color:.2f}}<extra></extra>'
    ))

    # Optimization trajectory with different colormap for losses
    x, y, z = trajectory_positions[:,0], trajectory_positions[:,1], trajectory_positions[:,2]
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+lines",
        marker=dict(
            size=4,
            color=combined_losses,
            colorscale='Cividis',
            colorbar=dict(
                title="Loss",
                len=0.5,
                x=1.08,
                thickness=15,
            ),
            showscale=True,
            opacity=0.8
        ),
        line=dict(color='gray', width=2),
        text=[f'Iteration: {i}<br>Loss: {loss:.6f}<br>'
              f'Vertex Loss: {v_loss:.6f}<br>'
              f'WC Loss: {w_loss:.6f}<br>'
              f'θ: {theta:.3f}, φ: {phi:.3f}'
              for i, (loss, v_loss, w_loss, theta, phi)
              in enumerate(zip(combined_losses, vertex_losses, counts_losses, trajectory_thetas, trajectory_phis))],
        hovertemplate='%{text}<extra></extra>',
        name="Optimization Trajectory"
    ))

    # Start and final points
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers", marker=dict(size=12, symbol="circle", color='blue', line=dict(width=3, color='darkblue')), name="Start Point"
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers", marker=dict(size=12, symbol="square", color='green', line=dict(width=3, color='darkgreen')), name="Final Point"
    ))

    # Add true track & origin
    t_vals = np.linspace(0, 8, 100)
    true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
    fig.add_trace(go.Scatter3d(
        x=true_track_points[0], y=true_track_points[1], z=true_track_points[2],
        mode='lines', line=dict(color='blue', width=8), name='True Track',
        hovertemplate='<b>True Track</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[true_position[0]], y=[true_position[1]], z=[true_position[2]],
        mode='markers', marker=dict(size=15, color='blue', symbol='diamond', line=dict(color='black', width=2)), name='True Origin',
        hovertemplate='<b>True Origin</b><br>' + f'Position: ({true_position[0]:.2f}, {true_position[1]:.2f}, {true_position[2]:.2f})<extra></extra>'
    ))

    # Add final/fitted track & origin (from final trajectory params)
    fitted_direction = spherical_to_cartesian(final_theta, final_phi)
    fitted_track_points = final_pos[:, np.newaxis] + t_vals[np.newaxis, :] * np.asarray(fitted_direction)[:, np.newaxis]
    fig.add_trace(go.Scatter3d(
        x=fitted_track_points[0], y=fitted_track_points[1], z=fitted_track_points[2],
        mode='lines', line=dict(color='red', width=8, dash='dash'), name='Fitted Track',
        hovertemplate='<b>Fitted Track</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[final_pos[0]], y=[final_pos[1]], z=[final_pos[2]],
        mode='markers', marker=dict(size=15, color='red', symbol='diamond', line=dict(color='black', width=2)), name='Fitted Origin',
        hovertemplate='<b>Fitted Origin</b><br>' + f'Position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})<extra></extra>'
    ))

    # Add multiple direction arrows along trajectory and compute/plot their cone intersections
    arrow_len = max(1.0, np.linalg.norm(trajectory_positions.max(axis=0) - trajectory_positions.min(axis=0)) * 0.05)
    arrow_indices = list(range(0, len(trajectory_positions), arrow_every_n))
    if (len(trajectory_positions) - 1) not in arrow_indices:
        arrow_indices.append(len(trajectory_positions) - 1)

    arrow_colors = ['darkblue', 'purple', 'orange', 'darkgreen', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, idx in enumerate(arrow_indices):
        if idx >= len(trajectory_positions):
            continue
        pos = trajectory_positions[idx]
        theta = trajectory_thetas[idx]
        phi = trajectory_phis[idx]
        direction = np.asarray(spherical_to_cartesian(theta, phi))
        color = arrow_colors[i % len(arrow_colors)]
        arrow_name = f"Direction at iter {idx}" if idx < len(trajectory_positions) - 1 else "Final Direction"
        showlegend = (i < show_legend_limit)

        # Arrow line
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + arrow_len * direction[0]],
            y=[pos[1], pos[1] + arrow_len * direction[1]],
            z=[pos[2], pos[2] + arrow_len * direction[2]],
            mode="lines",
            line=dict(color=color, width=6),
            name=arrow_name,
            showlegend=showlegend,
            hoverinfo='skip'
        ))

        # For each arrow, compute and plot intersection ring(s)/segments
        try:
            if det_type == 'cylinder':
                pts = compute_cone_cylinder_intersection(pos, direction, cherenkov_angle_rad, detector_bounds['r'], detector_bounds['H'])
                if len(pts) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='lines',
                        line=dict(color=color, width=4, dash='dash'),
                        name=f'Ring (iter {idx})',
                        showlegend=showlegend,
                        hovertemplate='<b>Cone-Cylinder</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                    ))
            elif det_type == 'sphere':
                pts = compute_cone_sphere_intersection(pos, direction, cherenkov_angle_rad, detector_bounds['r'])
                if len(pts) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='lines',
                        line=dict(color=color, width=4, dash='dash'),
                        name=f'Ring (iter {idx})',
                        showlegend=showlegend,
                        hovertemplate='<b>Cone-Sphere</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                    ))
            elif det_type == 'box':
                # compute_cone_box_intersection might return a list of segments (like in your function)
                segments = compute_cone_box_intersection(pos, direction, cherenkov_angle_rad,
                                                        detector_bounds['x'], detector_bounds['y'], detector_bounds['z'])
                if len(segments) > 0:
                    # segments might be list of arrays of points
                    for seg in segments:
                        seg = np.asarray(seg)
                        if seg.ndim == 2 and seg.shape[0] > 0:
                            fig.add_trace(go.Scatter3d(
                                x=seg[:,0], y=seg[:,1], z=seg[:,2],
                                mode='lines',
                                line=dict(color=color, width=4, dash='dash'),
                                name=f'RingSeg (iter {idx})',
                                showlegend=showlegend,
                                hovertemplate='<b>Cone-Box</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                            ))
        except Exception as e:
            # Do not fail the whole plot if a single intersection computation crashes
            print(f"Warning: could not compute/plot intersection for arrow idx {idx}: {e}")

    # Title with errors & improvements (try to mirror your earlier title fields)
    pos_err = optimization_results.get('final_position_error', np.nan)
    dir_err = optimization_results.get('final_direction_error', np.nan)
    energy_guess_err = event_result.get('energy_guess_error', np.nan)
    E_err = optimization_results.get('final_energy_error', np.nan)
    dir_guess_err = event_result.get('cone_direction_error', np.nan)
    pos_guess_err = event_result.get('hcp_position_error', np.nan)
    combined_loss = optimization_results.get('final_combined_loss', np.nan)

    # avoid division by zero on true_energy
    E_percent_guess = (100*energy_guess_err/true_energy) if (true_energy and not np.isnan(true_energy) and true_energy != 0) else np.nan
    E_percent_final = (100*E_err/true_energy) if (true_energy and not np.isnan(true_energy) and true_energy != 0) else np.nan

    title_text = (f"Event {event_ID}<br>"
                  f"Pos Guess Error: {pos_guess_err:.3f}m → Final Pos Error: {pos_err:.3f}m<br>"
                  f"Dir Guess Error: {dir_guess_err:.1f}° → Final Dir Error: {dir_err:.1f}°<br>"
                  f"E Guess Error: {E_percent_guess:.1f} % → Final E Error: {E_percent_final:.1f} % <br>")

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
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.85)")
    )

    # Save HTML
    filename = f'{detector_name}_full_event_{event_ID:03d}_3D_interactive.html'
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, filename)
    fig.write_html(filename)
    print(f"Full interactive 3D visualization saved to {filename}")

    return fig, filename



def create_optimization_path_3d_visualization(event_ID, all_event_results, arrow_every_n):
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
    trajectory_params = np.array(optimization_results['history']['parameters'])
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
    pos_guess_err = event_result['hcp_position_error']
    energy_guess_err = event_result['energy_guess_error']
    E_err = event_result['optimization_results']['final_energy_error']
    true_energy = event_result['event_data']['true_energy']
    
    combined_loss = optimization_results['final_combined_loss']


    title_text = (f"Event {event_ID}<br>"
                  f"Pos Guess Error: {pos_guess_err:.3f}m → Final Pos Error: {pos_err:.3f}m<br>"
                  f"Dir Guess Error: {dir_guess_err:.1f}° → Final Dir Error: {dir_err:.1f}°<br>"
                  f"E Guess Error: {100*energy_guess_err/true_energy:.1f} % → Final E Error: {100*E_err/true_energy:.1f} % <br>")
                  #f"t0 Error: {t0_err:.3f} | Loss: {combined_loss:.6f}")


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

    # Save + show
    filename = f"grid_search_optimization_3d_event_{event_ID}.html"
    fig.write_html(filename)
    print(f"Enhanced 3D visualization saved to {filename}")
    fig.show()

    return fig













def create_interactive_event_visualization(true_position, true_direction, true_energy, best_match, 
                                          true_charges, true_times, sensor_positions, detector_bounds,
                                          position_error, direction_error_deg, energy_error, energy_error_percent,
                                          event_idx=0, figures_dir=None, detector_name="Unknown", 
                                          color_by='time', min_charge=10.0):
    """Create interactive 3D visualization using plotly.
    
    Parameters:
    -----------
    color_by : str, optional
        What to use for coloring sensor hits. Options are 'time' or 'charge'.
        Default is 'time'.
    min_charge : float, optional
        Minimum charge threshold for displaying hits. Default is 10.0.
    """
    # Filter hits with charge > min_charge
    significant_mask = true_charges > min_charge
    hit_positions = sensor_positions[significant_mask]
    hit_charges_vis = true_charges[significant_mask]
    hit_times_vis = true_times[significant_mask]
    
    # Create plotly figure
    fig = go.Figure()
    
    # Plot detector hits
    if color_by == 'time':
        color_data = hit_times_vis
        color_label = 'Hit Time (ns)'
        colorscale = 'plasma'
    elif color_by == 'charge':
        color_data = hit_charges_vis  
        color_label = 'Hit Charge'
        colorscale = 'viridis'
    else:
        raise ValueError(f"color_by must be 'time' or 'charge', got '{color_by}'")
    
    # Add detector hits as scatter plot
    fig.add_trace(go.Scatter3d(
        x=hit_positions[:, 0],
        y=hit_positions[:, 1], 
        z=hit_positions[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_data,
            colorscale=colorscale,
            colorbar=dict(
                title=color_label,
                len=0.5,  # Make colorbar shorter (50% of default length)
                x=1.02,   # Move colorbar further right to avoid legend overlap
                thickness=15  # Make colorbar thinner
            ),
            cmin=0,    # Set minimum to 0 (same as matplotlib)
            cmax=50,   # Set maximum to 50 (same as matplotlib)
            opacity=0.8
        ),
        name='Detector Hits',
        hovertemplate='<b>Hit Info</b><br>' +
                     'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                     f'{color_label}: %{{marker.color:.2f}}<br>' +
                     '<extra></extra>'
    ))
    
    # Add true track
    t_vals = np.linspace(0, 8, 100)
    true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
    
    fig.add_trace(go.Scatter3d(
        x=true_track_points[0],
        y=true_track_points[1],
        z=true_track_points[2], 
        mode='lines',
        line=dict(color='blue', width=8),
        name='True Track',
        hovertemplate='<b>True Track</b><br>' +
                     'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                     '<extra></extra>'
    ))
    
    # Add true origin point
    fig.add_trace(go.Scatter3d(
        x=[true_position[0]],
        y=[true_position[1]],
        z=[true_position[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='blue', 
            symbol='diamond',
            line=dict(color='black', width=2)
        ),
        name='True Origin',
        hovertemplate='<b>True Origin</b><br>' +
                     f'Position: ({true_position[0]:.2f}, {true_position[1]:.2f}, {true_position[2]:.2f})<br>' +
                     f'Energy: {true_energy:.1f} MeV<br>' +
                     '<extra></extra>'
    ))
    
    # Add fitted track
    fitted_track_points = best_match['position'][:, np.newaxis] + t_vals[np.newaxis, :] * best_match['direction'][:, np.newaxis]
    
    fig.add_trace(go.Scatter3d(
        x=fitted_track_points[0],
        y=fitted_track_points[1], 
        z=fitted_track_points[2],
        mode='lines',
        line=dict(color='red', width=8, dash='dash'),
        name='Fitted Track',
        hovertemplate='<b>Fitted Track</b><br>' +
                     'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                     '<extra></extra>'
    ))
    
    # Add fitted origin point
    fig.add_trace(go.Scatter3d(
        x=[best_match['position'][0]],
        y=[best_match['position'][1]],
        z=[best_match['position'][2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond', 
            line=dict(color='black', width=2)
        ),
        name='Fitted Origin',
        hovertemplate='<b>Fitted Origin</b><br>' +
                     f'Position: ({best_match["position"][0]:.2f}, {best_match["position"][1]:.2f}, {best_match["position"][2]:.2f})<br>' +
                     f'Energy: {best_match["energy"]:.1f} MeV<br>' +
                     '<extra></extra>'
    ))
    
    # Add Cherenkov cone intersections if geometry allows
    try:
        cherenkov_angle = np.radians(41.2)  # Water refractive index ~1.33
        
        if detector_bounds['type'] == 'cylinder':
            from .geometry import compute_cone_cylinder_intersection
            
            # True cone intersection  
            true_intersection = compute_cone_cylinder_intersection(
                true_position, true_direction, cherenkov_angle,
                detector_bounds['r'], detector_bounds['H']
            )
            if len(true_intersection) > 0:
                fig.add_trace(go.Scatter3d(
                    x=true_intersection[:, 0],
                    y=true_intersection[:, 1],
                    z=true_intersection[:, 2],
                    mode='lines',
                    line=dict(color='blue', width=6),
                    name='True Cherenkov Ring',
                    hovertemplate='<b>True Cherenkov Ring</b><br>' +
                                 'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                 '<extra></extra>'
                ))
            
            # Fitted cone intersection
            fitted_intersection = compute_cone_cylinder_intersection(
                best_match['position'], best_match['direction'], cherenkov_angle,
                detector_bounds['r'], detector_bounds['H'] 
            )
            if len(fitted_intersection) > 0:
                fig.add_trace(go.Scatter3d(
                    x=fitted_intersection[:, 0],
                    y=fitted_intersection[:, 1],
                    z=fitted_intersection[:, 2],
                    mode='lines',
                    line=dict(color='red', width=6, dash='dash'),
                    name='Fitted Cherenkov Ring',
                    hovertemplate='<b>Fitted Cherenkov Ring</b><br>' +
                                 'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                                 '<extra></extra>'
                ))
            
            # Add cylinder boundary wireframe
            theta = np.linspace(0, 2*np.pi, 50)
            z_cyl = np.linspace(-detector_bounds['H']/2, detector_bounds['H']/2, 20)
            
            # Side of cylinder
            for z in [-detector_bounds['H']/2, detector_bounds['H']/2]:
                x_circle = detector_bounds['r'] * np.cos(theta)
                y_circle = detector_bounds['r'] * np.sin(theta)
                z_circle = z * np.ones_like(theta)
                
                fig.add_trace(go.Scatter3d(
                    x=x_circle, y=y_circle, z=z_circle,
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Vertical lines of cylinder
            for i in range(0, len(theta), 10):
                x_line = [detector_bounds['r'] * np.cos(theta[i])] * 2
                y_line = [detector_bounds['r'] * np.sin(theta[i])] * 2  
                z_line = [-detector_bounds['H']/2, detector_bounds['H']/2]
                
                fig.add_trace(go.Scatter3d(
                    x=x_line, y=y_line, z=z_line,
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
    except Exception as e:
        print(f"Warning: Could not add Cherenkov cone visualization: {e}")
    
    # Set layout and styling
    fig.update_layout(
        title=dict(
            text=f'{detector_name} Detector - Event {event_idx + 1} (Interactive)<br>' +
                 f'<span style="font-size:14px">Energy Error: {energy_error:.1f} MeV, ' +
                 f'Position Error: {position_error:.2f} m, ' +
                 f'Direction Error: {direction_error_deg:.1f}°</span>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)', 
            zaxis_title='Z (m)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            x=0.02,    # Move legend to left side
            y=0.98,    # Position at top
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    # Save interactive HTML
    output_file_html = f'{detector_name}_optimization_event_{event_idx + 1:03d}_3D_interactive.html'
    if figures_dir:
        output_file_html = os.path.join(figures_dir, output_file_html)
    
    fig.write_html(output_file_html)
    
    return output_file_html


def create_event_visualization(true_position, true_direction, true_energy, best_match, 
                              true_charges, true_times, sensor_positions, detector_bounds,
                              position_error, direction_error_deg, energy_error, energy_error_percent,
                              event_idx=0, figures_dir=None, verbose=True, config_file=None, color_by='time'):
    """Create visualization for a single event.
    
    Parameters:
    -----------
    color_by : str, optional
        What to use for coloring sensor hits. Options are 'time' or 'charge'.
        Default is 'time'.
    """
    # Extract detector name from config file path
    detector_name = "Unknown"
    if config_file:
        # Extract detector name from config filename
        # e.g., "config/IWCD_geom_config.json" -> "IWCD"
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
    # Filter hits with charge > 5
    min_charge = 10.
    significant_mask = true_charges > min_charge
    hit_positions = sensor_positions[significant_mask]
    hit_charges_vis = true_charges[significant_mask]
    hit_times_vis = true_times[significant_mask]
    
    if verbose:
        print(f"Creating visualization for event {event_idx + 1}...")
        print(f"Visualizing {len(hit_positions)} hits with charge > {min_charge}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot detector hits with equal size, colored by time or charge
    if color_by == 'time':
        color_data = hit_times_vis
        color_label = 'Hit Time'
        scatter_label = 'Detector Hits (colored by time)'
        colormap = 'plasma'
    elif color_by == 'charge':
        color_data = hit_charges_vis
        color_label = 'Hit Charge'
        scatter_label = 'Detector Hits (colored by charge)'
        colormap = 'viridis'
    else:
        raise ValueError(f"color_by must be 'time' or 'charge', got '{color_by}'")
    
    # Use fixed size for all sensor hits to avoid visualization issues with large charges
    scatter = ax.scatter(hit_positions[:, 0], hit_positions[:, 1], hit_positions[:, 2], 
                        c=color_data, s=30, cmap=colormap, alpha=0.7,
                        label=scatter_label, vmin=0, vmax=50)
    
    # Plot true track
    t_vals = np.linspace(0, 8, 100)
    true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
    ax.plot(true_track_points[0], true_track_points[1], true_track_points[2], 
            'blue', linewidth=5, label='True Track', zorder=30)
    ax.scatter(true_position[0], true_position[1], true_position[2], 
               c='blue', s=400, marker='*', edgecolors='black', linewidth=3, 
               label='True Origin', zorder=35)
    
    # Plot best fitted track
    fitted_track_points = best_match['position'][:, np.newaxis] + t_vals[np.newaxis, :] * best_match['direction'][:, np.newaxis]
    ax.plot(fitted_track_points[0], fitted_track_points[1], fitted_track_points[2], 
            'red', linewidth=5, linestyle='--', label='Best Fitted Track', zorder=30)
    ax.scatter(best_match['position'][0], best_match['position'][1], best_match['position'][2], 
               c='red', s=400, marker='*', edgecolors='black', linewidth=3, 
               label='Best Fitted Origin', zorder=35)
    
    # Add Cherenkov cone intersections
    cherenkov_angle = get_cherenkov_angle(1.33)  # Water refractive index
    
    # Detector-specific visualization
    if detector_bounds['type'] == 'cylinder':
        # True cone intersection
        true_intersection = compute_cone_cylinder_intersection(
            true_position, true_direction, cherenkov_angle,
            detector_bounds['r'], detector_bounds['H']
        )
        if len(true_intersection) > 0:
            ax.plot(true_intersection[:, 0], true_intersection[:, 1], true_intersection[:, 2],
                   'blue', linewidth=3, alpha=0.8, label='True Cherenkov Ring', zorder=25)
        
        # Fitted cone intersection
        fitted_intersection = compute_cone_cylinder_intersection(
            best_match['position'], best_match['direction'], cherenkov_angle,
            detector_bounds['r'], detector_bounds['H']
        )
        if len(fitted_intersection) > 0:
            ax.plot(fitted_intersection[:, 0], fitted_intersection[:, 1], fitted_intersection[:, 2],
                   'red', linewidth=3, linestyle='--', alpha=0.8, label='Fitted Cherenkov Ring', zorder=25)
        
        # Add cylinder boundaries
        x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector_bounds['r'], detector_bounds['H'])
        ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.15, color='gray')
        
    elif detector_bounds['type'] == 'sphere':
        # True cone intersection
        true_intersection = compute_cone_sphere_intersection(
            true_position, true_direction, cherenkov_angle, detector_bounds['r']
        )
        if len(true_intersection) > 0:
            ax.plot(true_intersection[:, 0], true_intersection[:, 1], true_intersection[:, 2],
                   'blue', linewidth=3, alpha=0.8, label='True Cherenkov Ring', zorder=25)
        
        # Fitted cone intersection
        fitted_intersection = compute_cone_sphere_intersection(
            best_match['position'], best_match['direction'], cherenkov_angle, detector_bounds['r']
        )
        if len(fitted_intersection) > 0:
            ax.plot(fitted_intersection[:, 0], fitted_intersection[:, 1], fitted_intersection[:, 2],
                   'red', linewidth=3, linestyle='--', alpha=0.8, label='Fitted Cherenkov Ring', zorder=25)
        
        # Add sphere boundaries
        x_sph, y_sph, z_sph = create_sphere_surface(detector_bounds['r'])
        ax.plot_surface(x_sph, y_sph, z_sph, alpha=0.15, color='gray')
        
    elif detector_bounds['type'] == 'box':
        # True cone intersection
        true_intersection = compute_cone_box_intersection(
            true_position, true_direction, cherenkov_angle,
            detector_bounds['x'], detector_bounds['y'], detector_bounds['z']
        )
        if len(true_intersection) > 0:
            # For box, we might get multiple segments
            for segment in true_intersection:
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                       'blue', linewidth=3, alpha=0.8, zorder=25)
        
        # Fitted cone intersection
        fitted_intersection = compute_cone_box_intersection(
            best_match['position'], best_match['direction'], cherenkov_angle,
            detector_bounds['x'], detector_bounds['y'], detector_bounds['z']
        )
        if len(fitted_intersection) > 0:
            for segment in fitted_intersection:
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                       'red', linewidth=3, linestyle='--', alpha=0.8, zorder=25)
        
        # Add box boundaries
        vertices, edges = create_box_surface(detector_bounds['x'], detector_bounds['y'], detector_bounds['z'])
        for edge in edges:
            ax.plot3D(*vertices[edge].T, 'gray', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'{detector_name} Detector - Event {event_idx + 1}\n' +
                f'Energy Error: {energy_error:.1f} MeV, ' +
                f'Position Error: {position_error:.2f} m, ' +
                f'Direction Error: {direction_error_deg:.1f}°',
                fontsize=14)
    
    # Add colorbar and legend
    plt.colorbar(scatter, ax=ax, label=color_label, shrink=0.6)
    ax.legend(loc='upper right', fontsize=10)
    
    # Set axis limits based on detector type with equal aspect ratio
    if detector_bounds['type'] == 'cylinder':
        max_extent = max(detector_bounds['r'], detector_bounds['H']/2) * 1.2
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_zlim([-max_extent, max_extent])
    elif detector_bounds['type'] == 'sphere':
        max_extent = detector_bounds['r'] * 1.2
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_zlim([-max_extent, max_extent])
    elif detector_bounds['type'] == 'box':
        max_extent = max(detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2) * 1.2
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_zlim([-max_extent, max_extent])
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    
    # Save 3D figure
    output_file_3d = f'{detector_name}_optimization_event_{event_idx + 1:03d}_3D.png'
    if figures_dir:
        output_file_3d = os.path.join(figures_dir, output_file_3d)
    plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"3D visualization saved to {output_file_3d}")
    plt.close()
    
    # Create interactive HTML visualization
    try:
        output_file_html = create_interactive_event_visualization(
            true_position, true_direction, true_energy, best_match,
            true_charges, true_times, sensor_positions, detector_bounds,
            position_error, direction_error_deg, energy_error, energy_error_percent,
            event_idx, figures_dir, detector_name, color_by
        )
        if verbose:
            print(f"Interactive 3D visualization saved to {output_file_html}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not create interactive visualization: {e}")
    
    # Create and save 2D detector display if it's a cylinder detector
    if detector_bounds['type'] == 'cylinder' and config_file:
        try:
            # Create detector display function
            display_func = create_detector_display(config_file, sparse=False)
            
            # Create 2D display for charge data
            output_file_2d_charge = f'{detector_name}_optimization_event_{event_idx + 1:03d}_2D_charge.png'
            if figures_dir:
                output_file_2d_charge = os.path.join(figures_dir, output_file_2d_charge)
            
            display_func(true_charges, true_times, 
                        file_name=output_file_2d_charge, 
                        plot_time=False, 
                        log_scale=False, 
                        vmin=1.1,
                        vmax=50.)
            
            # Create 2D display for time data
            output_file_2d_time = f'{detector_name}_optimization_event_{event_idx + 1:03d}_2D_time.png'
            if figures_dir:
                output_file_2d_time = os.path.join(figures_dir, output_file_2d_time)
            
            display_func(true_charges, true_times, 
                        file_name=output_file_2d_time, 
                        plot_time=True, 
                        log_scale=False)
            
            if verbose:
                print(f"2D detector displays saved to {output_file_2d_charge} and {output_file_2d_time}")
                
        except Exception as e:
            if verbose:
                print(f"Warning: Could not create 2D detector display: {e}")
    elif verbose:
        if detector_bounds['type'] != 'cylinder':
            print(f"2D detector display only supported for cylinder detectors (current: {detector_bounds['type']})")
        else:
            print("Config file not provided, skipping 2D detector display")


def print_summary_statistics(results, total_search_time):
    """Print summary statistics for multiple events."""
    successful_results = [r for r in results if r['success']]
    n_successful = len(successful_results)
    n_total = len(results)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS ({n_successful}/{n_total} successful)")
    print(f"{'='*80}")
    
    if n_successful > 0:
        position_errors = [r['position_error'] for r in successful_results]
        direction_errors = [r['direction_error_deg'] for r in successful_results]
        energy_errors = [r['energy_error_percent'] for r in successful_results]
        final_losses = [r['final_loss'] for r in successful_results]
        
        print(f"Position Error (m):")
        print(f"  Mean: {np.mean(position_errors):.3f} ± {np.std(position_errors):.3f}")
        print(f"  Median: {np.median(position_errors):.3f}")
        print(f"  Range: [{np.min(position_errors):.3f}, {np.max(position_errors):.3f}]")
        
        print(f"\nDirection Error (degrees):")
        print(f"  Mean: {np.mean(direction_errors):.1f} ± {np.std(direction_errors):.1f}")
        print(f"  Median: {np.median(direction_errors):.1f}")
        print(f"  Range: [{np.min(direction_errors):.1f}, {np.max(direction_errors):.1f}]")
        
        print(f"\nEnergy Error (%):")
        print(f"  Mean: {np.mean(energy_errors):.1f} ± {np.std(energy_errors):.1f}")
        print(f"  Median: {np.median(energy_errors):.1f}")
        print(f"  Range: [{np.min(energy_errors):.1f}, {np.max(energy_errors):.1f}]")
        
        print(f"\nFinal Loss:")
        print(f"  Mean: {np.mean(final_losses):.2e}")
        print(f"  Median: {np.median(final_losses):.2e}")
        print(f"  Range: [{np.min(final_losses):.2e}, {np.max(final_losses):.2e}]")
    
    print(f"\nTiming:")
    print(f"  Total search time: {total_search_time:.1f} seconds")
    print(f"  Average time per event: {total_search_time/n_total:.1f} seconds")
    print(f"  Success rate: {n_successful/n_total*100:.1f}%")


def create_convergence_plots(event_histories, figures_dir=None, show_individual=True, show_statistics=True, config_file=None):
    """
    Create comprehensive visualization of multi-event optimization convergence.
    Shows parameter errors evolution during iterations for both numerical and gradient phases.
    """
    # Extract detector name from config file path
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
    if not event_histories:
        print("No convergence histories to plot.")
        return
    
    N_events = len(event_histories)
    
    # Check if we have gradient history
    has_gradient = 'gradient_loss' in event_histories[0]
    
    if has_gradient:
        # For hybrid optimization, we need to combine numerical and gradient phases
        n_numerical = len(event_histories[0]['position_error'])
        n_gradient = len(event_histories[0]['gradient_loss']) if has_gradient else 0
        n_total = n_numerical + n_gradient
    else:
        n_iterations = len(event_histories[0]['position_error'])
        n_total = n_iterations
    
    # Create figure with subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Energy Error
    if show_individual:
        for i in range(N_events):
            ax1.plot(event_histories[i]['energy_error'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_energy_errors = np.array([h['energy_error'] for h in event_histories])
        mean_energy_error = np.mean(all_energy_errors, axis=0)
        std_energy_error = np.std(all_energy_errors, axis=0)
        median_energy_error = np.median(all_energy_errors, axis=0)
        
        iterations = range(n_iterations)
        ax1.plot(iterations, mean_energy_error, 'r-', linewidth=2, label=f'Mean (N={N_events})')
        ax1.fill_between(iterations, mean_energy_error - std_energy_error, 
                        mean_energy_error + std_energy_error, alpha=0.2, color='red', label='±1σ')
        ax1.plot(iterations, median_energy_error, 'g--', linewidth=2, label='Median')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy Error (MeV)')
    ax1.set_title('Energy Error Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position Error
    if show_individual:
        for i in range(N_events):
            ax2.plot(event_histories[i]['position_error'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_position_errors = np.array([h['position_error'] for h in event_histories])
        mean_position_error = np.mean(all_position_errors, axis=0)
        std_position_error = np.std(all_position_errors, axis=0)
        median_position_error = np.median(all_position_errors, axis=0)
        
        ax2.plot(iterations, mean_position_error, 'r-', linewidth=2, label='Mean')
        ax2.fill_between(iterations, mean_position_error - std_position_error, 
                        mean_position_error + std_position_error, alpha=0.2, color='red')
        ax2.plot(iterations, median_position_error, 'g--', linewidth=2, label='Median')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Direction Error
    if show_individual:
        for i in range(N_events):
            ax3.plot(event_histories[i]['direction_error'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_direction_errors = np.array([h['direction_error'] for h in event_histories])
        mean_direction_error = np.mean(all_direction_errors, axis=0)
        std_direction_error = np.std(all_direction_errors, axis=0)
        median_direction_error = np.median(all_direction_errors, axis=0)
        
        ax3.plot(iterations, mean_direction_error, 'r-', linewidth=2, label='Mean')
        ax3.fill_between(iterations, mean_direction_error - std_direction_error, 
                        mean_direction_error + std_direction_error, alpha=0.2, color='red')
        ax3.plot(iterations, median_direction_error, 'g--', linewidth=2, label='Median')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Direction Error (degrees)')
    ax3.set_title('Direction Error Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Best Loss Evolution
    if show_individual:
        for i in range(N_events):
            ax4.plot(event_histories[i]['best_loss'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_losses = np.array([h['best_loss'] for h in event_histories])
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        median_loss = np.median(all_losses, axis=0)
        
        ax4.plot(iterations, mean_loss, 'r-', linewidth=2, label='Mean')
        ax4.fill_between(iterations, mean_loss - std_loss, 
                        mean_loss + std_loss, alpha=0.2, color='red')
        ax4.plot(iterations, median_loss, 'g--', linewidth=2, label='Median')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Best Loss')
    ax4.set_title('Loss Function Convergence')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Energy Evolution
    if show_individual:
        for i in range(N_events):
            ax5.plot(event_histories[i]['best_energy'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_energies = np.array([h['best_energy'] for h in event_histories])
        mean_energy = np.mean(all_energies, axis=0)
        std_energy = np.std(all_energies, axis=0)
        median_energy = np.median(all_energies, axis=0)
        
        ax5.plot(iterations, mean_energy, 'r-', linewidth=2, label='Mean')
        ax5.fill_between(iterations, mean_energy - std_energy, 
                        mean_energy + std_energy, alpha=0.2, color='red')
        ax5.plot(iterations, median_energy, 'g--', linewidth=2, label='Median')
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Best Energy (MeV)')
    ax5.set_title('Energy Parameter Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined Error Metric
    if show_individual:
        for i in range(N_events):
            # Normalized combined error (position/5 + direction/90 + energy/500)
            combined_error = [
                p/5.0 + d/90.0 + e/500.0 
                for p, d, e in zip(event_histories[i]['position_error'], 
                                 event_histories[i]['direction_error'],
                                 event_histories[i]['energy_error'])
            ]
            ax6.plot(combined_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_combined_errors = []
        for i in range(N_events):
            combined_error = [
                p/5.0 + d/90.0 + e/500.0 
                for p, d, e in zip(event_histories[i]['position_error'], 
                                 event_histories[i]['direction_error'],
                                 event_histories[i]['energy_error'])
            ]
            all_combined_errors.append(combined_error)
        
        combined_error_array = np.array(all_combined_errors)
        mean_combined = np.mean(combined_error_array, axis=0)
        std_combined = np.std(combined_error_array, axis=0)
        median_combined = np.median(combined_error_array, axis=0)
        
        ax6.plot(iterations, mean_combined, 'r-', linewidth=2, label='Mean')
        ax6.fill_between(iterations, mean_combined - std_combined, 
                        mean_combined + std_combined, alpha=0.2, color='red')
        ax6.plot(iterations, median_combined, 'g--', linewidth=2, label='Median')
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Normalized Combined Error')
    ax6.set_title('Combined Error Convergence')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{detector_name}_optimization_convergence.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Convergence plots saved to {output_file}")
    plt.close()


def create_hybrid_convergence_plot(event_histories, figures_dir=None, config_file=None):
    """
    Create convergence plot showing both numerical and gradient optimization phases.
    """
    # Extract detector name
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
    
    if not event_histories:
        print("No convergence histories to plot.")
        return
    
    # Check if we have gradient history
    has_gradient = 'gradient_loss' in event_histories[0]
    
    if not has_gradient:
        print("No gradient optimization history found. Use regular convergence plot.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # For each event, create combined histories
    for i, history in enumerate(event_histories):
        # Get phase lengths
        n_numerical_iterations = len(history['best_loss'])
        n_gradient_iterations = len(history['gradient_loss'])
        
        # Calculate evaluation counts
        # For numerical optimization, we need to know population size
        # We can infer it from the history or pass it as parameter
        # For now, let's assume it's stored or we use a default
        population_size = history.get('population_size', 1)  # Default to 20 if not stored
        
        # Create x-axis based on evaluations
        numerical_evals = []
        eval_count = 0
        for iter in range(n_numerical_iterations):
            eval_count += population_size
            numerical_evals.append(eval_count)
        
        # Gradient evaluations (1 per iteration)
        gradient_evals = []
        for iter in range(n_gradient_iterations):
            eval_count += 1
            gradient_evals.append(eval_count)
        
        # Loss plot (ax1)
        ax1.semilogy(numerical_evals, history['best_loss'], 'b-', alpha=0.3, linewidth=1)
        ax1.semilogy(gradient_evals, history['gradient_loss'], 'g-', alpha=0.3, linewidth=1)
        
        # Position error plot (ax2)
        numerical_pos = history['position_error']
        # Handle case where numerical_pos is empty (when numerical_iterations=0)
        if len(numerical_pos) > 0:
            gradient_pos = history.get('gradient_position_error', [numerical_pos[-1]] * n_gradient_iterations)
        else:
            gradient_pos = history.get('gradient_position_error', [])
        
        ax2.plot(numerical_evals, numerical_pos, 'b-', alpha=0.3, linewidth=1)
        ax2.plot(gradient_evals, gradient_pos, 'g-', alpha=0.3, linewidth=1)
        
        # Direction error plot (ax3)
        numerical_dir = history['direction_error']
        # Handle case where numerical_dir is empty (when numerical_iterations=0)
        if len(numerical_dir) > 0:
            gradient_dir = history.get('gradient_direction_error', [numerical_dir[-1]] * n_gradient_iterations)
        else:
            gradient_dir = history.get('gradient_direction_error', [])
        
        ax3.plot(numerical_evals, numerical_dir, 'b-', alpha=0.3, linewidth=1)
        ax3.plot(gradient_evals, gradient_dir, 'g-', alpha=0.3, linewidth=1)
        
        # Energy error plot (ax4)
        numerical_energy = history['energy_error']
        # Handle case where numerical_energy is empty (when numerical_iterations=0)
        if len(numerical_energy) > 0:
            gradient_energy = history.get('gradient_energy_error', [numerical_energy[-1]] * n_gradient_iterations)
        else:
            gradient_energy = history.get('gradient_energy_error', [])
        
        ax4.plot(numerical_evals, numerical_energy, 'b-', alpha=0.3, linewidth=1)
        ax4.plot(gradient_evals, gradient_energy, 'g-', alpha=0.3, linewidth=1)
    
    # Add phase separators and labels (after all plots are done)
    # Calculate separator position
    total_numerical_evals = n_numerical_iterations * population_size
    total_evals = total_numerical_evals + n_gradient_iterations
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=total_numerical_evals, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add phase labels
        ax.text(total_numerical_evals/2, ax.get_ylim()[1]*0.95, 'Numerical', 
                ha='center', va='top', fontsize=12, color='darkblue',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        ax.text(total_numerical_evals + n_gradient_iterations/2, ax.get_ylim()[1]*0.95, 'Gradient', 
                ha='center', va='top', fontsize=12, color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # Titles and labels
    ax1.set_title('Loss Function Convergence')
    ax1.set_xlabel('Function Evaluations')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Position Error Convergence')
    ax2.set_xlabel('Function Evaluations')
    ax2.set_ylabel('Position Error (m)')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Direction Error Convergence')
    ax3.set_xlabel('Function Evaluations')
    ax3.set_ylabel('Direction Error (degrees)')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Energy Error Convergence')
    ax4.set_xlabel('Function Evaluations')
    ax4.set_ylabel('Energy Error (MeV)')
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'{detector_name} Hybrid Optimization Convergence (N={len(event_histories)} events)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if figures_dir:
        filename = os.path.join(figures_dir, f'{detector_name}_hybrid_convergence.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Hybrid convergence plot saved to {filename}")
    else:
        plt.show()
    
    plt.close()


def create_summary_plots(results, figures_dir=None, config_file=None):
    """Create summary histogram plots for multiple events."""
    # Extract detector name from config file path
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        print("No successful results to plot.")
        return
    
    # Extract data
    position_errors = [r['position_error'] for r in successful_results]
    direction_errors = [r['direction_error_deg'] for r in successful_results]
    energy_errors = [r['energy_error_percent'] for r in successful_results]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Position error histogram
    ax1.hist(position_errors, bins=max(5, len(position_errors)//3), alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Position Error (m)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Position Error Distribution\n(μ={np.mean(position_errors):.3f}±{np.std(position_errors):.3f}m)')
    ax1.grid(True, alpha=0.3)
    
    # Direction error histogram
    ax2.hist(direction_errors, bins=max(5, len(direction_errors)//3), alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Direction Error (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Direction Error Distribution\n(μ={np.mean(direction_errors):.1f}±{np.std(direction_errors):.1f}°)')
    ax2.grid(True, alpha=0.3)
    
    # Energy error histogram
    ax3.hist(energy_errors, bins=max(5, len(energy_errors)//3), alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('Energy Error (%)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Energy Error Distribution\n(μ={np.mean(energy_errors):.1f}±{np.std(energy_errors):.1f}%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{detector_name}_optimization_summary.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSummary plots saved to {output_file}")
    plt.close()