import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lucid.geometry import Box, Cylinder, Sphere, generate_detector
from lucid.geometry.utils import calculate_surface_normals, create_disc_mesh
from lucid.utils import sparse_to_full

def create_color_gradient(max_cnts, colormap='gnuplot'):
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=max_cnts)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def calculate_min_distance(positions):
    distances = pdist(positions)
    return np.min(distances) if len(distances) > 0 else 1.0


def create_detector_display(json_filename, sparse=True):
    """
    Create a detector display function that can handle both sparse and dense data formats.

    Parameters
    ----------
    json_filename : str
        Path to the configuration file for detector geometry
    sparse : bool
        If True, function expects sparse data format (indices, charges, times)
        If False, function expects dense data format (full arrays)

    Returns
    -------
    function
        Display function that can be called with appropriate data format
    """
    # Generate detector
    detector = generate_detector(json_filename)

    # Get geometry data from detector object
    radius = detector.r
    height = detector.H

    # Set up detector information from detector object
    sensor_positions = np.array(detector.all_points)
    sensor_cases = np.array([detector.ID_to_case[i] for i in range(len(detector.all_points))])
    n_sensors = len(sensor_positions)   # IMPORTANT: use actual number of points

    def display_detector_data(
        *args,
        file_name=None,
        plot_time=False,
        log_scale=False,
        vmin=None,
        vmax=None,
        perc_min=1,
        perc_max=99,
        facecolor='white',
        barcolor='black',
        colormap=None,
        colormap_norm=None,   # optional matplotlib norm (e.g. diverging TwoSlopeNorm); overrides the default
        zero_color=np.array([0.9, 0.9, 0.9, 1.0]),
        show_colorbar=True,
        colorbar_width='5%',
        colorbar_label=None,
        colorbar_ticks=None,   # explicit tick values (data units); overrides the auto locator
        external_legend=None,
    ):
        """
        Process and display detector data in either sparse or dense format.
        """

        # ------------------------
        # Sparse / dense handling
        # ------------------------
        if sparse:
            if len(args) != 3:
                raise ValueError("Sparse format requires three arguments: indices, charges, and times")
            loaded_indices, loaded_charges, loaded_times = args

            # Convert sparse to full arrays
            all_charges = sparse_to_full(loaded_indices, loaded_charges, n_sensors)
            all_times = sparse_to_full(loaded_indices, loaded_times, n_sensors)
        else:
            if len(args) != 2:
                raise ValueError("Dense format requires two arguments: charges and times arrays")
            all_charges, all_times = args

        # Select which values to plot based on plot_time
        all_values = all_times if plot_time else all_charges

        # ------------------------
        # vmin / vmax determination
        # ------------------------
        positive_values = all_values[all_values > 0]

        if len(positive_values) > 0:
            if vmin is None:
                vmin = np.percentile(positive_values, perc_min)
            if vmax is None:
                vmax = np.percentile(positive_values, perc_max)
        else:
            # Fallback if no positive values
            if vmin is None:
                vmin = 0.1 if log_scale else 0
            if vmax is None:
                vmax = 1

        if perc_min == 0 and (not plot_time) and (not log_scale):
            vmin = 0.001
        elif perc_min == 0 and (not plot_time) and log_scale:
            vmin = 0.1

        max_value = vmax

        # ------------------------
        # Color mapping
        # ------------------------
        if log_scale:
            plot_values = np.copy(all_values)
            plot_values[plot_values <= 0] = vmin
            plot_values = np.clip(plot_values, vmin, max_value)

            cmap = plt.get_cmap('viridis_r') if plot_time else plt.get_cmap('plasma')
            norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=max_value)
        else:
            plot_values = np.copy(all_values)
            plot_values = np.clip(plot_values, vmin, max_value)

            cmap = plt.get_cmap('viridis_r') if plot_time else plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=vmin, vmax=max_value)

        # If a custom colormap is provided, override cmap (but keep norm as-is)
        if colormap is not None:
            cmap = colormap
        # If a custom norm is provided (e.g. a diverging TwoSlopeNorm), use it verbatim.
        if colormap_norm is not None:
            norm = colormap_norm

        color_gradient = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # ------------------------
        # Geometry → 2D projection
        # ------------------------
        caps_offset = 1.05 * height / 2 + radius

        x = np.zeros(n_sensors)
        y = np.zeros(n_sensors)

        # Barrel (case 0)
        barrel_mask = sensor_cases == 0
        theta = np.arctan2(sensor_positions[barrel_mask, 1], sensor_positions[barrel_mask, 0])
        theta = (theta + np.pi * 3 / 2) % (2 * np.pi) / 2
        x[barrel_mask] = theta * radius * 2
        y[barrel_mask] = sensor_positions[barrel_mask, 2]

        # Top cap (case 1)
        top_mask = sensor_cases == 1
        x[top_mask] = sensor_positions[top_mask, 0] + np.pi * radius
        y[top_mask] = caps_offset + sensor_positions[top_mask, 1]

        # Bottom cap (case 2)
        bottom_mask = sensor_cases == 2
        x[bottom_mask] = sensor_positions[bottom_mask, 0] + np.pi * radius
        y[bottom_mask] = -caps_offset - sensor_positions[bottom_mask, 1]

        transformed_positions = np.column_stack((x, y))
        min_distance = calculate_min_distance(transformed_positions)
        circle_diameter = min_distance

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        padding = circle_diameter
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        x_range = x_max - x_min
        y_range = y_max - y_min

        fig_width = 8
        fig_height = fig_width * (y_range / x_range)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=facecolor)

        # ------------------------
        # Colors for sensors
        # ------------------------
        colors = color_gradient.to_rgba(plot_values)

        # Zero / no-hit sensors → zero_color
        zero_mask = all_values <= 0
        colors[zero_mask] = zero_color

        ells = EllipseCollection(
            widths=circle_diameter,
            heights=circle_diameter,
            angles=0,
            units='x',
            facecolors=colors,
            offsets=transformed_positions,
            transOffset=ax.transData,
            edgecolors='none'
        )

        ax.add_collection(ells)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # ------------------------
        # Optional colorbar
        # ------------------------
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=colorbar_width, pad=0.1)
            cbar = plt.colorbar(color_gradient, cax=cax)
            if colorbar_label is not None:               # explicit label -> use it verbatim
                label_text = colorbar_label
            else:
                label_text = ('Time (ns)' if plot_time else 'Photoelectron Count (a.u.)') \
                    + (' (log scale)' if log_scale else '')
            cbar.set_label(label_text, color=barcolor, fontsize=14)
            cbar.ax.yaxis.set_tick_params(color=barcolor, labelcolor=barcolor)
            cbar.outline.set_edgecolor(barcolor)

            if colorbar_ticks is not None:
                from matplotlib.ticker import FixedLocator, FormatStrFormatter
                cbar.ax.yaxis.set_major_locator(FixedLocator(list(colorbar_ticks)))
                cbar.ax.yaxis.set_minor_locator(FixedLocator([]))
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            elif log_scale:
                from matplotlib.ticker import LogLocator, LogFormatter
                cbar.ax.yaxis.set_major_locator(LogLocator(numticks=10))
                cbar.ax.yaxis.set_minor_locator(LogLocator(numticks=10, subs='auto'))
                cbar.ax.yaxis.set_major_formatter(LogFormatter())
            else:
                from matplotlib.ticker import MaxNLocator
                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            cbar.ax.tick_params(axis='y', which='both', colors=barcolor, labelcolor=barcolor)
            for label in cbar.ax.get_yticklabels():
                label.set_color(barcolor)

        # ------------------------
        # Optional external legend
        # ------------------------
        if external_legend is not None:
            ax.legend(handles=external_legend, loc='upper right', fontsize=12)

        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1,
                        facecolor=facecolor, edgecolor='none')
        plt.show()

    return display_detector_data


def unroll_layout(detector):
    """2D unrolled-cylinder layout for a detector — the canonical seam (barrel → strip, caps above/
    below). Returns ``dict(x, y, cases, diameter, xlim, ylim)`` with per-sensor 2D positions.

    The same transform `create_detector_display` uses inline; exposed module-level so animation and
    custom displays reuse it instead of re-deriving it.
    """
    radius, height = detector.r, detector.H
    pos = np.array(detector.all_points)
    cases = np.array([detector.ID_to_case[i] for i in range(len(pos))])
    n = len(pos); x = np.zeros(n); y = np.zeros(n); caps = 1.05 * height / 2 + radius
    b = cases == 0
    th = (np.arctan2(pos[b, 1], pos[b, 0]) + np.pi * 3 / 2) % (2 * np.pi) / 2
    x[b] = th * radius * 2; y[b] = pos[b, 2]
    t = cases == 1; x[t] = pos[t, 0] + np.pi * radius; y[t] = caps + pos[t, 1]
    bo = cases == 2; x[bo] = pos[bo, 0] + np.pi * radius; y[bo] = -caps - pos[bo, 1]
    xy = np.column_stack((x, y)); d = calculate_min_distance(xy)
    return dict(x=x, y=y, cases=cases, diameter=d,
                xlim=(x.min() - d, x.max() + d), ylim=(y.min() - d, y.max() + d))


def animate_event(detector, charges, times, out_path='event.gif', n_frames=40, fps=12, cmap='gnuplot'):
    """Animate a single event's PMT hits arriving over time on the 2D unrolled detector → a GIF.

    Consolidates the inline `render_frame`/`create_gif` notebook copies. ``detector`` is a detector
    object (or a geometry-JSON path); ``charges``/``times`` are per-sensor arrays. Sensors light up
    in time order (cumulative); unlit/not-yet-arrived sensors are light grey. Writes ``out_path``
    and returns it. Needs ``imageio``.
    """
    import imageio.v2 as imageio
    if isinstance(detector, str):
        detector = generate_detector(detector)
    L = unroll_layout(detector)
    charges = np.asarray(charges); times = np.asarray(times); hit = charges > 0
    if not hit.any():
        raise ValueError('no lit sensors')
    tlo, thi = np.quantile(times[hit], 0.02), np.quantile(times[hit], 0.98)
    cmax = np.quantile(charges[hit], 0.98); thresholds = np.linspace(tlo, thi, n_frames)
    sm = create_color_gradient(cmax, cmap); frames = []
    for tcut in thresholds:
        fig, ax = plt.subplots(figsize=(9, 9 * (L['ylim'][1] - L['ylim'][0]) / (L['xlim'][1] - L['xlim'][0])),
                               facecolor='white')
        ax.scatter(L['x'], L['y'], s=4, c='0.88')                      # all sensors, faint
        on = hit & (times <= tcut)
        if on.any():
            ax.scatter(L['x'][on], L['y'][on], s=18, c=sm.to_rgba(charges[on]))
        ax.set(xlim=L['xlim'], ylim=L['ylim']); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f't ≤ {tcut:.0f} ns  ({int(on.sum())} hits)', fontsize=10)
        fig.canvas.draw(); w, h = fig.canvas.get_width_height()
        frames.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path


def event_display_3d(detector, loaded_indices, loaded_charges, loaded_times,
                     plot_time=False, log_scale=False,
                     show_all_sensors=True, marker_size=6, show_colorbar=True,
                     opacity=1.0, dark_theme=True, n_disc_segments=12,
                     colorscale='viridis', surface_color='gray',
                     inactive_color='red', inactive_opacity=0.3,
                     cmin=None, cmax=None, colorbar_label=None,
                     colorbar_thickness=20, colorbar_tickfont_size=None,
                     colorbar_x=1.0, colorbar_len=0.5,
                     colorbar_tickfont_family=None,
                     colorbar_tickvals=None, colorbar_ticktext=None,
                     figname=None, return_fig=False):
    """
    Visualize detector event data in 3D using circular discs oriented according to surface normals.
    Shows red discs for sensors without charge and color-coded discs for sensors with hits.

    Parameters:
    -----------
    detector : Cylinder, Sphere or Box
        Detector whose sensors and wall are drawn
    loaded_indices : array-like
        Indices of non-zero hits
    loaded_charges : array-like
        Charge values at non-zero indices
    loaded_times : array-like
        Time values at non-zero indices
    plot_time : bool, default=False
        If True, color by time values; if False, color by charge values
    log_scale : bool, default=False
        If True, apply logarithmic scaling to the color gradient
    show_all_sensors : bool, default=True
        If True, shows all sensor positions as red discs for inactive sensors
    marker_size : int, default=6
        Size scaling factor for the disc radius
    show_colorbar : bool, default=True
        If True, shows the colorbar; if False, creates minimal display with just sensors
    opacity : float, default=1.0
        Opacity of the hit detector discs
    dark_theme : bool, default=True
        If True, use black background; if False, use white background
    n_disc_segments : int, default=12
        Number of segments for each disc (higher = smoother circles)
    colorscale : str, default='viridis'
        Plotly colorscale name (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
        'turbo', 'rainbow', 'jet', 'hot', 'cool', 'RdYlBu', 'RdBu', 'Spectral')
    surface_color : str, default='gray'
        Color of the detector surface (e.g., 'gray', 'black', 'darkgray', 'lightgray', 'silver')
    inactive_color : str, default='red'
        Color for sensors without charge
    inactive_opacity : float, default=0.3
        Opacity for inactive sensor discs
    """
    # Set color scheme based on theme
    if dark_theme:
        paper_color = 'black'
        colorbar_color = 'white'
    else:
        paper_color = 'white'
        colorbar_color = 'black'

    import plotly.graph_objects as go

    # Convert inputs to numpy arrays if not already
    loaded_indices = np.array(loaded_indices)
    loaded_charges = np.array(loaded_charges)
    loaded_times = np.array(loaded_times)

    # Validate inputs
    if len(loaded_indices) != len(loaded_charges) or len(loaded_indices) != len(loaded_times):
        raise ValueError("loaded_indices, loaded_charges, and loaded_times must have the same length")

    # Create the plot
    fig = go.Figure()

    # Add detector surface
    _add_detector_surface(fig, detector, surface_color)

    # Calculate disc radius based on sensor radius and marker_size scaling
    disc_radius = detector.S_radius * (marker_size / 6.0)  # Scale relative to default marker_size

    # Show all sensors as red discs if requested
    if show_all_sensors:
        # Find inactive sensor indices (all indices not in loaded_indices)
        all_indices = np.arange(len(detector.all_points))
        inactive_indices = np.setdiff1d(all_indices, loaded_indices)

        if len(inactive_indices) > 0:
            # Get positions and normals for inactive sensors
            inactive_positions = detector.all_points[inactive_indices]
            inactive_normals = calculate_surface_normals(detector, inactive_indices)

            # Create inactive disc meshes
            inactive_vertices = []
            inactive_faces = []
            vertex_offset = 0

            for pos, normal in zip(inactive_positions, inactive_normals):
                # Create disc mesh
                vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_disc_segments)

                # Adjust face indices for global vertex array
                faces_adjusted = faces + vertex_offset

                # Add to global arrays
                inactive_vertices.append(vertices)
                inactive_faces.append(faces_adjusted)

                vertex_offset += len(vertices)

            # Combine all inactive vertices and faces
            if inactive_vertices:
                combined_inactive_vertices = np.vstack(inactive_vertices)
                combined_inactive_faces = np.vstack(inactive_faces)

                # Create mesh trace for inactive sensors
                inactive_mesh_trace = go.Mesh3d(
                    x=combined_inactive_vertices[:, 0],
                    y=combined_inactive_vertices[:, 1], 
                    z=combined_inactive_vertices[:, 2],
                    i=combined_inactive_faces[:, 0],
                    j=combined_inactive_faces[:, 1],
                    k=combined_inactive_faces[:, 2],
                    color=inactive_color,
                    opacity=inactive_opacity,
                    name=f'Inactive Sensors ({len(inactive_indices)})',
                    hoverinfo='skip',  # Disable hover for inactive sensors
                    lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
                    showscale=False
                )

                fig.add_trace(inactive_mesh_trace)

    # Process hit sensors if any exist
    if len(loaded_indices) > 0:
        # Get positions of hit sensors
        hit_positions = detector.all_points[loaded_indices]

        # Select which values to use for coloring
        color_values = loaded_times if plot_time else loaded_charges

        # Handle log scaling
        if log_scale:
            # Handle zero/negative values for log scale
            positive_mask = color_values > 0
            if not np.any(positive_mask):
                print("Warning: No positive values found for log scale. Using linear scale instead.")
                log_scale = False
            else:
                min_positive = np.min(color_values[positive_mask])
                color_values_log = np.copy(color_values)
                color_values_log[~positive_mask] = min_positive * 0.1
                color_values_log = np.log10(color_values_log)
                colorbar_title = f"{'Time' if plot_time else 'Charge'} (log₁₀ scale)"
                plot_color_values = color_values_log

        if not log_scale:
            colorbar_title = 'Time (ns)' if plot_time else 'Charge (PE)'
            plot_color_values = color_values

        # Optional explicit colorbar label (e.g. for a ratio/fraction map
        # where 'Charge (PE)' would be misleading).
        if colorbar_label is not None:
            colorbar_title = colorbar_label

        # Sort points by depth (z-coordinate) for better depth rendering
        depth_order = np.argsort(hit_positions[:, 2])
        hit_positions_sorted = hit_positions[depth_order]
        plot_color_values_sorted = plot_color_values[depth_order]
        sorted_indices = loaded_indices[depth_order]
        sorted_charges = loaded_charges[depth_order] 
        sorted_times = loaded_times[depth_order]

        # Calculate surface normals for sorted sensors
        normals_sorted = calculate_surface_normals(detector, sorted_indices)
        color_min, color_max = plot_color_values_sorted.min(), plot_color_values_sorted.max()
        # Optional explicit colour range (e.g. pin a fraction map to [0, 1]).
        if cmin is not None:
            color_min = float(cmin)
        if cmax is not None:
            color_max = float(cmax)
        color_normalized = plot_color_values_sorted

        # Create individual disc meshes for hit sensors
        all_vertices = []
        all_faces = []
        all_intensities = []
        vertex_offset = 0

        for i, (pos, normal, color_val, norm_color) in enumerate(zip(
            hit_positions_sorted, normals_sorted, plot_color_values_sorted, color_normalized)):

            # Create disc mesh
            vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_disc_segments)

            # Adjust face indices for global vertex array
            faces_adjusted = faces + vertex_offset

            # Add to global arrays
            all_vertices.append(vertices)
            all_faces.append(faces_adjusted)

            # Each vertex gets the same color intensity for this disc
            all_intensities.extend([norm_color] * len(vertices))

            vertex_offset += len(vertices)

        # Combine all vertices and faces for hit sensors
        if all_vertices:
            combined_vertices = np.vstack(all_vertices)
            combined_faces = np.vstack(all_faces)
            combined_intensities = np.array(all_intensities)

            # Create mesh trace for hit sensors
            mesh_trace = go.Mesh3d(
                x=combined_vertices[:, 0],
                y=combined_vertices[:, 1], 
                z=combined_vertices[:, 2],
                i=combined_faces[:, 0],
                j=combined_faces[:, 1],
                k=combined_faces[:, 2],
                intensity=combined_intensities,
                colorscale=colorscale,
                opacity=opacity,
                name=f'Event Data ({len(loaded_indices)} hits)',
                hoverinfo='skip',  # Disable hover for mesh
                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
                showscale=show_colorbar
            )

            # Pin the colour range (data min/max, or the explicit cmin/cmax)
            # so the mapping is well-defined with or without a colorbar.
            mesh_trace.update(cmin=color_min, cmax=color_max)

            # Add colorbar settings if requested
            if show_colorbar:
                _tickfont = dict(color=colorbar_color)
                if colorbar_tickfont_size is not None:
                    _tickfont['size'] = colorbar_tickfont_size
                if colorbar_tickfont_family is not None:
                    _tickfont['family'] = colorbar_tickfont_family
                _titlefont = dict(color=colorbar_color)
                if colorbar_tickfont_family is not None:
                    _titlefont['family'] = colorbar_tickfont_family
                _cbar = dict(
                    title=dict(text=colorbar_title, font=_titlefont),
                    tickfont=_tickfont,
                    thickness=colorbar_thickness,
                    len=colorbar_len,
                    x=colorbar_x,
                )
                if colorbar_tickvals is not None:
                    _cbar['tickvals'] = list(colorbar_tickvals)
                    if colorbar_ticktext is not None:
                        _cbar['ticktext'] = list(colorbar_ticktext)
                mesh_trace.update(colorbar=_cbar)

            fig.add_trace(mesh_trace)

            # Add invisible scatter trace for hover information on hit sensors only
            fig.add_trace(go.Scatter3d(
                x=hit_positions_sorted[:, 0],
                y=hit_positions_sorted[:, 1], 
                z=hit_positions_sorted[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    opacity=0,  # Invisible
                ),
                text=[f'Sensor ID: {idx}<br>Charge: {charge:.3f} PE<br>Time: {time:.3f} ns' 
                      for idx, charge, time in zip(sorted_indices, sorted_charges, sorted_times)],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
                name='Hover Info'
            ))

        # Calculate reasonable axis ranges based on hit data
        all_x = hit_positions_sorted[:, 0]
        all_y = hit_positions_sorted[:, 1]
        all_z = hit_positions_sorted[:, 2]
    else:
        # If no hits, use all sensor positions for range calculation
        print("No event data to display - showing only inactive sensors")
        all_x = detector.all_points[:, 0]
        all_y = detector.all_points[:, 1]
        all_z = detector.all_points[:, 2]

    # Calculate axis ranges
    margin = 0.1 * max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))
    x_range = [np.min(all_x) - margin, np.max(all_x) + margin]
    y_range = [np.min(all_y) - margin, np.max(all_y) + margin]
    z_range = [np.min(all_z) - margin, np.max(all_z) + margin]

    # Update layout for clean display
    margin_right = 80 if show_colorbar and len(loaded_indices) > 0 else 0

    # Use cube aspect mode for spheres, data for cylinders and boxes
    aspect_mode = 'cube' if isinstance(detector, Sphere) else 'data'

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                visible=False,  
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=x_range
            ),
            yaxis=dict(
                visible=False,  
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=y_range
            ),
            zaxis=dict(
                visible=False,  
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                range=z_range
            ),
            aspectmode=aspect_mode,
            bgcolor=paper_color,
        ),
        height=800,
        width=1000,
        showlegend=False,
        paper_bgcolor=paper_color,
        plot_bgcolor=paper_color,   
        margin=dict(l=0, r=margin_right+0, t=0, b=0)
    )

    # Return the assembled figure instead of rendering (lets callers compose
    # multi-scene subplots / export vector PDF themselves).
    if return_fig:
        return fig

    # Save figure if filename is provided
    if figname:
        fig.write_image(figname, width=2000, height=2000, scale=2)
        fig.show()
    else:
        fig.show()

def _add_detector_surface(fig, detector, surface_color):
    """Add detector surface to the plot"""
    if isinstance(detector, Cylinder):
        _add_cylinder_surface(fig, detector, surface_color)
    elif isinstance(detector, Sphere):
        _add_sphere_surface(fig, detector, surface_color)
    elif isinstance(detector, Box):
        _add_box_surface(fig, detector, surface_color)
    else:
        raise TypeError(
            f"event_display_3d draws Cylinder, Sphere and Box detectors, not {type(detector).__name__}")


def _add_cylinder_surface(fig, detector, surface_color):
    """Add cylindrical surface to the plot with offset to avoid disc overlap"""
    import plotly.graph_objects as go

    # Offset to avoid overlap with discs
    offset = 0.995

    # Barrel surface
    theta = np.linspace(0, 2 * np.pi, 50)
    z_barrel = np.linspace(-(offset*detector.H)/2, (offset*detector.H)/2, 20)
    theta_mesh, z_mesh = np.meshgrid(theta, z_barrel)

    x_barrel = (offset*detector.r) * np.cos(theta_mesh) + detector.C[0]
    y_barrel = (offset*detector.r) * np.sin(theta_mesh) + detector.C[1]
    z_barrel_mesh = z_mesh + detector.C[2]

    # Add barrel surface
    fig.add_trace(go.Surface(
        x=x_barrel, y=y_barrel, z=z_barrel_mesh,
        opacity=1.0,
        showscale=False,
        colorscale=[[0, surface_color], [1, surface_color]],
        name='Barrel Surface',
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=None,
        hoverlabel=None
    ))

    # Cap surfaces
    r_cap = np.linspace(0, offset*detector.r, 20)
    theta_cap = np.linspace(0, 2 * np.pi, 50)
    r_mesh, theta_mesh = np.meshgrid(r_cap, theta_cap)

    x_cap = r_mesh * np.cos(theta_mesh) + detector.C[0]
    y_cap = r_mesh * np.sin(theta_mesh) + detector.C[1]

    # Top cap
    z_top = np.full_like(x_cap, (offset*detector.H)/2 + detector.C[2])
    fig.add_trace(go.Surface(
        x=x_cap, y=y_cap, z=z_top,
        opacity=1.0,
        showscale=False,
        colorscale=[[0, surface_color], [1, surface_color]],
        name='Top Cap',
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=None,
        hoverlabel=None
    ))

    # Bottom cap
    z_bottom = np.full_like(x_cap, -(offset*detector.H)/2 + detector.C[2])
    fig.add_trace(go.Surface(
        x=x_cap, y=y_cap, z=z_bottom,
        opacity=1.0,
        showscale=False,
        colorscale=[[0, surface_color], [1, surface_color]],
        name='Bottom Cap',
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=None,
        hoverlabel=None
    ))

def _add_sphere_surface(fig, detector, surface_color):
    """Add spherical surface to the plot with offset to avoid disc overlap"""
    import plotly.graph_objects as go

    # Offset to avoid overlap with discs
    offset = 0.995

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = (offset*detector.r) * np.outer(np.cos(u), np.sin(v)) + detector.C[0]
    y_sphere = (offset*detector.r) * np.outer(np.sin(u), np.sin(v)) + detector.C[1]
    z_sphere = (offset*detector.r) * np.outer(np.ones(np.size(u)), np.cos(v)) + detector.C[2]

    # Add sphere surface
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=1.0,
        showscale=False,
        colorscale=[[0, surface_color], [1, surface_color]],
        name='Sphere Surface',
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=None,
        hoverlabel=None
    ))

def _add_box_surface(fig, detector, surface_color):
    """Add box surface to the plot with offset to avoid disc overlap"""
    import plotly.graph_objects as go

    # Offset to avoid overlap with discs
    offset = 0.995

    # Define box vertices with offset
    x_min, x_max = -offset*detector.L/2 + detector.C[0], offset*detector.L/2 + detector.C[0]
    y_min, y_max = -offset*detector.W/2 + detector.C[1], offset*detector.W/2 + detector.C[1]
    z_min, z_max = -offset*detector.H/2 + detector.C[2], offset*detector.H/2 + detector.C[2]

    # Front face (+y)
    x_front = [x_min, x_max, x_max, x_min, x_min]
    z_front = [z_min, z_min, z_max, z_max, z_min]
    y_front = [y_max] * 5

    fig.add_trace(go.Mesh3d(
        x=x_front, y=y_front, z=z_front,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=1.0,
        color=surface_color,
        showscale=False,
        name='Front Face',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Back face (-y)
    y_back = [y_min] * 5

    fig.add_trace(go.Mesh3d(
        x=x_front, y=y_back, z=z_front,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=1.0,
        color=surface_color,
        showscale=False,
        name='Back Face',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Left face (-x)
    x_left = [x_min] * 5
    y_left = [y_min, y_max, y_max, y_min, y_min]
    z_left = [z_min, z_min, z_max, z_max, z_min]

    fig.add_trace(go.Mesh3d(
        x=x_left, y=y_left, z=z_left,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=1.0,
        color=surface_color,
        showscale=False,
        name='Left Face',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Right face (+x)
    x_right = [x_max] * 5

    fig.add_trace(go.Mesh3d(
        x=x_right, y=y_left, z=z_left,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=1.0,
        color=surface_color,
        showscale=False,
        name='Right Face',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Top face (+z)
    x_top = [x_min, x_max, x_max, x_min, x_min]
    y_top = [y_min, y_min, y_max, y_max, y_min]
    z_top = [z_max] * 5

    fig.add_trace(go.Mesh3d(
        x=x_top, y=y_top, z=z_top,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=1.0,
        color=surface_color,
        showscale=False,
        name='Top Face',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Bottom face (-z)
    z_bottom = [z_min] * 5

    fig.add_trace(go.Mesh3d(
        x=x_top, y=y_top, z=z_bottom,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=1.0,
        color=surface_color,
        showscale=False,
        name='Bottom Face',
        showlegend=False,
        hoverinfo='skip'
    ))
