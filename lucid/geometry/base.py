"""
Base detector class with complex visualization methods.
"""

import numpy as np
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from .utils import calculate_surface_normals, create_disc_mesh


class Detector(ABC):
    """Base class for detector geometries"""
    
    def __init__(self, n_sensors, sensor_radius):
        """
        Initialize common detector attributes.
        
        Parameters:
        -----------
        n_sensors : int
            Number of photosensors
        sensor_radius : float
            Radius of individual sensors
        """
        self.C = np.array([0.0, 0.0, 0.0])  # Always centered at origin
        self.n_sensors = n_sensors
        self.S_radius = sensor_radius
        
        # These will be set by place_photosensors()
        self.all_points = None
        self.ID_to_position = None
        self.ID_to_case = None

    @abstractmethod
    def place_photosensors(self):
        """Position the photo sensor centers. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def visualize_geometry_wireframe(self, show_sensors=True):
        """Visualize the detector geometry as wireframe. Must be implemented by subclasses."""
        pass

    # ── Propagation abstract methods (Phase 9) ─────────────────────────
    # Subclasses implement these to enable the shared create_propagator().
    # bounds_check() was added in Phase 6.

    def intersect_ray(self, origins, directions):
        """Batch ray-geometry intersection.

        Parameters
        ----------
        origins : jnp.ndarray, shape (n_rays, 3)
        directions : jnp.ndarray, shape (n_rays, 3)

        Returns
        -------
        intersection_point : jnp.ndarray, shape (n_rays, 3)
        t_value : jnp.ndarray, shape (n_rays,)
        grid_info : geometry-specific opaque data passed to point_to_grid_cell
        surface_info : geometry-specific data passed to compute_normal
        """
        raise NotImplementedError

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward surface normals at intersection points.

        Parameters
        ----------
        intersection_point : jnp.ndarray, shape (n_rays, 3)
        surface_info : from intersect_ray

        Returns
        -------
        normals : jnp.ndarray, shape (n_rays, 3), outward convention
        """
        raise NotImplementedError

    def point_to_grid_cell(self, grid_info):
        """Map intersection data to linear grid cell indices.

        Parameters
        ----------
        grid_info : from intersect_ray

        Returns
        -------
        linear_idx : jnp.ndarray, shape (n_rays,), int
        """
        raise NotImplementedError

    def assign_sensor_to_cells(self, sensors, sensor_radius):
        """Map each sensor to the grid cells it overlaps.

        Parameters
        ----------
        sensors : jnp.ndarray, shape (n_sensors, 3)
        sensor_radius : float

        Returns
        -------
        assignments : jnp.ndarray, geometry-specific shape
        """
        raise NotImplementedError

    def grid_cell_centers(self):
        """Compute (n_cells, 3) array of grid cell center positions.

        Returns
        -------
        centers : jnp.ndarray, shape (n_total_cells, 3)
        """
        raise NotImplementedError

    def total_grid_cells(self):
        """Return total number of grid cells.

        Returns
        -------
        int
        """
        raise NotImplementedError

    def cell_index_to_coords(self, linear_idx):
        """Decode linear cell index back to grid coordinates.

        Used by the inverted sensor map builder to match geometric
        assignments (which store grid coordinates) against cell indices.

        Parameters
        ----------
        linear_idx : int or jnp.ndarray

        Returns
        -------
        coords : geometry-specific (cell_i, cell_j, cell_k) or similar
        """
        raise NotImplementedError

    def visualize_event_data_plotly_discs(self, loaded_indices, loaded_charges, loaded_times, 
                                     plot_time=False, log_scale=False, title=None, 
                                     show_all_sensors=True, marker_size=6, show_colorbar=True,
                                     opacity=1.0, dark_theme=True, n_disc_segments=12, 
                                     colorscale='viridis', surface_color='gray', 
                                     inactive_color='red', inactive_opacity=0.3, figname=None):
        """
        Visualize detector event data in 3D using circular discs oriented according to surface normals.
        Shows red discs for sensors without charge and color-coded discs for sensors with hits.
        
        Parameters:
        -----------
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
        title : str, optional
            Title for the plot. If None, auto-generates title.
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
            bg_color = 'black'
            paper_color = 'black'
            colorbar_color = 'white'
        else:
            bg_color = 'white'
            paper_color = 'white'
            colorbar_color = 'black'
        
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
        self._add_detector_surface(fig, surface_color)
        
        # Calculate disc radius based on sensor radius and marker_size scaling
        disc_radius = self.S_radius * (marker_size / 6.0)  # Scale relative to default marker_size
        
        # Show all sensors as red discs if requested
        if show_all_sensors:
            # Find inactive sensor indices (all indices not in loaded_indices)
            all_indices = np.arange(len(self.all_points))
            inactive_indices = np.setdiff1d(all_indices, loaded_indices)
            
            if len(inactive_indices) > 0:
                # Get positions and normals for inactive sensors
                inactive_positions = self.all_points[inactive_indices]
                inactive_normals = calculate_surface_normals(self, inactive_indices)
                
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
            hit_positions = self.all_points[loaded_indices]
            
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
            
            # Sort points by depth (z-coordinate) for better depth rendering
            depth_order = np.argsort(hit_positions[:, 2])
            hit_positions_sorted = hit_positions[depth_order]
            plot_color_values_sorted = plot_color_values[depth_order]
            sorted_indices = loaded_indices[depth_order]
            sorted_charges = loaded_charges[depth_order] 
            sorted_times = loaded_times[depth_order]
            
            # Calculate surface normals for sorted sensors
            normals_sorted = calculate_surface_normals(self, sorted_indices)
            color_min, color_max = plot_color_values_sorted.min(), plot_color_values_sorted.max()
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
                
                # Add colorbar settings if requested
                if show_colorbar:
                    # Map normalized color range back to original values for colorbar
                    mesh_trace.update(
                        cmin=color_min,
                        cmax=color_max,
                        colorbar=dict(
                            title=dict(
                                text=colorbar_title,
                                font=dict(color=colorbar_color)
                            ),
                            tickfont=dict(color=colorbar_color),
                            thickness=20,
                            len=0.5,
                            x=1.0
                        )
                    )
                
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
            all_x = self.all_points[:, 0]
            all_y = self.all_points[:, 1]
            all_z = self.all_points[:, 2]
        
        # Calculate axis ranges
        margin = 0.1 * max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))
        x_range = [np.min(all_x) - margin, np.max(all_x) + margin]
        y_range = [np.min(all_y) - margin, np.max(all_y) + margin]
        z_range = [np.min(all_z) - margin, np.max(all_z) + margin]
        
        # Update layout for clean display
        margin_right = 80 if show_colorbar and len(loaded_indices) > 0 else 0
        
        # Use cube aspect mode for spheres, data for cylinders
        aspect_mode = 'cube' if not hasattr(self, 'H') else 'data'
        
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
        
        # Save figure if filename is provided
        if figname:
            fig.write_image(figname, width=2000, height=2000, scale=2)
            fig.show()
        else:
            fig.show()

    def _add_detector_surface(self, fig, surface_color='gray'):
        """Add detector surface to the plot"""
        # Import here to avoid circular imports
        from .cylinder import Cylinder
        from .sphere import Sphere
        from .box import Box
        
        if isinstance(self, Cylinder):
            self._add_cylinder_surface(fig, surface_color)
        elif isinstance(self, Sphere):
            self._add_sphere_surface(fig, surface_color)
        elif isinstance(self, Box):
            self._add_box_surface(fig, surface_color)
    
    def _add_cylinder_surface(self, fig, surface_color='gray'):
        """Add cylindrical surface to the plot with offset to avoid disc overlap"""
        # Offset to avoid overlap with discs
        offset = 0.995
        
        # Barrel surface
        theta = np.linspace(0, 2 * np.pi, 50)
        z_barrel = np.linspace(-(offset*self.H)/2, (offset*self.H)/2, 20)
        theta_mesh, z_mesh = np.meshgrid(theta, z_barrel)
        
        x_barrel = (offset*self.r) * np.cos(theta_mesh) + self.C[0]
        y_barrel = (offset*self.r) * np.sin(theta_mesh) + self.C[1]
        z_barrel_mesh = z_mesh + self.C[2]

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
        r_cap = np.linspace(0, offset*self.r, 20)
        theta_cap = np.linspace(0, 2 * np.pi, 50)
        r_mesh, theta_mesh = np.meshgrid(r_cap, theta_cap)
        
        x_cap = r_mesh * np.cos(theta_mesh) + self.C[0]
        y_cap = r_mesh * np.sin(theta_mesh) + self.C[1]
        
        # Top cap
        z_top = np.full_like(x_cap, (offset*self.H)/2 + self.C[2])
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
        z_bottom = np.full_like(x_cap, -(offset*self.H)/2 + self.C[2])
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
    
    def _add_sphere_surface(self, fig, surface_color='gray'):
        """Add spherical surface to the plot with offset to avoid disc overlap"""
        # Offset to avoid overlap with discs
        offset = 0.995
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = (offset*self.r) * np.outer(np.cos(u), np.sin(v)) + self.C[0]
        y_sphere = (offset*self.r) * np.outer(np.sin(u), np.sin(v)) + self.C[1]
        z_sphere = (offset*self.r) * np.outer(np.ones(np.size(u)), np.cos(v)) + self.C[2]

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
    
    def _add_box_surface(self, fig, surface_color='gray'):
        """Add box surface to the plot with offset to avoid disc overlap"""
        # Offset to avoid overlap with discs
        offset = 0.995
        
        # Define box vertices with offset
        x_min, x_max = -offset*self.L/2 + self.C[0], offset*self.L/2 + self.C[0]
        y_min, y_max = -offset*self.W/2 + self.C[1], offset*self.W/2 + self.C[1]
        z_min, z_max = -offset*self.H/2 + self.C[2], offset*self.H/2 + self.C[2]
        
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