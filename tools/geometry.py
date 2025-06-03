import numpy as np
import json
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc


def generate_dataset_origins(center, heights, radii, divisions):
    """Creates a collection of origins for the dataset generation."""
    xs, ys, zs = [], [], []

    for Z in heights:
        for R, N in zip(radii, divisions):
            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            x = R * np.cos(theta) + center[0]
            y = R * np.sin(theta) + center[1]

            xs.extend(x)
            ys.extend(y)
            zs.extend([Z] * len(x))

    return xs, ys, zs

def rotate_vector(vector, axis, angle):
    """ Rotate a vector around an axis by a given angle in radians. """
    axis = normalize(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product = np.cross(axis, vector)
    dot_product = np.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis
        
def normalize(v):
    """ Normalize a vector. """
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def generate_isotropic_random_vectors(N=1):
    """ A function to generate N isotropic random vectors. """
    # Generate random azimuthal angles (phi) in the range [0, 2*pi)
    phi = 2 * np.pi * np.random.rand(N)

    # Generate random polar angles (theta) in the range [0, pi)
    theta = np.arccos(2 * np.random.rand(N) - 1)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Stack the Cartesian coordinates into a 2D array
    vectors = np.column_stack((x, y, z))

    # Normalize the vectors
    vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors_normalized

def generate_vectors_on_cone_surface(R, theta, num_vectors=10):
    """ Generate vectors on the surface of a cone around R. """
    R = normalize(R)

    # Generate random azimuthal angles from 0 to 2pi
    phi_values = np.random.uniform(0, 2 * np.pi, num_vectors)

    # Spherical to Cartesian coordinates in the local system
    x_values = np.sin(theta) * np.cos(phi_values)
    y_values = np.sin(theta) * np.sin(phi_values)
    z_value = np.cos(theta)

    local_vectors = np.column_stack((x_values, y_values, z_value * np.ones_like(x_values)))
    #local_vectors = np.column_stack((x_values, y_values, z_values))

    # Find rotation axis and angle to align local z-axis with R
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, R)
    non_zero_indices = np.linalg.norm(axis, axis=-1) != 0  # Check for non-zero norms

    # If R is not already along z-axis
    #angles = np.arccos(np.einsum('ij,ij->i', z_axis, R[non_zero_indices]))
    angles = np.arccos(np.sum(z_axis * R[non_zero_indices], axis=-1))


    # Apply rotation to vectors
    rotated_vectors = rotate_vector_batch(local_vectors[non_zero_indices], axis[non_zero_indices], angles)

    # Update the original local vectors with rotated vectors
    local_vectors[non_zero_indices] = rotated_vectors

    # Convert local vectors to global coordinates
    vectors = np.dot(local_vectors, np.linalg.norm(R))

    return vectors

def rotate_vector_batch(vectors, axes, angles):
    """ Rotate multiple vectors by specified angles around the given axes. """
    norms = np.linalg.norm(axes, axis=-1)
    axes_normalized = axes / norms[:, np.newaxis]
    quaternion = Rotation.from_rotvec(axes_normalized * angles[:, np.newaxis]).as_quat()

    # Reshape vectors to (50000, 3) if needed
    if vectors.shape[0] == 1:
        vectors = vectors.reshape(-1, 3)

    rotated_vectors = Rotation.from_quat(quaternion).apply(vectors)

    return rotated_vectors


def generate_concentric_hexagons(n_sensors, radius_eff):
    """Generate hexagonal pattern using concentric rings.
    
    Returns exact number of sensors in a regular hexagonal pattern.
    Pattern: center (1) + rings of 6k sensors each, where k is ring number.
    Total sensors for n rings: 1 + 6(1 + 2 + ... + n) = 1 + 3n(n+1)
    """
    if n_sensors == 0:
        return np.array([]).reshape(0, 2)
    
    if n_sensors == 1:
        return np.array([[0, 0]])
    
    # Find how many complete rings we can fit
    # Solve: 1 + 3n(n+1) <= n_sensors for largest n
    n_rings = 0
    while 1 + 3 * (n_rings + 1) * (n_rings + 2) <= n_sensors:
        n_rings += 1
    
    # Calculate spacing to fit the outermost ring within radius_eff
    if n_rings == 0:
        spacing = radius_eff  # Single sensor at center
    else:
        spacing = radius_eff / n_rings
    
    points = []
    
    # Center point
    points.append([0, 0])
    
    # Generate concentric hexagonal rings
    for ring in range(1, n_rings + 1):
        ring_radius = ring * spacing
        n_sensors_in_ring = 6 * ring
        
        # Generate points around the ring
        for i in range(n_sensors_in_ring):
            angle = 2 * np.pi * i / n_sensors_in_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            points.append([x, y])
    
    # If we need more sensors and have space, add partial outer ring
    current_count = len(points)
    if current_count < n_sensors and n_rings * spacing < radius_eff:
        remaining = n_sensors - current_count
        next_ring = n_rings + 1
        next_ring_radius = next_ring * spacing
        
        if next_ring_radius <= radius_eff:
            # Add sensors from the next ring
            sensors_to_add = min(remaining, 6 * next_ring)
            for i in range(sensors_to_add):
                angle = 2 * np.pi * i / (6 * next_ring)
                x = next_ring_radius * np.cos(angle)
                y = next_ring_radius * np.sin(angle)
                points.append([x, y])
    
    return np.array(points[:n_sensors])  # Ensure exact count


class Cylinder:
    """Manage the detector geometry"""
    def __init__(self, radius, height, n_sensors, sensor_radius):
        self.C = np.array([0.0, 0.0, 0.0])  # Fixed at origin
        self.r = radius
        self.H = height 
        self.n_sensors = n_sensors
        self.S_radius = sensor_radius

        self.place_photosensors()

    def place_photosensors(self):
        """Position the photo sensor centers proportionally by surface area."""
        # Calculate surface areas
        barrel_area = 2 * np.pi * self.r * self.H
        caps_area = 2 * np.pi * self.r**2  # Both caps combined
        total_area = barrel_area + caps_area
        
        # Distribute sensors proportionally
        n_barrel = int(self.n_sensors * barrel_area / total_area)
        n_caps = self.n_sensors - n_barrel
        n_per_cap = n_caps // 2  # Split equally between top and bottom
        
        # Place barrel sensors
        self.barr_points = self._place_barrel_sensors(n_barrel)
        
        # Place cap sensors
        self.tcap_points = self._place_cap_sensors(n_per_cap, self.H/2)  # Top cap
        self.bcap_points = self._place_cap_sensors(n_per_cap, -self.H/2)  # Bottom cap
        
        # Combine all points
        self.all_points = np.concatenate([self.barr_points, self.tcap_points, self.bcap_points], axis=0)
        
        # Create ID mappings
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # Create case mappings (0=barrel, 1=top cap, 2=bottom cap)
        self.ID_to_case = {}
        n_barr = len(self.barr_points)
        n_tcap = len(self.tcap_points)
        n_bcap = len(self.bcap_points)
        
        for i in range(len(self.all_points)):
            if i < n_barr:
                self.ID_to_case[i] = 0
            elif i < n_barr + n_tcap:
                self.ID_to_case[i] = 1
            else:
                self.ID_to_case[i] = 2

    def _place_barrel_sensors(self, n_sensors):
        """Place sensors on barrel surface with rectangular grid."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        # Calculate effective dimensions (with margins)
        height_eff = self.H - 3 * self.S_radius  # Top and bottom margins
        circumference_eff = 2 * np.pi * self.r
        
        # Find optimal rows and columns for approximately square spacing
        aspect_ratio = height_eff / circumference_eff
        n_rows = int(np.sqrt(n_sensors * aspect_ratio))
        n_cols = n_sensors // n_rows
        
        # Adjust to get closer to target
        while n_rows * n_cols < n_sensors and n_rows > 1:
            if (n_rows + 1) * n_cols <= n_sensors:
                n_rows += 1
            elif n_rows * (n_cols + 1) <= n_sensors:
                n_cols += 1
            else:
                break
        
        # Generate grid
        z_positions = np.linspace(-height_eff/2, height_eff/2, n_rows) + self.C[2]
        theta_positions = np.linspace(0, 2*np.pi, n_cols, endpoint=False)
        
        points = []
        for z in z_positions:
            for theta in theta_positions:
                x = self.r * np.cos(theta) + self.C[0]
                y = self.r * np.sin(theta) + self.C[1]
                points.append([x, y, z])
        
        return np.array(points[:n_sensors])  # Trim to exact count

    def _place_cap_sensors(self, n_sensors, z_position):
        """Place sensors on cap surface with concentric hexagonal rings."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        # Calculate effective radius (with margin)
        radius_eff = self.r - 1.5 * self.S_radius
        
        if radius_eff <= 0:
            return np.array([]).reshape(0, 3)
        
        # Generate concentric hexagonal pattern
        hex_points = generate_concentric_hexagons(n_sensors, radius_eff)
        
        # Convert to 3D and translate
        points_3d = np.zeros((len(hex_points), 3))
        points_3d[:, 0] = hex_points[:, 0] + self.C[0]
        points_3d[:, 1] = hex_points[:, 1] + self.C[1]
        points_3d[:, 2] = z_position + self.C[2]
        
        return points_3d

    def visualize_event_data_plotly(self, loaded_indices, loaded_charges, loaded_times, 
                                   plot_time=False, log_scale=False, title=None, 
                                   show_all_detectors=False, marker_size=6, show_colorbar=True,
                                   opacity=1.0, dark_theme=True):
        """
        Visualize detector event data in 3D using Plotly with colors proportional to charge or time.
        
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
        show_all_detectors : bool, default=False
            If True, shows all detector positions as light gray background points
        marker_size : int, default=6
            Size of the markers for hit detectors
        show_colorbar : bool, default=True
            If True, shows the colorbar; if False, creates minimal display with just sensors
        opacity : float, default=1.0
            Opacity of the markers (1.0 = fully opaque, avoids depth sorting issues)
        dark_theme : bool, default=True
            If True, use black background; if False, use white background
        """
        import plotly.graph_objects as go
        import numpy as np
        
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
        
        if len(loaded_indices) == 0:
            print("No event data to display")
            return
        
        # Get positions of hit detectors
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
        
        # Create the plot
        fig = go.Figure()
        
        # Optionally add all detector positions as background
        if show_all_detectors:
            fig.add_trace(go.Scatter3d(
                x=self.all_points[:, 0], 
                y=self.all_points[:, 1], 
                z=self.all_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2, 
                    color='lightgray', 
                    opacity=0.3
                ),
                name='All Detectors',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Sort points by depth (z-coordinate) to improve transparency rendering
        # Sort from back to front for better depth rendering
        depth_order = np.argsort(hit_positions[:, 2])
        hit_positions_sorted = hit_positions[depth_order]
        plot_color_values_sorted = plot_color_values[depth_order]
        sorted_indices = loaded_indices[depth_order]
        sorted_charges = loaded_charges[depth_order] 
        sorted_times = loaded_times[depth_order]
        
        # Create marker dictionary with full opacity to avoid transparency issues
        marker_dict = dict(
            size=marker_size,
            color=plot_color_values_sorted,
            colorscale='viridis',
            opacity=opacity,  # Use parameter-controlled opacity
            line=dict(width=0, color='rgba(0,0,0,0)')  # Remove outlines for cleaner look
        )
        
        # Add colorbar only if requested
        if show_colorbar:
            marker_dict['colorbar'] = dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(color=colorbar_color)
                ),
                tickfont=dict(color=colorbar_color),
                thickness=20,
                len=0.7,
                x=1.02
            )
        
        # Add scatter plot for hit detectors
        fig.add_trace(go.Scatter3d(
            x=hit_positions_sorted[:, 0], 
            y=hit_positions_sorted[:, 1], 
            z=hit_positions_sorted[:, 2],
            mode='markers',
            marker=marker_dict,
            name=f'Event Data ({len(loaded_indices)} hits)',
            text=[f'Detector ID: {idx}<br>Charge: {charge:.3f} PE<br>Time: {time:.3f} ns' 
                  for idx, charge, time in zip(sorted_indices, sorted_charges, sorted_times)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Calculate reasonable axis ranges based on data
        all_x = hit_positions_sorted[:, 0]
        all_y = hit_positions_sorted[:, 1]
        all_z = hit_positions_sorted[:, 2]
        
        margin = 0.1 * max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))
        
        # Update layout for full black display
        margin_right = 80 if show_colorbar else 0
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    visible=False,  # Hide x-axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    range=[np.min(all_x) - margin, np.max(all_x) + margin]
                ),
                yaxis=dict(
                    visible=False,  # Hide y-axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    range=[np.min(all_y) - margin, np.max(all_y) + margin]
                ),
                zaxis=dict(
                    visible=False,  # Hide z-axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    range=[np.min(all_z) - margin, np.max(all_z) + margin]
                ),
                aspectmode='data',
                bgcolor=paper_color,  # Pure black background
            ),
            height=800,
            width=1000,
            showlegend=False,  # Remove legend for cleaner look
            paper_bgcolor=paper_color,  # Theme-based paper background
            plot_bgcolor=paper_color,   # Theme-based plot background
            margin=dict(l=0, r=margin_right, t=0, b=0)  # Minimal margins, space for colorbar if needed
        )
        
        fig.show()

    def visualize_cylinder_wireframe(self, show_detectors=True):
        """Visualize the cylinder as a wireframe with detectors"""
        fig = go.Figure()

        # Create cylinder wireframe
        # Barrel surface
        theta = np.linspace(0, 2 * np.pi, 50)
        z_barrel = np.linspace(-self.H/2, self.H/2, 20)
        theta_mesh, z_mesh = np.meshgrid(theta, z_barrel)
        
        x_barrel = self.r * np.cos(theta_mesh) + self.C[0]
        y_barrel = self.r * np.sin(theta_mesh) + self.C[1]
        z_barrel_mesh = z_mesh + self.C[2]

        # Add barrel wireframe
        fig.add_trace(go.Surface(
            x=x_barrel, y=y_barrel, z=z_barrel_mesh,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            name='Barrel Surface'
        ))

        # Top cap
        r_cap = np.linspace(0, self.r, 20)
        theta_cap = np.linspace(0, 2 * np.pi, 50)
        r_mesh, theta_mesh = np.meshgrid(r_cap, theta_cap)
        
        x_top = r_mesh * np.cos(theta_mesh) + self.C[0]
        y_top = r_mesh * np.sin(theta_mesh) + self.C[1]
        z_top = np.full_like(x_top, self.H/2 + self.C[2])

        # Add top cap wireframe
        fig.add_trace(go.Surface(
            x=x_top, y=y_top, z=z_top,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
            name='Top Cap'
        ))

        # Bottom cap
        z_bottom = np.full_like(x_top, -self.H/2 + self.C[2])

        # Add bottom cap wireframe
        fig.add_trace(go.Surface(
            x=x_top, y=y_top, z=z_bottom,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightcoral'], [1, 'lightcoral']],
            name='Bottom Cap'
        ))

        if show_detectors:
            # Color code detectors by type
            barrel_indices = [i for i, case in self.ID_to_case.items() if case == 0]
            tcap_indices = [i for i, case in self.ID_to_case.items() if case == 1]
            bcap_indices = [i for i, case in self.ID_to_case.items() if case == 2]
            
            if barrel_indices:
                barrel_points = self.all_points[barrel_indices]
                fig.add_trace(go.Scatter3d(
                    x=barrel_points[:, 0], 
                    y=barrel_points[:, 1], 
                    z=barrel_points[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='blue', opacity=0.8),
                    name=f'Barrel Detectors ({len(barrel_indices)})'
                ))
            
            if tcap_indices:
                tcap_points = self.all_points[tcap_indices]
                fig.add_trace(go.Scatter3d(
                    x=tcap_points[:, 0], 
                    y=tcap_points[:, 1], 
                    z=tcap_points[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='green', opacity=0.8),
                    name=f'Top Cap Detectors ({len(tcap_indices)})'
                ))
            
            if bcap_indices:
                bcap_points = self.all_points[bcap_indices]
                fig.add_trace(go.Scatter3d(
                    x=bcap_points[:, 0], 
                    y=bcap_points[:, 1], 
                    z=bcap_points[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='red', opacity=0.8),
                    name=f'Bottom Cap Detectors ({len(bcap_indices)})'
                ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title=f'Cylindrical Detector Geometry (R={self.r}, H={self.H})',
            height=800
        )

        fig.show()


def fibonacci_sphere_points_numpy(n_points, radius=1.0):
    """Generate approximately equidistant points on sphere surface using Fibonacci spiral.
    
    Parameters
    ----------
    n_points : int
        Number of points to generate
    radius : float, optional
        Radius of the sphere, default 1.0
        
    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing point coordinates
    """
    center = np.array([0.0, 0.0, 0.0])
    
    indices = np.arange(0, n_points, dtype=float)
    
    # Golden ratio
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    # Fibonacci spiral algorithm
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * indices / n_points)
    
    # Convert to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    points = np.stack([x, y, z], axis=1) + center
    
    return points


class Sphere:
    """Manage the spherical detector geometry"""
    def __init__(self, radius, n_sensors, sensor_radius):
        self.C = np.array([0.0, 0.0, 0.0])  # Always centered at origin
        self.r = radius
        self.n_sensors = n_sensors
        self.S_radius = sensor_radius

        self.place_photosensors()

    def place_photosensors(self):
        """Position the photo sensor centers on the sphere surface using Fibonacci spiral."""
        self.all_points = fibonacci_sphere_points_numpy(self.n_sensors, self.r) + self.C
        
        # Create ID to position dictionary
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # For sphere, all sensors are on surface (case 0)
        self.ID_to_case = {i: 0 for i in range(len(self.all_points))}

    def visualize_event_data_plotly(self, loaded_indices, loaded_charges, loaded_times, 
                                   plot_time=False, log_scale=False, title=None, 
                                   show_all_detectors=False, marker_size=6, show_colorbar=True,
                                   opacity=1.0, dark_theme=True):
        """
        Visualize detector event data in 3D using Plotly with colors proportional to charge or time.
        
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
        show_all_detectors : bool, default=False
            If True, shows all detector positions as light gray background points
        marker_size : int, default=6
            Size of the markers for hit detectors
        show_colorbar : bool, default=True
            If True, shows the colorbar; if False, creates minimal display with just sensors
        opacity : float, default=1.0
            Opacity of the markers (1.0 = fully opaque, avoids depth sorting issues)
        dark_theme : bool, default=True
            If True, use black background; if False, use white background
        """
        import plotly.graph_objects as go
        import numpy as np
        
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
        
        if len(loaded_indices) == 0:
            print("No event data to display")
            return
        
        # Get positions of hit detectors
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
        
        # Create the plot
        fig = go.Figure()
        
        # Optionally add all detector positions as background
        if show_all_detectors:
            fig.add_trace(go.Scatter3d(
                x=self.all_points[:, 0], 
                y=self.all_points[:, 1], 
                z=self.all_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2, 
                    color='lightgray', 
                    opacity=0.3
                ),
                name='All Detectors',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Sort points by depth (distance from center) for better transparency rendering
        #distances = np.linalg.norm(hit_positions - self.C, axis=1)
        depth_order = np.argsort(hit_positions[:, 2])
        hit_positions_sorted = hit_positions[depth_order]
        plot_color_values_sorted = plot_color_values[depth_order]
        sorted_indices = loaded_indices[depth_order]
        sorted_charges = loaded_charges[depth_order] 
        sorted_times = loaded_times[depth_order]
        
        # Create marker dictionary with full opacity to avoid transparency issues
        marker_dict = dict(
            size=marker_size,
            color=plot_color_values_sorted,
            colorscale='viridis',
            opacity=opacity,  # Use parameter-controlled opacity
            line=dict(width=0, color='rgba(0,0,0,0)')  # Remove outlines for cleaner look
        )
        
        # Add colorbar only if requested
        if show_colorbar:
            marker_dict['colorbar'] = dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(color=colorbar_color)
                ),
                tickfont=dict(color=colorbar_color),
                thickness=20,
                len=0.7,
                x=1.02
            )
        
        # Add scatter plot for hit detectors
        fig.add_trace(go.Scatter3d(
            x=hit_positions_sorted[:, 0], 
            y=hit_positions_sorted[:, 1], 
            z=hit_positions_sorted[:, 2],
            mode='markers',
            marker=marker_dict,
            name=f'Event Data ({len(loaded_indices)} hits)',
            text=[f'Detector ID: {idx}<br>Charge: {charge:.3f} PE<br>Time: {time:.3f} ns' 
                  for idx, charge, time in zip(sorted_indices, sorted_charges, sorted_times)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Calculate reasonable axis ranges based on data
        all_x = hit_positions_sorted[:, 0]
        all_y = hit_positions_sorted[:, 1]
        all_z = hit_positions_sorted[:, 2]
        
        margin = 0.1 * max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))
        
        # Update layout for clean display
        margin_right = 80 if show_colorbar else 0
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    visible=False,  # Hide x-axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    range=[np.min(all_x) - margin, np.max(all_x) + margin]
                ),
                yaxis=dict(
                    visible=False,  # Hide y-axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    range=[np.min(all_y) - margin, np.max(all_y) + margin]
                ),
                zaxis=dict(
                    visible=False,  # Hide z-axis completely
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    range=[np.min(all_z) - margin, np.max(all_z) + margin]
                ),
                aspectmode='cube',  # Important for sphere to maintain shape
                bgcolor=paper_color,  # Theme-based background
            ),
            height=800,
            width=1000,
            showlegend=False,  # Remove legend for cleaner look
            paper_bgcolor=paper_color,  # Theme-based paper background
            plot_bgcolor=paper_color,   # Theme-based plot background
            margin=dict(l=0, r=margin_right, t=0, b=0)  # Minimal margins, space for colorbar if needed
        )
        
        fig.show()

    def visualize_sphere_wireframe(self, show_detectors=True):
        """Visualize the sphere as a wireframe with detectors"""
        fig = go.Figure()

        # Create sphere wireframe
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.r * np.outer(np.cos(u), np.sin(v)) + self.C[0]
        y_sphere = self.r * np.outer(np.sin(u), np.sin(v)) + self.C[1]
        z_sphere = self.r * np.outer(np.ones(np.size(u)), np.cos(v)) + self.C[2]

        # Add wireframe
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            name='Sphere Surface'
        ))

        if show_detectors:
            fig.add_trace(go.Scatter3d(
                x=self.all_points[:, 0], 
                y=self.all_points[:, 1], 
                z=self.all_points[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.8),
                name=f'Detectors ({self.n_sensors})'
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            title=f'Spherical Detector Geometry (R={self.r})',
            height=800
        )

        fig.show()


# Unified configuration and generation functions

def load_detector_config(file_path):
    """Function to load detector configuration from JSON file"""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def load_detector_geom(file_path):
    """Load detector geometry from JSON config"""
    config = load_detector_config(file_path)
    
    detector_type = config['detector_type']
    geom_def = config['geometry_definitions']
    
    if detector_type == 'cylinder':
        return (detector_type, geom_def['radius'], geom_def['height'], 
                geom_def['n_sensors'], geom_def['sensor_radius'])
    elif detector_type == 'sphere':
        return (detector_type, geom_def['radius'], None, 
                geom_def['n_sensors'], geom_def['sensor_radius'])
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

def generate_detector(file_path):
    """Function to generate detector from json config"""
    detector_type, radius, height, n_sensors, sensor_radius = load_detector_geom(file_path)
    
    if detector_type == 'cylinder':
        return Cylinder(radius, height, n_sensors, sensor_radius)
    elif detector_type == 'sphere':
        return Sphere(radius, n_sensors, sensor_radius)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

def generate_detector_direct(detector_type, radius, n_sensors, sensor_radius, height=None):
    """Function to generate detector directly with parameters"""
    if detector_type == 'cylinder':
        if height is None:
            raise ValueError("Height must be specified for cylinder detector")
        return Cylinder(radius, height, n_sensors, sensor_radius)
    elif detector_type == 'sphere':
        return Sphere(radius, n_sensors, sensor_radius)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
