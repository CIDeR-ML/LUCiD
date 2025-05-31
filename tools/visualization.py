import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools.geometry import load_cyl_geom, generate_detector
from tools.utils import sparse_to_full
from matplotlib.colors import LinearSegmentedColormap

def create_color_gradient(max_cnts, colormap='viridis'):
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=max_cnts)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def calculate_min_distance(positions):
    distances = pdist(positions)
    return np.min(distances) if len(distances) > 0 else 1.0


def create_detector_display(json_filename='config/cyl_geom_config.json', sparse=True):
    """
    Create a detector display function that can handle both sparse and dense data formats.

    Parameters:
    -----------
    json_filename : str
        Path to the configuration file for detector geometry
    sparse : bool
        If True, function expects sparse data format (indices, charges, times)
        If False, function expects dense data format (full arrays)

    Returns:
    --------
    function
        Display function that can be called with appropriate data format
    """
    # Generate detector
    detector = generate_detector(json_filename)

    # Load geometry data
    cyl_center, cyl_axis, cyl_radius, _, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius = load_cyl_geom(
        json_filename)

    # Set up detector information
    detector_positions = np.array(detector.all_points)
    detector_cases = np.array([detector.ID_to_case[i] for i in range(len(detector.all_points))])
    n_detectors = len(detector_positions)

    def display_detector_data(*args, file_name=None, plot_time=False, log_scale=False):
        """
        Process and display detector data in either sparse or dense format.

        Parameters (for sparse=True):
        ---------------------------
        loaded_indices : array-like
            Indices of non-zero hits
        loaded_charges : array-like
            Charge values at non-zero indices
        loaded_times : array-like
            Time values at non-zero indices

        Parameters (for sparse=False):
        ---------------------------
        charges : array-like
            Full array of charge values
        times : array-like
            Full array of time values

        Other Parameters:
        ----------------
        file_name : str, optional
            If provided, saves the plot to this file
        plot_time : bool
            If True, plot time instead of charge
        log_scale : bool
            If True, apply logarithmic scaling to the color gradient
        """
        if sparse:
            if len(args) != 3:
                raise ValueError("Sparse format requires three arguments: indices, charges, and times")
            loaded_indices, loaded_charges, loaded_times = args

            # Convert sparse to full arrays
            all_charges = sparse_to_full(loaded_indices, loaded_charges, n_detectors)
            all_times = sparse_to_full(loaded_indices, loaded_times, n_detectors)
        else:
            if len(args) != 2:
                raise ValueError("Dense format requires two arguments: charges and times arrays")
            all_charges, all_times = args

        # Select which values to plot based on plot_time
        all_values = all_times if plot_time else all_charges

        # Generate color gradient based on scale type
        max_value = np.max(all_values)
        if log_scale:
            # Handle zero values for log scale - set to small positive number
            min_positive = np.min(all_values[all_values > 0]) if np.any(all_values > 0) else 1.0
            vmin = min_positive * 0.1  # Set minimum to fraction of smallest positive value

            # Create a copy of values for color mapping
            plot_values = np.copy(all_values)
            plot_values[plot_values <= 0] = vmin  # Replace zeros/negatives with minimum

            # Use LogNorm for logarithmic color scaling
            cmap = plt.get_cmap('viridis')
            norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=max_value)
            color_gradient = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            # Linear scaling (original behavior)
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=0, vmax=max_value)
            color_gradient = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            plot_values = all_values

        corr = 1.
        caps_offset = cyl_height

        # Calculate positions for all cases
        x = np.zeros(n_detectors)
        y = np.zeros(n_detectors)

        # Barrel case (0)
        barrel_mask = detector_cases == 0
        theta = np.arctan2(detector_positions[barrel_mask, 1], detector_positions[barrel_mask, 0])
        theta = (theta + np.pi * 3 / 2) % (2 * np.pi) / 2
        x[barrel_mask] = theta * cyl_radius*2
        y[barrel_mask] = detector_positions[barrel_mask, 2]

        # Top cap case (1)
        top_mask = detector_cases == 1
        x[top_mask] = corr * detector_positions[top_mask, 0] + np.pi * cyl_radius
        y[top_mask] = 1 + corr * (caps_offset + detector_positions[top_mask, 1])

        # Bottom cap case (2)
        bottom_mask = detector_cases == 2
        x[bottom_mask] = corr * detector_positions[bottom_mask, 0] + np.pi * cyl_radius
        y[bottom_mask] = -1 + corr * (-caps_offset - detector_positions[bottom_mask, 1] )

        # Calculate the minimum distance between points in the transformed space
        transformed_positions = np.column_stack((x, y))
        min_distance = calculate_min_distance(transformed_positions)

        # Set the circle diameter to be equal to the minimum distance
        circle_diameter = min_distance

        # Calculate exact dimensions needed
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # Add padding
        padding = circle_diameter
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Set figure size based on data range, accounting for colorbar
        fig_width = 12
        fig_height = fig_width * (y_range / x_range)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='black')

        # Create EllipseCollection
        ells = EllipseCollection(widths=circle_diameter, heights=circle_diameter, angles=0, units='x',
                                 facecolors=color_gradient.to_rgba(plot_values),
                                 offsets=transformed_positions,
                                 transOffset=ax.transData,
                                 edgecolors='none')

        ax.add_collection(ells)

        ax.set_facecolor("black")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

        # Remove axes
        ax.axis('off')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(color_gradient, cax=cax)
        value_label = 'Time' if plot_time else 'Photoelectron Count (a.u.)'
        scale_label = ' (log scale)' if log_scale else ''
        cbar.set_label(f'{value_label}{scale_label}', color='white', fontsize=18)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Adjust layout
        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1, facecolor='black', edgecolor='none')
        plt.show()

    return display_detector_data

def create_detector_comparison_display(json_filename='config/cyl_geom_config.json', sparse=True):
    """
    Create a detector comparison display function that can handle both sparse and dense data formats.
    Specifically designed for showing differences with a diverging color scale centered at zero.

    Parameters:
    -----------
    json_filename : str
        Path to the configuration file for detector geometry
    sparse : bool
        If True, function expects sparse data format (indices, charges, times)
        If False, function expects dense data format (full arrays)

    Returns:
    --------
    function
        Display function that can be called with appropriate data format
    """
    # Generate detector
    detector = generate_detector(json_filename)

    # Load geometry data
    cyl_center, cyl_axis, cyl_radius, _, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius = load_cyl_geom(
        json_filename)

    # Set up detector information
    detector_positions = np.array(detector.all_points)
    detector_cases = np.array([detector.ID_to_case[i] for i in range(len(detector.all_points))])
    n_detectors = len(detector_positions)

    def display_detector_data(true_data, sim_data, file_name=None, plot_time=False, align_time=False, colorbar_range=None):
        """
        Process and display detector comparison data, handling both true and simulated data.

        Parameters:
        -----------
        true_data : tuple
            For sparse=True: (indices, charges, times)
            For sparse=False: (charges, times)
        sim_data : tuple
            Same format as true_data
        file_name : str, optional
            If provided, saves the plot to this file
        plot_time : bool
            If True, plot time differences instead of charge differences
        align_time : bool
            If True, subtract mean from both times arrays respectively
        colorbar_range : float, optional
            If provided, sets the symmetric colorbar range to [-colorbar_range, +colorbar_range]
            If None, calculates range from current data (default behavior)
        """
        if sparse:
            # Unpack sparse data
            true_indices, true_charges, true_times = true_data
            sim_indices, sim_charges, sim_times = sim_data

            # Convert to full arrays
            true_charges_full = sparse_to_full(true_indices, true_charges, n_detectors)
            sim_charges_full = sparse_to_full(sim_indices, sim_charges, n_detectors)
            true_times_full = sparse_to_full(true_indices, true_times, n_detectors)
            sim_times_full = sparse_to_full(sim_indices, sim_times, n_detectors)
        else:
            # Data is already in full format
            true_charges_full, true_times_full = true_data
            sim_charges_full, sim_times_full = sim_data

        # Calculate charge differences
        charge_diff = sim_charges_full - true_charges_full

        # Handle time differences with alignment if requested
        if align_time:
            # Find active time points
            active_times_true = true_times_full > 0
            active_times_sim = sim_times_full > 0

            # Calculate means for active times
            true_time_mean = np.mean(true_times_full[active_times_true]) if np.any(active_times_true) else 0
            sim_time_mean = np.mean(sim_times_full[active_times_sim]) if np.any(active_times_sim) else 0

            # Subtract means from active times
            true_times_aligned = np.where(active_times_true, true_times_full - true_time_mean, 0)
            sim_times_aligned = np.where(active_times_sim, sim_times_full - sim_time_mean, 0)

            time_diff = sim_times_aligned - true_times_aligned
        else:
            time_diff = sim_times_full - true_times_full

        # Select which values to plot
        all_values = time_diff if plot_time else charge_diff

        # Determine colorbar range
        if colorbar_range is not None:
            max_abs_value = colorbar_range
        else:
            # Find maximum absolute value for symmetric color scale (original behavior)
            max_abs_value = np.max(np.abs(all_values))

        # Create colors array: viridis(0) for non-active, diverging colormap for active
        viridis = plt.cm.viridis
        diverging_cmap = plt.cm.seismic

        # Create color array
        colors = np.zeros((len(all_values), 4))  # RGBA array
        active_mask = all_values != 0

        # Set non-active detectors to viridis(0)
        colors[~active_mask] = viridis(0)

        # Set active detectors using diverging colormap
        norm = plt.Normalize(-max_abs_value, max_abs_value)
        colors[active_mask] = diverging_cmap(norm(all_values[active_mask]))

        # Create color gradient for colorbar only (using only the diverging colormap)
        color_gradient = plt.cm.ScalarMappable(
            norm=norm,
            cmap=diverging_cmap
        )

        corr = 1.
        caps_offset = cyl_height

        # Calculate positions for all cases
        x = np.zeros(n_detectors)
        y = np.zeros(n_detectors)

        # Barrel case (0)
        barrel_mask = detector_cases == 0
        theta = np.arctan2(detector_positions[barrel_mask, 1], detector_positions[barrel_mask, 0])
        theta = (theta + np.pi * 3 / 2) % (2 * np.pi) / 2
        x[barrel_mask] = theta * cyl_radius*2
        y[barrel_mask] = detector_positions[barrel_mask, 2]

        # Top cap case (1)
        top_mask = detector_cases == 1
        x[top_mask] = corr * detector_positions[top_mask, 0] + np.pi * cyl_radius
        y[top_mask] = 1 + corr * (caps_offset + detector_positions[top_mask, 1])

        # Bottom cap case (2)
        bottom_mask = detector_cases == 2
        x[bottom_mask] = corr * detector_positions[bottom_mask, 0] + np.pi * cyl_radius
        y[bottom_mask] = -1 + corr * (-caps_offset - detector_positions[bottom_mask, 1] )

        # Calculate the minimum distance between points in the transformed space
        transformed_positions = np.column_stack((x, y))
        min_distance = calculate_min_distance(transformed_positions)

        # Set the circle diameter to be equal to the minimum distance
        circle_diameter = min_distance

        # Calculate exact dimensions needed
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # Add padding
        padding = circle_diameter
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Set figure size based on data range, accounting for colorbar
        fig_width = 12
        fig_height = fig_width * (y_range / x_range)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='black')

        # Create EllipseCollection with the combined colors
        ells = EllipseCollection(widths=circle_diameter, heights=circle_diameter, angles=0, units='x',
                                 facecolors=colors,
                                 offsets=transformed_positions,
                                 transOffset=ax.transData,
                                 edgecolors='black')

        ax.add_collection(ells)

        ax.set_facecolor("black")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

        # Remove axes
        ax.axis('off')

        # Add colorbar with explicit scientific notation
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(color_gradient, cax=cax, format='%.1e')
        label_text = 'Time Difference' if plot_time else 'Photoelectron Count Difference (a.u.)'
        cbar.set_label(label_text, color='white', fontsize=18)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Adjust layout
        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1,
                        facecolor='black', edgecolor='none')
        plt.show()

    return display_detector_data