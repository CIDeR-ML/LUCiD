import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools.geometry import load_cyl_geom, generate_detector

def create_color_gradient(max_cnts, colormap='viridis'):
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=max_cnts)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def calculate_min_distance(positions):
    distances = pdist(positions)
    return np.min(distances) if len(distances) > 0 else 1.0


def create_detector_display(json_filename='config/cyl_geom_config.json'):
    # Generate detector
    detector = generate_detector(json_filename)

    # Load geometry data
    cyl_center, cyl_axis, cyl_radius, _, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius = load_cyl_geom(
        json_filename)

    # Set up detector information
    detector_positions = np.array(detector.all_points)
    detector_cases = np.array([detector.ID_to_case[i] for i in range(len(detector.all_points))])

    def display_detector_data(loaded_indices, loaded_charges, loaded_times, file_name=None, plot_time=False):
        """
        Process detector data and create a 2D event display.

        Parameters:
        - loaded_indices: numpy array of indices of non-zero hits
        - loaded_charges: numpy array of charge values corresponding to loaded_indices
        - loaded_times: numpy array of time values corresponding to loaded_indices
        - file_name: optional, if provided, saves the plot to this file
        - plot_time: boolean, if True, plot time instead of charge
        """
        n_detectors = len(detector_positions)

        # Initialize arrays for all detectors
        all_charges = np.zeros(n_detectors)
        all_times = np.zeros(n_detectors)
        all_charges[loaded_indices] = loaded_charges
        all_times[loaded_indices] = loaded_times

        # Select which values to plot based on plot_time
        all_values = all_times if plot_time else all_charges

        max_value = np.max(all_values)
        color_gradient = create_color_gradient(max_value)

        corr = cyl_radius / cyl_height
        caps_offset = -0.1

        # Calculate positions for all cases
        x = np.zeros(n_detectors)
        y = np.zeros(n_detectors)

        # Barrel case (0)
        barrel_mask = detector_cases == 0
        theta = np.arctan2(detector_positions[barrel_mask, 1], detector_positions[barrel_mask, 0])
        theta = (theta + np.pi * 3 / 2) % (2 * np.pi) / 2
        x[barrel_mask] = theta
        y[barrel_mask] = detector_positions[barrel_mask, 2] / cyl_height

        # Top cap case (1)
        top_mask = detector_cases == 1
        x[top_mask] = corr * detector_positions[top_mask, 0] / cyl_height + np.pi / 2
        y[top_mask] = 1 + corr * (caps_offset + detector_positions[top_mask, 1] / cyl_height)

        # Bottom cap case (2)
        bottom_mask = detector_cases == 2
        x[bottom_mask] = corr * detector_positions[bottom_mask, 0] / cyl_height + np.pi / 2
        y[bottom_mask] = -1 + corr * (-caps_offset - detector_positions[bottom_mask, 1] / cyl_height)

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
                                 facecolors=color_gradient.to_rgba(all_values),
                                 offsets=transformed_positions,
                                 transOffset=ax.transData)

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
        cbar.set_label('Time' if plot_time else 'Photoelectron Count (a.u.)', color='white', fontsize=18)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Adjust layout
        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1, facecolor='black', edgecolor='none')
        plt.show()

    return display_detector_data