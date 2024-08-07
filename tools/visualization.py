import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_color_gradient(max_cnts, colormap='viridis'):
    """Define the color scale in the 2D event display"""
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=max_cnts)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def calculate_min_distance(positions):
    """Calculate the minimum distance between any two points"""
    distances = pdist(positions)
    return np.min(distances)


def show_2D_display(ID_to_position, ID_to_PE, ID_to_case, cyl_radius, cyl_height, file_name=None, plot_time=False):
    """Do the 2D event display with circles touching each other and a colorbar"""
    max_PE = max(ID_to_PE.values())
    color_gradient = create_color_gradient(max_PE)

    corr = cyl_radius / cyl_height
    caps_offset = -0.1

    # Convert dictionaries to numpy arrays for vectorized operations
    positions = np.array([ID_to_position[ID] for ID in ID_to_position])
    PEs = np.array([ID_to_PE[ID] for ID in ID_to_position])
    cases = np.array([ID_to_case[ID] for ID in ID_to_position])

    # Calculate positions for all cases
    x = np.zeros_like(PEs)
    y = np.zeros_like(PEs)

    # Barrel case (0)
    barrel_mask = cases == 0
    theta = np.arctan2(positions[barrel_mask, 1], positions[barrel_mask, 0])
    theta = (theta + np.pi / 2) % (2 * np.pi) / 2
    x[barrel_mask] = theta
    y[barrel_mask] = positions[barrel_mask, 2] / cyl_height

    # Top cap case (1)
    top_mask = cases == 1
    x[top_mask] = corr * positions[top_mask, 0] / cyl_height + np.pi / 2
    y[top_mask] = 1 + corr * (caps_offset + positions[top_mask, 1] / cyl_height)

    # Bottom cap case (2)
    bottom_mask = cases == 2
    x[bottom_mask] = corr * positions[bottom_mask, 0] / cyl_height + np.pi / 2
    y[bottom_mask] = -1 + corr * (-caps_offset - positions[bottom_mask, 1] / cyl_height)

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
    fig_width = 8  # Base width in inches, increased to accommodate colorbar
    fig_height = fig_width * (y_range / x_range)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='black')

    # Create EllipseCollection
    ells = EllipseCollection(widths=circle_diameter, heights=circle_diameter, angles=0, units='x',
                             facecolors=color_gradient.to_rgba(PEs),
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
    if plot_time:
        cbar.set_label('Averaged Time', color='white', fontsize=12)
    else:
        cbar.set_label('Photoelectrons', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Adjust layout
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1, facecolor='black', edgecolor='none')
    plt.show()