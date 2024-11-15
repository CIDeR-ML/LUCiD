import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import matplotlib.pyplot as plt
import os, sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector
from tools.utils import load_single_event, save_single_event, generate_random_params, print_params
from tools.losses import compute_loss
from tools.simulation import setup_event_simulator


def analyze_gradient_profiles(true_params, param_changes, simulate_event, detector_points,
                              true_data, filename_prefix, num_points=121):
    """Analyze and save gradient profiles for different parameters.

    Parameters
    ----------
    true_params : tuple
        Tuple containing (opening_angle, position, direction, intensity)
        where opening_angle and intensity are scalars,
        position and direction are 3D vectors
    param_changes : tuple
        Tuple containing variations for each parameter in same format as true_params
    simulate_event : callable
        Function that simulates events given parameters
    detector_points : ndarray
        Array of detector point coordinates
    true_data : tuple
        Tuple containing (indices, charges, times) of true event data
    filename_prefix : str
        Prefix for saved files. Will save as prefix_parameter.npy and prefix_plots.png
    num_points : int, optional
        Number of points to sample for each parameter, default 121

    Returns
    -------
    None
        Saves files:
            - {prefix}_opening_angle_data.npy
            - {prefix}_position_x_data.npy
            - {prefix}_direction_x_data.npy
            - {prefix}_intensity_data.npy
            - {prefix}_gradient_profiles.png

    Notes
    -----
    Each .npy file contains a dictionary with:
        - parameter_values: The x-axis values
        - losses: The corresponding loss values 
        - gradients: The corresponding gradient values
    """
    # Create directory if it doesn't exist

    key = jax.random.PRNGKey(0)

    @jit
    def loss_and_grad(params):
        """Compute loss and gradient for given parameters."""

        def loss_fn(params):
            simulated_data = simulate_event(params, key)
            return compute_loss(detector_points, *true_data, *simulated_data)

        return value_and_grad(loss_fn)(params)

    def generate_param_ranges(true_params, param_changes, num_points):
        """Generate parameter ranges for analysis."""
        param_ranges = []
        for i, (true_param, change) in enumerate(zip(true_params, param_changes)):
            if i in [1, 2]:  # position and direction
                start = true_param[0] - change[0]
                end = true_param[0] + change[0]
            else:  # opening angle and intensity
                start = true_param - change
                end = true_param + change
            param_ranges.append(jnp.linspace(start, end, num_points))
        return param_ranges

    def generate_plot_data(param_index, param_values):
        """Generate loss and gradient data for plotting."""
        losses = []
        gradients = []

        for new_value in param_values:
            new_params = list(true_params)
            if param_index in [1, 2]:  # position and direction
                new_params[param_index] = new_params[param_index].at[0].set(new_value)
            else:  # opening angle and intensity
                new_params[param_index] = new_value
            new_params = tuple(new_params)

            loss, grad = loss_and_grad(new_params)
            gradient = grad[param_index]
            if param_index in [1, 2]:
                gradient = gradient[0]

            losses.append(loss)
            gradients.append(gradient)

        return jnp.array(losses), jnp.array(gradients)

    # Generate parameter ranges
    param_ranges = generate_param_ranges(true_params, param_changes, num_points)
    param_names = ['Opening Angle', 'Position X', 'Direction X', 'Intensity']

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Analyze each parameter
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        param_values = param_ranges[i]
        losses, gradients = generate_plot_data(i, param_values)

        # Save numerical data
        data_dict = {
            'parameter_values': param_values,
            'losses': losses,
            'gradients': gradients
        }
        save_path = f"{filename_prefix}_{param_names[i].lower().replace(' ', '_')}_data.npy"
        jnp.save(save_path, data_dict)

        # Plot
        ax1 = axs[row, col]
        ax2 = ax1.twinx()

        ax1.plot(param_values, losses, 'b-', label='Loss')
        ax2.plot(param_values, gradients, 'm-', label='Gradient')

        # Add vertical line at the center (true parameter value)
        true_value = true_params[i] if i not in [1, 2] else true_params[i][0]
        ax1.axvline(x=true_value, color='b', linestyle='--', label='True Value')

        # Add horizontal line for gradient at zero
        ax2.axhline(y=0, color='m', linestyle=':', label='Gradient = 0')

        ax1.set_xlabel(f'{param_names[i]} Value')
        ax1.set_ylabel('Loss', color='b')
        ax2.set_ylabel('Gradient', color='m')

        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='m')

        ax1.set_title(f'Loss and Gradient for {param_names[i]}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_gradient_profiles.png")
    plt.close()


def main(random_params=False):
    """Run gradient profile analysis.

    Parameters
    ----------
    random_params : bool, optional
        If True, generate random parameters for true values, default False
    """
    # Setup parameters
    default_json_filename = 'config/cyl_geom_config.json'

    detector = generate_detector(default_json_filename)
    detector_points = jnp.array(detector.all_points)
    NUM_DETECTORS = len(detector_points)
    Nphot = 1_000_000
    temperature = 100.0

    # Setup simulator
    simulate_event = setup_event_simulator(default_json_filename, Nphot, temperature)

    # Define parameters based on random flag
    current_time = int(time.time() * 1e9)

    # Create a PRNGKey using the current time
    key = jax.random.PRNGKey(current_time)
    if random_params:
        true_params = generate_random_params(key, L=2)
        print_params(true_params)
    else:
        true_params = (
            jnp.array(40.0),  # opening angle
            jnp.array([0.5, 0.0, 1.0]),  # position
            jnp.array([0.0, 1.0, 0.0]),  # direction
            jnp.array(5.0)  # intensity
        )

    # Define parameter changes
    param_changes = (
        jnp.array(10.0),  # opening angle
        jnp.array([1.0, 0.0, 0.0]),  # position (only changing first component)
        jnp.array([1.0, 0.0, 0.0]),  # direction (only changing first component)
        jnp.array(2.0)  # intensity
    )

    # Generate and save true data
    key = jax.random.PRNGKey(0)
    true_data_temp = jax.lax.stop_gradient(simulate_event(true_params, key))

    # Save data
    save_single_event(true_data_temp, true_params, filename='events/true_event_data.h5')

    # Load data (excluding true_params)
    true_data = load_single_event('events/true_event_data.h5', NUM_DETECTORS, sparse=False)[1:]

    # Analyze and save gradient profiles
    analyze_gradient_profiles(
        true_params=true_params,
        param_changes=param_changes,
        simulate_event=simulate_event,
        detector_points=detector_points,
        true_data=true_data,
        filename_prefix='outputs'
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate gradient profiles')
    parser.add_argument('--random_params', action='store_true', help='Use random parameters for true values')
    args = parser.parse_args()

    main(random_params=args.random_params)