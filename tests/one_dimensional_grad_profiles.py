
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from tools.losses import *
from jax import random

def one_dimensional_grad_profiles(detector, true_indices, true_times, detector_points, detector_radius, Nphot, true_params, use_time_loss):
    def test_parameter(param_name, true_params, param_range, param_index=None):
        results = []
        loss_and_grad = jax.value_and_grad(combined_loss_function, argnums=(2, 3, 4))
        
        for i, param_value in enumerate(param_range):
            print(i)
            if param_name == 'cone_opening':
                params = [param_value, true_params[1], true_params[2]]
            elif param_name.startswith('track_origin'):
                params = list(true_params)
                if isinstance(params[1], jnp.ndarray):
                    params[1] = params[1].at[param_index].set(param_value)
                else:  # NumPy array
                    params[1] = params[1].copy()
                    params[1][param_index] = param_value
            elif param_name.startswith('track_direction'):
                params = list(true_params)
                if isinstance(params[2], jnp.ndarray):
                    params[2] = params[2].at[param_index].set(param_value)
                else:  # NumPy array
                    params[2] = params[2].copy()
                    params[2][param_index] = param_value
                params[2] = normalize(params[2])
            
            key = random.PRNGKey(0)
            loss, (grad_cone, grad_origin, grad_direction) = loss_and_grad(
                true_indices, true_times, *params, detector_points, detector_radius, Nphot, key, use_time_loss
            )

            if param_name == 'cone_opening':
                grad = grad_cone
            elif param_name.startswith('track_origin'):
                grad = grad_origin[param_index]
            elif param_name.startswith('track_direction'):
                grad = grad_direction[param_index]
            
            # Generate and store event
            event_filename = f'test_events/{param_name}_step_{i}.h5'
            generate_and_store_event(event_filename, *params, detector, Nphot, key)
            
            results.append({
                'param_value': param_value,
                'loss': loss,
                'grad': grad,
                'event_filename': event_filename
            })
        
        return results

    Nsteps = 2

    # Test 1: Cone opening angle
    cone_results = test_parameter('cone_opening', true_params, jnp.linspace(20, 60, Nsteps))

    # Test 2: X component of track origin
    origin_x_results = test_parameter('track_origin_x', true_params, jnp.linspace(0, 2, Nsteps), 0)

    # Test 3: Y component of track direction
    direction_y_results = test_parameter('track_direction_y', true_params, jnp.linspace(-0.4, 0.4, Nsteps), 1)

    def plot_results(figname, results, title, xlabel, true_value):
        param_values = [r['param_value'] for r in results]
        losses = [r['loss'] for r in results]
        grads = [r['grad'] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        
        # Plot loss
        ax1.plot(param_values, losses)
        ax1.set_ylabel('Loss')
        ax1.set_title(title)
        ax1.axvline(x=true_value, color='r', linestyle='--', label='True Value')
        ax1.legend()
        
        # Plot gradient
        ax2.plot(param_values, grads)
        ax2.set_ylabel('Gradient')
        ax2.set_xlabel(xlabel)
        ax2.axvline(x=true_value, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(figname)
        plt.close(fig)  # Close the figure to free up memory


    true_cone_opening = true_params[0]
    true_track_origin = true_params[1]
    true_track_direction = true_params[2]

    # In the main function, modify the calls to plot_results:
    plot_results('output_plots/test1.pdf', cone_results, 'Cone Opening Angle Test', 'Cone Opening Angle (degrees)', true_cone_opening)
    plot_results('output_plots/test2.pdf', origin_x_results, 'Track Origin X Test', 'Track Origin X', true_track_origin[0])
    plot_results('output_plots/test3.pdf', direction_y_results, 'Track Direction Y Test', 'Track Direction Y', true_track_direction[1])














