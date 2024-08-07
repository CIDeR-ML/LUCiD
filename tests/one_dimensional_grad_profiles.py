
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from tools.losses import *
from jax import random

def one_dimensional_grad_profiles(detector, true_indices, true_cts, true_times, detector_points, detector_radius, detector_height, Nphot, true_params):
    def test_parameter(param_name, true_params, param_range, param_index=None):
        results = []
        loss_and_grad = jax.value_and_grad(smooth_combined_loss_function, argnums=(3, 4, 5, 6, 7, 8, 9, 10))
        
        for i, param_value in enumerate(param_range):
            print(i)
            params = list(true_params)
            
            if param_name == 'reflection_prob':
                params[0] = param_value
            elif param_name == 'cone_opening':
                params[1] = param_value
            elif param_name.startswith('track_origin'):
                if isinstance(params[2], jnp.ndarray):
                    params[2] = params[2].at[param_index].set(param_value)
                else:  # NumPy array
                    params[2] = params[2].copy()
                    params[2][param_index] = param_value
            elif param_name.startswith('track_direction'):
                if isinstance(params[3], jnp.ndarray):
                    params[3] = params[3].at[param_index].set(param_value)
                else:  # NumPy array
                    params[3] = params[3].copy()
                    params[3][param_index] = param_value
                params[3] = normalize(params[3])
            elif param_name == 'photon_norm':
                params[4] = param_value
            elif param_name == 'att_L':
                params[5] = param_value
            elif param_name == 'trk_L':
                params[6] = param_value
            elif param_name == 'scatt_L':
                params[7] = param_value
            
            key = random.PRNGKey(0)
            true_hits = np.zeros(len(detector.all_points))
            true_hits[true_indices] = true_cts
            loss, (grad_refl_prob, grad_cone, grad_origin, grad_direction, grad_photon_norm, grad_att_L, grad_trk_L, grad_scatt_L) = loss_and_grad(
                true_indices, true_hits, true_times, *params, detector_points, detector_radius, detector_height, Nphot, key
            )

            if param_name == 'reflection_prob':
                grad = grad_refl_prob
            elif param_name == 'cone_opening':
                grad = grad_cone
            elif param_name.startswith('track_origin'):
                grad = grad_origin[param_index]
            elif param_name.startswith('track_direction'):
                grad = grad_direction[param_index]
            elif param_name == 'photon_norm':
                grad = grad_photon_norm
            elif param_name == 'att_L':
                grad = grad_att_L
            elif param_name == 'trk_L':
                grad = grad_trk_L
            elif param_name == 'scatt_L':
                grad = grad_scatt_L
            
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

    Nsteps = 11


    # # Test 1: Cone opening angle
    # cone_results = test_parameter('cone_opening', true_params, jnp.linspace(30, 50, Nsteps))

    # # Test 2: X component of track origin
    # origin_x_results = test_parameter('track_origin_x', true_params, jnp.linspace(0, 2, Nsteps), 0)

    # # Test 3: Y component of track direction
    # direction_y_results = test_parameter('track_direction_y', true_params, jnp.linspace(-0.4, 0.4, Nsteps), 1)

    # # Test 4: Reflection probability
    refl_prob_results = test_parameter('reflection_prob', true_params, jnp.linspace(0.15, 0.55, Nsteps), 1)

    # # Test 5: Number of photons
    photon_norm_results = test_parameter('photon_norm', true_params, jnp.linspace(0.5, 1.5, Nsteps))

    # # Test 6: Attenuation length (att_L)
    # att_L_results = test_parameter('att_L', true_params, jnp.linspace(1, 5, Nsteps))

    # # Test 7: Track length (trk_L)
    # trk_L_results = test_parameter('trk_L', true_params, jnp.linspace(0.5, 1.5, Nsteps))

    # # Test 8: Scattering length (scatt_L)
    # scatt_L_results = test_parameter('scatt_L', true_params, jnp.linspace(1, 5, Nsteps))

    def plot_results(figname, results, title, xlabel, true_value):
        param_values = [r['param_value'] for r in results]
        losses = [r['loss'] for r in results]
        grads = [r['grad'] for r in results]
        print(min(grads), max(grads))
        print(grads)

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


    true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction, \
    true_photon_norm, true_att_L, true_trk_L, true_scatt_L = true_params

    # In the main function, modify the calls to plot_results:
    plot_results('output_plots/test0.png', refl_prob_results, 'True Reflection Prob Test', 'Reflection Probability', true_reflection_prob)
    # plot_results('output_plots/test1.png', cone_results, 'Cone Opening Angle Test', 'Cone Opening Angle (degrees)', true_cone_opening)
    # plot_results('output_plots/test2.png', origin_x_results, 'Track Origin X Test', 'Track Origin X', true_track_origin[0])
    # plot_results('output_plots/test3.png', direction_y_results, 'Track Direction Y Test', 'Track Direction Y', true_track_direction[1])
    plot_results('output_plots/test4.png', photon_norm_results, 'Number of Photons Test', 'Number of Photons', true_photon_norm)
    # plot_results('output_plots/test5.png', att_L_results, 'Attenuation Length Test', 'Attenuation Length', true_att_L)
    # plot_results('output_plots/test6.png', trk_L_results, 'Track Length Test', 'Track Length', true_trk_L)
    # plot_results('output_plots/test7.png', scatt_L_results, 'Scattering Length Test', 'Scattering Length', true_scatt_L)













