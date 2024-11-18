from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
import time
from tools.losses import *
from tools.utils import *

loss_and_grad = jax.value_and_grad(smooth_combined_loss_function, argnums=(3, 4, 5, 6, 7, 8, 9, 10))

def optimize_params(detector, true_cts, true_indices, true_times, true_reflection_prob, true_cone_opening, true_track_origin, \
    true_track_direction, true_photon_norm, true_att_L, true_trk_L, true_scatt_L, \
    reflection_prob, cone_opening, track_origin, track_direction, photon_norm, att_L, trk_L, scatt_L, Nphot):

    log = Logger()
    log.true_ref_prob = true_reflection_prob
    log.true_dir = true_track_direction
    log.true_ori = true_track_origin
    log.true_ch_angle = true_cone_opening
    log.true_ref_prob = true_reflection_prob
    log.true_photon_norm = true_photon_norm
    log.true_att_L = true_att_L
    log.true_trk_L = true_trk_L
    log.true_scatt_L = true_scatt_L

    print('--------------------')
    print('true_refl_prob:     ',  true_reflection_prob)
    print('true_ch_angle:      ',  true_cone_opening)
    print('true_photon_norm:   ',  true_photon_norm)
    print('true_att_L:         ',  true_att_L)
    print('true_trk_L:         ',  true_trk_L)
    print('true_scatt_L:       ',  true_scatt_L)
    print('true_trk_origin:    ',  true_track_origin)
    print('true_trk_direction: ',  true_track_direction)
    print('--------------------')

    print('--------------------')
    print('refl_prob:     ',  reflection_prob)
    print('ch_angle:      ',  cone_opening)
    print('photon_norm:   ',  photon_norm)
    print('att_L:         ',  att_L)
    print('trk_L:         ',  trk_L)
    print('scatt_L:       ',  scatt_L)
    print('trk_origin:    ',  track_origin)
    print('trk_direction: ',  track_direction)
    print('--------------------')


    key = random.PRNGKey(0)
    filename = 'test_events/optimization_start.h5'
    generate_and_store_event(filename, reflection_prob, cone_opening, track_origin, track_direction, \
        photon_norm, att_L, trk_L, scatt_L, detector, Nphot, key)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.r
    detector_height = detector.H
    
    # Optimization parameters
    num_iterations = 200
    
    best_params = None
    patience_counter = 0
    
    # Optimization loop
    for i in range(num_iterations):
        A = time.time()
        params = [reflection_prob, cone_opening, track_origin, track_direction, \
        float(photon_norm), float(att_L), float(trk_L), float(scatt_L)]

        # for some reason doing this inside of the loss does not work        
        true_hits = np.zeros(len(detector.all_points))
        true_hits[true_indices] = true_cts
        # ---------
        #print('sum true hits: ', np.sum(true_hits), photon_norm, Nphot)
        loss, (grad_refl_prob, grad_cone, grad_origin, grad_direction, grad_photon_norm, grad_att_L, grad_trk_L, grad_scatt_L) = loss_and_grad(
            true_indices, true_hits, true_times, *params, detector_points, detector_radius, detector_height, Nphot, key
        )
        B = time.time()

        track_direction = normalize(track_direction)

        # Print progress every 10 iterations
        if i % 10 == 0:
            generate_and_store_event('test_events/optimization_step'+str(i)+'.h5', reflection_prob, cone_opening, track_origin, track_direction, \
                photon_norm, att_L, trk_L, scatt_L, detector, Nphot, key)
            print("\n\n")
            print(f"Iteration {i}, Loss: {loss}")
            print(f"Refl probability: {reflection_prob}")
            print(f"Cone opening: {cone_opening}")
            print(f"Track origin: {track_origin}")
            print(f"Track direction: {track_direction}")
            if i > 0:
                print(f"Time in the last group of iterations: {time.time()-iteration_time} s.")
            iteration_time = time.time()
            print()

        print('grad num phot": ', grad_photon_norm)
        print('grad refl prob": ', grad_refl_prob)
        print(photon_norm, Nphot)

        Scale = 1
        cone_opening -= Scale*10*grad_cone
        track_origin -= Scale*0.05*grad_origin
        track_direction -= Scale*0.02*grad_direction
        reflection_prob -= Scale*1*grad_refl_prob

        # if i>50:
        photon_norm -= Scale*10.*grad_photon_norm


        att_L = 1
        trk_L = 1
        scatt_L = 1
        

        log.add_data(track_origin, track_direction, cone_opening, reflection_prob, att_L, trk_L, scatt_L, photon_norm, loss)
        #break

    print("\nOptimization complete.")
    print("Final parameters:")
    print(f"Cone opening: {cone_opening}")
    print(f"Track origin: {track_origin}")
    print(f"Track direction: {track_direction}")
    print("\nTrue parameters:")
    print(f"Cone opening: {true_cone_opening}")
    print(f"Track origin: {true_track_origin}")
    print(f"Track direction: {true_track_direction}")
    print(f"\nFinal Loss: {loss}")
    filename = 'test_events/optimization_result.h5'
    generate_and_store_event(filename, reflection_prob, cone_opening, track_origin, track_direction, \
        photon_norm, att_L, trk_L, scatt_L, detector, Nphot, key)
    log.plot_all()