
from tools.losses import *
from tools.utils import *

def optimize_params(detector, true_indices, true_times, true_cone_opening, true_track_origin, true_track_direction, cone_opening, track_origin, track_direction, Nphot):


        log = Logger()
        key = random.PRNGKey(0)

        filename = 'test_events/optimization_start.h5'
        generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key)

        detector_points = jnp.array(detector.all_points)
        detector_radius = detector.S_radius

        loss_and_grad = jax.value_and_grad(combined_loss_function, argnums=(2, 3, 4))

        # Optimization parameters
        num_iterations = 100
        
        best_params = None
        patience_counter = 0
        
        # Optimization loop
        for i in range(num_iterations):
            A= time.time()
            loss, (grad_cone, grad_origin, grad_direction) = loss_and_grad(
                true_indices, true_times, cone_opening, track_origin, track_direction, 
                detector_points, detector_radius, Nphot, key,  False
            )
            print('this iteration time: ', time.time()-A, ' seconds.')

            Scale = 1
            cone_opening -= Scale*10*grad_cone
            track_origin -= Scale*0.05*grad_origin
            track_direction -= Scale*0.02*grad_direction

            log.add_data(track_origin, track_direction, true_track_direction, true_track_origin, cone_opening, true_cone_opening, loss)

            # Print progress every 10 iterations
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss}")
                print(f"Cone opening: {cone_opening}")
                print(f"Track origin: {track_origin}")
                print(f"Track direction: {track_direction}")
                if i>0:
                    print(f"Time in the last group of iterations: {time.time()-iteration_time} s.")
                iteration_time = time.time()
                print()

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
        generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key)

        log.plot_all()