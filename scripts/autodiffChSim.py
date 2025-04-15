import time
import pandas as pd
import h5py
import json
import numpy as np
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.geometry import *
from tools.utils import *
from tools.generate import *
from tools.losses import *
from tools.optimization import *
from tests.one_dimensional_grad_profiles import *

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from jax.lax import scan, while_loop

from functools import partial

import warnings

warnings.filterwarnings("ignore", message="unhashable type: .*. Attempting to hash a tracer will lead to an error in a future JAX release.")

directory = 'output_plots'
if not os.path.exists(directory):
    os.makedirs(directory)

Nphot = 1000

def main():
    # Set default values
    default_json_filename = 'config/cyl_geom_config.json'
    output_filename = 'autodiff_datasets/data_events.h5'
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--is_data', type=bool, default=False, help='This creates the data event.')
    parser.add_argument('--json_filename', type=str, default=default_json_filename, help='The JSON filename')
    parser.add_argument('--test', action='store_true', help='Run tests instead of optimization')
    parser.add_argument('--use_time_loss', action='store_true', help='Use time-based loss function')

    args = parser.parse_args()
    
    json_filename = args.json_filename
    use_time_loss = args.use_time_loss

    if args.is_data:
        print('Using data mode')

        true_reflection_prob = 0.3
        true_cone_opening = np.random.uniform(37., 43)
        true_track_origin = np.random.uniform(-1., 1., size=3)
        true_track_direction = normalize(np.random.uniform(-1., 1., size=3))

        # true_reflection_prob = 0.
        # true_cone_opening = 40.
        # true_track_origin = np.array([1., -0.2, 0.3])
        # true_track_direction = normalize(np.array([0.4, 0.7, 1.]))

        true_photon_norm = 1.
        true_att_L = 10    # [meters]
        true_trk_L = 1     # [meters]
        true_scatt_L = 10  # [meters]

        true_params = [true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction, \
        true_photon_norm, true_att_L, true_trk_L, true_scatt_L]

        detector = generate_detector(json_filename)
        key = random.PRNGKey(0)
        generate_and_store_event(output_filename, *true_params, detector, Nphot, key)

    elif args.test:

        print('Running tests')

        true_reflection_prob = 0.33
        true_cone_opening = 40.
        true_track_origin = np.array([1., 0., 0.])
        true_track_direction = np.array([1., 0., 0.])

        true_photon_norm = 1.
        true_att_L = 10.    # [meters]
        true_trk_L = 1.     # [meters]
        true_scatt_L = 10.  # [meters]

        true_params = [true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction, \
        float(true_photon_norm), true_att_L, true_trk_L, true_scatt_L]

        detector = generate_detector(json_filename)
        key = random.PRNGKey(0)
        generate_and_store_event(output_filename, *true_params, detector, Nphot, key)

        true_indices, true_cts, true_times, true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction,  \
        true_photon_norm, true_att_L, true_trk_L, true_scatt_L = load_data(output_filename)
        
        detector_points = jnp.array(detector.all_points)
        detector_radius = detector.r
        detector_height = detector.H
        
        true_params = [true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction, \
        true_photon_norm, true_att_L, true_trk_L, true_scatt_L]

        one_dimensional_grad_profiles(
            detector, true_indices, true_cts, true_times, detector_points, detector_radius, detector_height, Nphot, true_params
        )

    else:
        print('Inference mode')

        detector = generate_detector(json_filename)
        true_indices, true_cts, true_times, true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction,  \
        true_photon_norm, true_att_L, true_trk_L, true_scatt_L = load_data(output_filename)
        
        true_params = [true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction, \
        true_photon_norm, true_att_L, true_trk_L, true_scatt_L]

        reflection_prob = 0.2
        cone_opening = 40.
        track_origin = np.array([1., 0., 0.])
        track_direction = np.array([1., 0., 0.])

        photon_norm = 1.
        att_L = 10.    # [meters]
        trk_L = 1.     # [meters]
        scatt_L = 10.  # [meters]

        use_new_method = True
        if use_new_method:
            grid_result  = loss_search_in_grid(detector, output_filename)
            cone_opening = grid_result[0]
            track_origin = np.array([grid_result[1], grid_result[2], grid_result[3]])
            track_direction = np.array([grid_result[4], grid_result[5], grid_result[6]])

            #let's not guess all the parameters (no calibration nor normalization --maybe add num photons in the future--)
            print('--------------------')
            print('guessed_cone_opening: ',    cone_opening)
            print('guessed_track_origin: ',    track_origin)
            print('guessed_track_direction: ', track_direction)
            print('--------------------')

        params = [reflection_prob, true_cone_opening, track_origin, track_direction, \
        true_photon_norm*1.2, true_att_L, true_trk_L, true_scatt_L]

        optimize_params(detector, true_cts, true_indices, true_times, *true_params, *params, Nphot)



if __name__ == "__main__":
    stime = time.perf_counter()
    main()
    print('Total exec. time: ', f"{time.perf_counter()-stime:.2f} s.")
