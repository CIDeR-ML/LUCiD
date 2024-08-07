
import h5py
import numpy as np
import matplotlib.pyplot as plt


import jax.numpy as jnp
from itertools import product
from tqdm import tqdm
from jax import random
import jax

from tools.geometry import *
from tools.losses import *


def loss_search_in_grid(detector, true_event_filename='autodiff_datasets/data_events.h5'):

    loss_and_grad = jax.value_and_grad(smooth_combined_loss_function, argnums=(3, 4, 5, 6, 7, 8, 9, 10))

    true_indices, true_cts, true_times, true_reflection_prob, true_cone_opening, true_track_origin, true_track_direction,  \
    true_photon_norm, true_att_L, true_trk_L, true_scatt_L = load_data(true_event_filename)

    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.r
    detector_height = detector.H

    key = random.PRNGKey(0)

    # Define grid ranges
    cone_opening_range = np.linspace(36., 44., 1)
    track_origin_range = np.linspace(-1., 1., 3)
    track_direction_range = np.linspace(-1., 1., 3)

    reflection_prob = 0.3
    photon_norm = 1.
    att_L = 10.    # [meters]
    trk_L = 1.     # [meters]
    scatt_L = 10.  # [meters]

    # Create grid
    grid = list(product(cone_opening_range, 
                        track_origin_range, track_origin_range, track_origin_range,
                        track_direction_range, track_direction_range, track_direction_range))

    # Calculate loss for each grid point
    losses = []
    for params in tqdm(grid, desc="Calculating grid losses"):
        cone_opening = params[0]
        track_origin = np.array(params[1:4])
        track_direction = normalize(np.array(params[4:]))
        
        Nphot = 50
        true_hits = np.zeros(len(detector.all_points))
        true_hits[true_indices] = true_cts
        loss, (grad_refl_prob, grad_cone, grad_origin, grad_direction, grad_photon_norm, grad_att_L, grad_trk_L, grad_scatt_L) = loss_and_grad(
            true_indices, true_hits, true_times, reflection_prob, cone_opening, track_origin, track_direction, float(photon_norm), att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key
        )
        if loss>0:
            losses.append(loss)
        else:
            losses.append(1000000)

    # Convert to numpy array for easier analysis
    losses = np.array(losses)

    return np.array(grid)[np.argsort(losses)][0]



def load_data(filename):
    with h5py.File(filename, 'r') as f:
        hit_pmt = np.array(f['hit_pmt'])
        hit_charge = np.array(f['hit_charge'])
        hit_time = np.array(f['hit_time'])

        reflection_prob = np.array(f['reflection_prob'])[0]
        cone_opening = np.array(f['cone_opening'])[0]
        track_origin = np.array(f['track_origin'])
        track_direction = np.array(f['track_direction'])

        photon_norm = np.array(f['photon_norm'])
        att_L = np.array(f['att_L'])
        trk_L = np.array(f['trk_L'])
        scatt_L = np.array(f['scatt_L'])

    return hit_pmt, hit_charge, hit_time, reflection_prob, cone_opening, track_origin, track_direction, photon_norm, att_L, trk_L, scatt_L

def relative_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)

    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

class Logger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.origins = []
        self.directions = []
        self.losses = []
        self.ch_angles = []
        self.ref_probs = []
        self.photon_norm = []
        self.att_Ls = []
        self.trk_Ls = []
        self.scatt_Ls = []
        
        self.true_dir = None
        self.true_ori = None
        self.true_ch_angle = None
        self.true_ref_prob = None
        self.true_photon_norm = None
        self.true_att_L = None
        self.true_trk_L = None
        self.true_scatt_L = None

    #def add_data(self, origin, direction, true_dir, true_ori, ch_angle, ref_prob, photon_norm, att_L, trk_L, scatt_L, loss):
    def add_data(self, origin, direction, ch_angle, ref_prob, att_L, trk_L, scatt_L, photon_norm, loss):
        self.origins.append(origin)
        self.directions.append(direction)
        self.losses.append(float(loss))
        self.ch_angles.append(ch_angle)
        self.ref_probs.append(ref_prob)
        self.photon_norm.append(photon_norm)
        self.att_Ls.append(att_L)
        self.trk_Ls.append(trk_L)
        self.scatt_Ls.append(scatt_L)

    def plot_angle_err(self):
        plt.plot(range(len(self.directions)), np.linalg.norm(np.array(self.directions)-np.array(self.true_dir), axis=1), label='Direction angle error', color='darkorange')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Angle Error (degrees)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0.)
        plt.xlim(0, len(self.directions))  # Add this line

    def plot_distance_err(self):
        plt.plot(range(len(self.origins)), np.linalg.norm(np.array(self.origins)-np.array(self.true_ori), axis=1), label='Origin distance error', color='cornflowerblue')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Distance Error (meters)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0.)
        plt.xlim(0, len(self.origins))  # Add this line

    def plot_ch_angle(self):
        plt.plot(range(len(self.ch_angles)), self.ch_angles, label='Reco', color='hotpink')
        plt.axhline(self.true_ch_angle, color='darkgray', linestyle='--', label='True')
        plt.xlim(0, len(self.ch_angles))  # Modify this line
        plt.ylim(bottom=min(self.true_ch_angle, min(self.ch_angles)) / 1.43, top=max(self.true_ch_angle, max(self.ch_angles)) * 1.3)
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Cone Opening')
        plt.legend(frameon=False, loc='best')

    def plot_ref_prob(self):
        plt.plot(range(len(self.ref_probs)), self.ref_probs, label='Reco', color='limegreen')
        if self.true_ref_prob is not None:
            plt.axhline(self.true_ref_prob, color='darkgreen', linestyle='--', label='True')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Reflection Probability')
        plt.legend(frameon=False, loc='best')
        plt.ylim(0, 1)
        plt.xlim(0, len(self.ref_probs))  # Add this line

    def plot_photon_norm(self):
        plt.plot(range(len(self.photon_norm)), self.photon_norm, label='Reco', color='purple')
        if self.true_photon_norm is not None:
            plt.axhline(self.true_photon_norm, color='darkviolet', linestyle='--', label='True')
        plt.ylim(bottom=min(self.true_photon_norm, min(self.photon_norm)) / 1.43, top=max(self.true_photon_norm, max(self.photon_norm)) * 2)
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Number of Photons')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0)
        plt.xlim(0, len(self.photon_norm))  # Add this line

    def plot_att_L(self):
        plt.plot(range(len(self.att_Ls)), self.att_Ls, label='Reco', color='brown')
        if self.true_att_L is not None:
            plt.axhline(self.true_att_L, color='darkred', linestyle='--', label='True')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Attenuation Length (m)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0)
        plt.xlim(0, len(self.att_Ls))  # Add this line

    def plot_trk_L(self):
        plt.plot(range(len(self.trk_Ls)), self.trk_Ls, label='Reco', color='teal')
        if self.true_trk_L is not None:
            plt.axhline(self.true_trk_L, color='darkcyan', linestyle='--', label='True')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Track Length (m)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0)
        plt.xlim(0, len(self.trk_Ls))  # Add this line

    def plot_scatt_L(self):
        plt.plot(range(len(self.scatt_Ls)), self.scatt_Ls, label='Reco', color='olive')
        if self.true_scatt_L is not None:
            plt.axhline(self.true_scatt_L, color='darkolivegreen', linestyle='--', label='True')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Scattering Length (m)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0)
        plt.xlim(0, len(self.scatt_Ls))  # Add this line

    def plot_loss(self):
        plt.plot(range(len(self.losses)), self.losses, color='k')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Loss')
        plt.xlim(0, len(self.losses))  # This line is already correct
        plt.yscale('log')

    def plot_all(self):
        plt.figure(figsize=(12, 9))
        plt.subplot(3, 3, 1)
        self.plot_angle_err()
        plt.subplot(3, 3, 2)
        self.plot_distance_err()
        plt.subplot(3, 3, 3)
        self.plot_ch_angle()
        plt.subplot(3, 3, 4)
        self.plot_ref_prob()
        plt.subplot(3, 3, 5)
        self.plot_photon_norm()
        plt.subplot(3, 3, 6)
        self.plot_att_L()
        plt.subplot(3, 3, 7)
        self.plot_trk_L()
        plt.subplot(3, 3, 8)
        self.plot_scatt_L()
        plt.subplot(3, 3, 9)
        self.plot_loss()
        
        plt.tight_layout()
        plt.savefig('output_plots/optimization_results.png')
        plt.show()
