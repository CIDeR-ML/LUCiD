
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    with h5py.File(filename, 'r') as f:
        hit_pmt = np.array(f['hit_pmt'])
        hit_charge = np.array(f['hit_charge'])
        hit_time = np.array(f['hit_time'])
        true_cone_opening = np.array(f['true_cone_opening'])[0]
        true_track_origin = np.array(f['true_track_origin'])
        true_track_direction = np.array(f['true_track_direction'])

    return hit_pmt, hit_charge, hit_time, true_cone_opening, true_track_origin, true_track_direction

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
        self.origins         = []
        self.directions      = []
        self.losses          = []
        self.dir_err         = []
        self.ori_err         = []
        self.ch_angles       = []
        self.ch_angles_err   = []

    def add_data(self, origin, direction, true_dir, true_ori, ch_angle, true_ch_angle, loss):
        self.origins.append(origin)
        self.directions.append(direction)
        self.losses.append(float(loss))
        self.dir_err.append(relative_angle(true_dir, direction))
        self.ori_err.append(np.linalg.norm(true_ori-origin))
        self.ch_angles.append(ch_angle)
        self.ch_angles_err.append(true_ch_angle-ch_angle)

    def plot_angle_err(self):
        plt.plot(range(len(self.dir_err[:])), self.dir_err[:], label='Direction angle error', color='darkorange')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Angle Error (degrees)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0.)

    def plot_distance_err(self):
        plt.plot(range(len(self.ori_err[:])), self.ori_err[:], label='Origin distance error', color='cornflowerblue')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Distance Error (meters)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0.)

    def plot_ch_angle(self):
        self.expected_cone_opening = 40
        plt.plot(range(len(self.ch_angles[:])), self.ch_angles[:], color='hotpink')
        plt.axhline(self.expected_cone_opening, color='darkgray', linestyle='--', label='expected')
        plt.xlim(1, len(self.losses))
        plt.ylim(bottom=min(self.expected_cone_opening, min(self.ch_angles[:])) / 1.43, top=max(self.expected_cone_opening, max(self.ch_angles[:])) * 1.3)
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Cone Opening')
        plt.legend(frameon=False, loc='best')

    def plot_loss(self):
        plt.plot(range(len(self.losses[:])), self.losses[:], color='k')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Loss')
        plt.xlim(0, len(self.losses))
        plt.yscale('log')

    def plot_all(self):
        plt.figure(figsize=(12, 9))

        plt.subplot(2, 2, 1)
        self.plot_angle_err()

        plt.subplot(2, 2, 2)
        self.plot_distance_err()

        plt.subplot(2, 2, 3)
        self.plot_ch_angle()

        plt.subplot(2, 2, 4)
        self.plot_loss()

        plt.tight_layout()
        plt.savefig('output_plots/optimization_results.pdf')
        plt.show()

