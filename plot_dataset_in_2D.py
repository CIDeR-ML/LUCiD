import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.geometry import *
from tools.visualization import *


# Set default values
default_evt_ID = 0
default_filename = 'datasets/sim_mode_0_dataset_0_events.h5'
default_json_filename = 'cyl_geom_config.json'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--evt_ID', type=int, default=default_evt_ID, help='The event ID to plot')
parser.add_argument('--filename', type=str, default=default_filename, help='The input filename')
parser.add_argument('--json_filename', type=str, default=default_json_filename, help='The JSON filename')

args = parser.parse_args()

# Access the values
evt_ID = args.evt_ID
filename = args.filename
json_filename = args.json_filename

cyl_center, cyl_axis, cyl_radius, cyl_radius, cyl_height, cyl_barrel_grid, cyl_cap_rings, cyl_sensor_radius = load_cyl_geom(json_filename)
detector = generate_detector(json_filename)


# Extract info from h5 file
with h5py.File(filename, 'r') as f:

    # Access datasets
    h5_evt_ids = f['evt_id']
    h5_evt_pos = f['positions']
    h5_evt_hit_idx = f['event_hits_index']
    h5_evt_hit_IDs = f['hit_pmt']
    h5_evt_hit_Qs = f['hit_charge']
    h5_evt_hit_Ts = f['hit_time']

    # Access data
    evt_ids = h5_evt_ids[:]
    evt_pos = h5_evt_pos[:]
    evt_hit_idx = h5_evt_hit_idx[:]
    evt_hit_IDs = h5_evt_hit_IDs[:]
    evt_hit_Qs = h5_evt_hit_Qs[:]
    evt_hit_Ts = h5_evt_hit_Ts[:]
# -----------------------------


# Process info to make it compatible with event display format
IDs = None
if evt_ID == 0:
    IDs = evt_hit_IDs[0:evt_hit_idx[0]]
    Qs  = evt_hit_Qs[0:evt_hit_idx[0]]
    Ts  = evt_hit_Ts[0:evt_hit_idx[0]]
else:
    IDs = evt_hit_IDs[evt_hit_idx[evt_ID-1]:evt_hit_idx[evt_ID]]
    Qs  = evt_hit_Qs[evt_hit_idx[evt_ID-1]:evt_hit_idx[evt_ID]]
    Ts  = evt_hit_Ts[evt_hit_idx[evt_ID-1]:evt_hit_idx[evt_ID]]

print("Number of PMTs: ", len(detector.all_points))

ID_to_PE = np.zeros(len(detector.all_points))
ID_to_PE[IDs] = Qs
ID_to_position = {i:x for i,x in enumerate(detector.all_points)}
ID_to_case = detector.ID_to_case
ID_to_PE = {i:x for i,x in enumerate(ID_to_PE)}
# -----------------------------

# do the 2D plot.
show_2D_display(ID_to_position, ID_to_PE, ID_to_case, cyl_sensor_radius, cyl_radius, cyl_height)#, file_name='evt_example.pdf')







