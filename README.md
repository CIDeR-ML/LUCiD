### Basic instructions to run the toy MC

=== The code is designed to generate MC samples in a cylindrical detector doing ray tracing.

-> The main application is toyMC.py
There are 2 simulation modes: 
0: Isotropic Photons from a common origin.
1: Cherenkov Photons from a common origin.

Usage:
python toyMC.py --sim_mode N --json_filename "json_filename.json"
By default sim_mode has N=0 and json_filename="cyl_geom_config.json".

-> To visualize the dataset one can use 'plot_dataset_in_2D.py'
Usage:
python plot_dataset_in_2D.py --evt_ID N --filename "filename.h5" --json_filename "json_filename.json"
By default evt_ID has N=0, filename is "datasets/sim_mode_0_dataset_0_events.h5" and json_filename="cyl_geom_config.json".