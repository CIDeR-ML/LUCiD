### Basic instructions to run the Differentiable Simulator

The code is designed to generate MC samples in a cylindrical detector doing ray tracing.

The optimization code is not implemented yet. (in progress)

The main code to run the forward is in `CherenkovSimulator.py`. The main parameters to specify are:
- The detector configuration file
- The output filename for the simulated event data
- The number of photons to simulate
- The temperature parameter which controls the ray tracing relaxation
- The random seed / random parameters

### Dataset Visualization:
-> To visualize the dataset one can use `plot_dataset_in_2D.py`

Usage:
`python3 plot_dataset_in_2D.py --filename events/test_event_data.h5 --plot_time --output event_time.png`

By default, filename is `events/test_event_data.h5` and --plot_time to plot time or not

### Tests

- To run 1D gradient variation tests, run `tests/one_dimensional_grad_profiles.py`. You can run with random parameters
- To run runtime tests, run `tests/propagate_runtime.py` or `tests/simulation_loss_grad_runtime.py`. You can run them both in CPU mode (or GPU, make sure to download the GPU version of JAX)

### Code structure:
- The main simulation code is in `tools/simulation.py`
- The main propagation code is in `tools/propagate.py`
- The main detector geometry code is in `config/cyl_geom_config.js`

