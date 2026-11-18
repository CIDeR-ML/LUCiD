# Scripts

Command-line tools that are not console entry points (those are listed in the
[CLI reference](../docs/reference/cli.md)). Run them from the repository root, since they read
`config/` and `data/`. Each takes `--help`.

| script | what it does |
|--------|--------------|
| `download_data.py` | installs the example data (`lucid-data-v1`: SIREN weights, Geant4 events, PMT overlap tables), checking every file's sha256. `--paper` installs the paper figures' inputs instead; `--from-dir` installs from a local copy; `--verify-only` checks what is installed |
| `benchmark_forward.py` | warm wall time of the track fit's forward, loss and gradient and of the calibration forward and Jacobian, over photon counts `--n` and scatter iterations `--k`; `--what track\|calibration\|both` |
| `check_gradients.py` | compares the autodiff gradient of the calibration loss with central finite differences, per optical parameter |
| `find_optimal_k.py` | sweeps the scatter-iteration count K for a `--detector` and reports where the per-sensor charge converges |
| `geofile_to_npz.py` | converts a WCSim-style detector geofile (`.txt`) to a PMT-array `.npz` ([schema](../docs/reference/pmt-npz-schema.md)); single-type cylindrical detectors only |
| `inspect_dataset.py` | prints a text summary of a dataset batch (`sensor`, `hits`, `step`, `labl`) |
| `visualize_detector.py` | draws the sensors of any `*_geom_config.json` in 3D, and unrolled in 2D for a cylinder; needs the `[viz]` extra |

`python scripts/X.py` imports the installed `lucid`, not necessarily the checkout the script is
in. To run the scripts of a second checkout, use `PYTHONPATH=<that checkout> python scripts/X.py`.
