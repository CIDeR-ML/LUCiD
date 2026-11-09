# Install

Simulation, reconstruction and calibration need only Python packages: JAX, flax, optax, numpy,
scipy, h5py, tqdm, matplotlib and uproot. GEANT4 and PhotonSim are needed only to
[produce new datasets](../guides/production/index.md).

Install from a clone, in editable mode: `config/` and `data/` are read relative to the
repository.

```bash
git clone https://github.com/CIDeR-ML/LUCiD.git && cd LUCiD
python3.10 -m venv .venv && . .venv/bin/activate
pip install -e . -c tests/constraints.txt
```

`tests/constraints.txt` pins the versions CI and the published numbers use (jax 0.5.3, Python
3.10); `pyproject.toml` allows `jax>=0.5.3,<0.6` and Python 3.10 or later.

JAX runs on the CPU unless a GPU build is installed:

```bash
pip install "jax[cuda12]==0.5.3" -c tests/constraints.txt
```

A visible GPU is then used automatically. `JAX_PLATFORMS=cpu` keeps a run on the CPU.

Optional extras, for example `pip install -e '.[viz]' -c tests/constraints.txt`:

| extra | adds | needed for |
|---|---|---|
| `[viz]` | plotly, kaleido, imageio | `event_display_3d`, `animate_event`, `scripts/visualize_detector.py`, the paper's display figures |
| `[docs]` | mkdocs-material | building these pages |
| `[dev]` | pytest, pre-commit, ruff | tests and lint hooks |
| `[all]` | all of the above | |

On SLAC's S3DF, the published numbers were produced in a specific Apptainer image; installing
into it is described under [the reference environment](../guides/production/clusters.md#the-reference-environment-developsif).

## Download the example data

The trained SIREN emitter weights, two small GEANT4 event files and the sensor-overlap tables
are a GitHub release asset (`lucid-data-v1`, 142 MB), not part of the git tree. Install it once
per checkout, from the repository root:

```bash
python scripts/download_data.py                  # download and install into data/
python scripts/download_data.py --from-dir DIR   # install DIR/lucid-data-v1.tar.gz (offline)
python scripts/download_data.py --verify-only    # check data/ against data/MANIFEST.sha256
```

Every file is checked against `data/MANIFEST.sha256` before anything is installed. The asset
contains:

| path | contents |
|---|---|
| `data/water/{muon,electron}/{cherenkov,dedx}/weights.npz` | the four SIREN emitters (their `emitter.json` is tracked in git) |
| `data/water/muon/1000MeV_5events.root` | five 1 GeV muon events from GEANT4 (PhotonSim) |
| `data/water/muon/1500MeV_6events.root` | six 1.5 GeV muon events |
| `spatial_overlap_integrals/` | 12 sensor-overlap tables for the shipped detectors and the paper's settings |

WbLS and ice use the water emitters. `--paper` installs the inputs of the paper figures instead
(see [Paper figures](../guides/paper.md)).

## Check the install

```bash
python examples/hello_simulate.py
```

This prints `expected charge: <total> pe over 10764 PMTs` and writes the event display to
`hello_simulate.png`. The total is the expected (mean) charge summed over the tank; it should be
finite and positive. It takes about a minute on 4 CPU cores, most of it compilation.

A `FileNotFoundError` naming a `weights.npz` under `data/` means the example data is not
installed.

A sensor radius or soft-overlap temperature with no shipped overlap table builds its table on
first use, which takes one to two minutes. The table is cached in `spatial_overlap_integrals/`,
or in `~/.cache/lucid/spatial_overlap_integrals/` when the checkout is not writable.

Next: the [quickstart](quickstart.md).
