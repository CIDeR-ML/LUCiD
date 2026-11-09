# Installation

Simulating, calibrating and reconstructing need **no GEANT4 or PhotonSim**: the core install is
JAX and a few Python packages (flax, optax, numpy, scipy, h5py, tqdm, matplotlib, uproot;
`pyproject.toml`). LUCiD is installed editable from a clone, because `config/` and `data/`
resolve relative to the repository.

```bash
git clone https://github.com/CIDeR-ML/LUCiD.git && cd LUCiD
python3.10 -m venv .venv && . .venv/bin/activate
pip install -e . -c tests/constraints.txt
```

`pyproject.toml` states the supported range; `tests/constraints.txt` pins the one point in it
that CI and the published numbers run at (jax 0.5.3 on Python 3.10). JAX runs on the CPU unless a
GPU build is installed: `pip install "jax[cuda12]==0.5.3" -c tests/constraints.txt`.

Optional extras, e.g. `pip install -e '.[viz]' -c tests/constraints.txt`:

| Extra | Adds | For |
|-------|------|-----|
| `[viz]` | plotly, kaleido, imageio | the 3D event display (`event_display_3d`), its image export, `animate_event`; the paper's display figures |
| `[docs]` | mkdocs-material | building this documentation |
| `[dev]` | pytest, pre-commit, ruff | the tests and the lint hooks |
| `[all]` | all three | |

The paper's numbers and the test gate were produced in one reference environment (Python
3.10.12, jax 0.5.3, an Apptainer image on SLAC's S3DF); how to install LUCiD into it is in the
[S3DF section](../guides/production/clusters.md#the-reference-environment-developsif) of the
cluster page. `ghcr.io/cider-ml/lucid` is the *production* image (GENIE + PhotonSim), not the
reference for published numbers.

## Get the example data

The trained SIREN emitters' weights, the example GEANT4 events and the sensor-overlap tables are
not in git; they are the `lucid-data-v1` GitHub release asset (142 MB, `tools/assets/lucid-data-v1.manifest`). Each checkout installs it once, from the
repository root:

```bash
python scripts/download_data.py                  # fetch it from the GitHub release
python scripts/download_data.py --from-dir DIR   # offline: install DIR/lucid-data-v1.tar.gz
python scripts/download_data.py --verify-only    # check data/ against data/MANIFEST.sha256
python scripts/download_data.py --paper          # the paper figures' inputs (lucid-paper-results-v1)
```

!!! note "Before the release is published"
    Until `lucid-data-v1` is on the GitHub release page, ask the maintainers for a copy of
    `lucid-data-v1.tar.gz` and install it with `--from-dir`.

Every file is checked against `data/MANIFEST.sha256` before any is moved into `data/`, so a
broken download installs nothing. It installs:

| file under `data/` | what |
|---|---|
| `water/{muon,electron}/{cherenkov,dedx}/weights.npz` | the four SIREN emitters, beside their tracked `emitter.json` |
| `water/muon/1000MeV_5events.root` | five 1 GeV muon events from GEANT4 (PhotonSim) |
| `water/muon/1500MeV_6events.root` | six 1.5 GeV muon events; event 5 is the paper's cylinder display (`analysis/paper/fig_cylinder_2d_displays.py`) |

and, at the repository root, `spatial_overlap_integrals/`: the 12 sensor-overlap tables of the
shipped detectors at the default temperature (0.2), of the paper's reconstruction (SK_like at 0.1)
and of the paper's display figures (0.05).

wbls and ice run water's SIREN emitters (their material JSON says `"siren": "water"`).

## Verify

```bash
python examples/hello_simulate.py
```

It prints `expected charge: <total> pe over 10764 PMTs` and writes the Cherenkov-ring display to
`hello_simulate.png`. The charge is the *expected* charge per PMT (the forward's mean, not a
sampled count), summed over the tank; a healthy run prints a finite, positive total. No value is
quoted here because the total moves with the forward model and, in the last digits, with the
machine and its thread count. A `FileNotFoundError` naming a `weights.npz` under `data/` means the
example data is not installed yet.

The run takes about a minute on 4 CPU cores, most of it building and compiling the simulator. The
PMT overlap table it needs came with the example data, in `spatial_overlap_integrals/` at the
repository root. A sensor radius or temperature with no shipped table costs one to two minutes on
the first run, which computes the table and caches it there (or in `~/.cache/lucid/` when the
checkout is not writable) for every later run.

Next: the [quickstart](quickstart.md).
