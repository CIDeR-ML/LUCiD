# Installation

The core install needs only JAX — **no GEANT4/PhotonSim** — to simulate, calibrate, and
reconstruct.

```bash
git clone https://github.com/CIDeR-ML/LUCiD.git
cd LUCiD
pip install -e .              # core (JAX, etc.)
```

Optional extras:

| Extra | Adds | For |
|-------|------|-----|
| `[training]` | `torch` | training the SIREN emitter |
| `[docs]` | mkdocs-material, mkdocs-jupyter, mkdocstrings | building this documentation |
| `[dev]` | pytest, jupyter, nbmake | development & tests |
| `[all]` | everything | |

e.g. `pip install -e .[dev]`.

## Get the example data

The example SIREN emitter weights and PhotonSim tables are **not** in git — fetch them once:

```bash
./scripts/download_data.sh              # muon + electron, 1000 MeV (~2 GB)
# ./scripts/download_data.sh --all-energies    # 500/1000/1500 MeV
```

This also wires up `data/wbls` and `data/ice` as symlinks reusing the water files, so those
materials work out of the box.

## Verify

```bash
python examples/hello_simulate.py
```

You should see a muon simulated in an SK-like tank and its per-PMT Cherenkov ring displayed.
If it fails with a `FileNotFoundError` pointing at a SIREN model path under `data/`, run
`./scripts/download_data.sh` first.

Next: the [quickstart](quickstart.md).
