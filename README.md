# LUCiD: Light-based Unified Calibration and trackIng Differentiable simulation

**The first end-to-end *differentiable* optical particle-detector simulator** — gradients flow
through emission, propagation, scattering, and sensor response, so calibration and track
reconstruction become gradient-based optimization in one framework.

Accompanies the paper *"End-to-end Differentiable Calibration and Reconstruction for Optical
Particle Detectors"* — [arXiv:2602.24129](https://arxiv.org/abs/2602.24129).

![Per-PMT charge displays for six detector geometries](docs/assets/combined_3x2_charge_displays.png)

LUCiD is a [JAX](https://github.com/google/jax)-based simulation of optical photon transport in
particle detectors — water-Cherenkov tanks, water-based liquid scintillator (WbLS), and neutrino
telescopes. Inside the loop, a **SIREN** (a small sinusoidal neural network trained on
GEANT4/PhotonSim photon tables) stands in for GEANT4 as the differentiable light emitter. Units
are **meters, nanoseconds, MeV** throughout.

## Get started

Simulating, calibrating and reconstructing need **no GEANT4 or PhotonSim**, only JAX and a few
Python packages. Python 3.10; jax 0.5.3 is the tested point.

```bash
git clone https://github.com/CIDeR-ML/LUCiD.git && cd LUCiD
python3.10 -m venv .venv && . .venv/bin/activate
pip install -e . -c tests/constraints.txt   # CPU; for a GPU also install "jax[cuda12]==0.5.3"
python scripts/download_data.py             # SIREN weights + example GEANT4 events
python examples/hello_simulate.py           # one 1 GeV muon -> hello_simulate.png
```

[Installation](docs/getting-started/install.md) has the extras, the reference environment
and what the first run prints; the [quickstart](docs/getting-started/quickstart.md) walks through
it. The documentation is at **https://cider-ml.github.io/LUCiD/**.

## What you can do

- **Simulate** optical events in any supported geometry — one differentiable forward call
  (`examples/hello_simulate.py`; `examples/hello_gradient.py` differentiates it).
- **Reconstruct** particle tracks (energy, vertex, direction, t₀) by gradient fit —
  `examples/seed_reconstruct.py` runs the published method; many events with the paper's
  `analysis/paper/utils/run_study.py`.
- **Calibrate** optical parameters (scattering, absorption, reflection, QE, per-PMT QE) and
  read the achievable uncertainty from the Fisher / Cramér–Rao bound — `examples/hello_calibrate.py`.
- **Model diverse detectors**: cylinder (SK / HK / WCTE, algorithmic or from measured PMT
  `.npz`), sphere (JUNO), box, and neutrino-telescope strings (IceCube); water, WbLS, and ice
  ([bundled detectors](docs/getting-started/detectors.md)).
- **Produce datasets** from GEANT4/PhotonSim (+ optional GENIE flux) → four parallel HDF5 files —
  `lucid-run-job` ([producing datasets](docs/guides/production/index.md); the container
  `ghcr.io/cider-ml/lucid` ships PhotonSim + GENIE pre-built).

[`examples/`](examples/README.md) holds the runnable scripts, [`analysis/paper/`](analysis/paper/README.md)
reproduces the paper's figures, and [architecture](docs/concepts/architecture.md) maps the package.

## Citation

If you use LUCiD, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

> O. Alterkait, C. Jesús-Valls, R. Matsumoto, P. de Perio, K. Terao,
> *End-to-end Differentiable Calibration and Reconstruction for Optical Particle Detectors*,
> arXiv:2602.24129 (2026).

## License

All rights reserved: no licence to use, copy, modify or distribute LUCiD is granted yet.
[`LICENSE`](LICENSE) states the terms and why; an open licence is being pursued through SLAC's
Office of Technology Licensing. Third-party material keeps its own terms
([`THIRD_PARTY.md`](THIRD_PARTY.md)).
