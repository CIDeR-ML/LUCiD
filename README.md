# LUCiD: Light-based Unified Calibration and trackIng Differentiable simulation

**The first end-to-end *differentiable* optical particle-detector simulator** — gradients flow
through emission, propagation, scattering, and sensor response, so calibration and track
reconstruction become gradient-based optimization in one framework.

Accompanies the paper *"End-to-end Differentiable Calibration and Reconstruction for Optical
Particle Detectors"* — [arXiv:2602.24129](https://arxiv.org/abs/2602.24129).

![Repository Overview](figures/combined_3x2_charge_displays.png)

LUCiD is a [JAX](https://github.com/google/jax)-based simulation of optical photon transport in
particle detectors — water-Cherenkov tanks, scintillator (WbLS), and neutrino telescopes. Units
are **meters, nanoseconds, MeV** throughout.

## Install & first event

The core install needs only JAX — **no GEANT4/PhotonSim** — to simulate, calibrate, and
reconstruct:

```bash
pip install -e .              # core; extras: [training] (torch), [docs], [all]
./scripts/download_data.sh    # example SIREN emitter + PhotonSim tables
python examples/hello_simulate.py   # simulate a 1 GeV muon, display its Cherenkov ring
```

Then open the tutorials in [`tutorials/`](tutorials/) (start with `00_quickstart.ipynb`) or read
the docs at **https://cider-ml.github.io/LUCiD/**.

## What you can do

- **Simulate** optical events in any supported geometry — one differentiable forward call.
- **Reconstruct** particle tracks (energy, vertex, direction, t₀) by gradient fit —
  `examples/hello_reconstruct.py`, or config-driven `lucid-optimize`.
- **Calibrate** optical parameters (scattering, absorption, reflection, QE, per-PMT QE) and
  read the achievable uncertainty from the Fisher / Cramér–Rao bound — `examples/hello_calibrate.py`.
- **Model diverse detectors**: cylinder (SK / HK / WCTE, algorithmic or from measured PMT
  `.npz`), sphere (JUNO), box, and neutrino-telescope strings (IceCube); water, WbLS, and ice.
- **Produce datasets** from GEANT4/PhotonSim (+ optional GENIE flux) → four-file HDF5 — `lucid-run-job`.

## Tutorials

Narrated notebooks in [`tutorials/`](tutorials/):

| Notebook | What it shows |
|----------|---------------|
| `00_quickstart` | build a simulator, shoot a muon, display the event |
| `track_optimization` | reconstruct a track (seed → Fisher-Gauss-Newton fit); + a JUNO-sphere example |
| `calibration_optimization` | fit optical parameters vs the Cramér–Rao bound |
| `calibration_gradients` | calibration loss landscape, the Hessian/Fisher, and the fit before/after |
| `track_gradients` | reconstruction loss landscapes and gradients |
| `data_vs_prediction` | per-PMT charge/time likelihood: data vs the differentiable model |
| `event_displays` | 2D / animation / 3D event views across geometries and materials |

Short, copy-paste **scripts** live in [`examples/`](examples/):
`hello_simulate.py`, `hello_reconstruct.py`, `hello_calibrate.py`, `seed_reconstruct.py`, `hello_telescope.py`.

## Package layout

- **`lucid/simulation/`** — `setup_event_simulator` (the JIT-compiled forward hub),
  `photon_step`, `sensor_response`.
- **`lucid/geometry/`** — registry-dispatched detectors (`cylinder`, `sphere`, `box`, `string`);
  `Cylinder.from_pmt_file(npz)` for measured SK/HK/WCTE arrays.
- **`lucid/propagation/`** — differentiable ray–geometry intersection and multi-bounce transport.
- **`lucid/wavelength/`** — wavelength-dependent optics (`medium`, `spectrum`, `optical_model`).
- **`lucid/siren/`** — SIREN surrogate for Cherenkov/dE-dx emission (trained on PhotonSim tables).
- **`lucid/sources/`** — track/cascade/calibration emitters and dataset event I/O.
- **`lucid/fitting/`** — the Gauss-Newton engine: reconstruction (`ReconModel`,
  `fit_track_multistart`) and calibration (`build_calibration_problem`, `fit`, `crb`).
- **`lucid/optimization/`** — hierarchical seed search + the `lucid-optimize` driver.
- **`lucid/gradient_analysis/`** — 1D/2D loss-landscape sweeps.
- **`lucid/production/`** — PhotonSim/GENIE → HDF5 dataset chain and cluster deployment.

Console entry points: `lucid-optimize`, `lucid-train-siren`, `lucid-run-job`,
`lucid-build-photon-table`, `lucid-build-dedx-table`.

## Data & production (PhotonSim / GENIE → HDF5)

Generating training data or datasets uses the external
[PhotonSim](https://github.com/cesarjesusvalls/PhotonSim) GEANT4 utility (only needed for this
chain, not for simulate/calibrate/reconstruct):

```bash
export PHOTONSIM_BIN=/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/configs/dataprod_01_mu.json \
              --n-events 1000 --job-id 0 --master-seed 42 --output-dir out/
```

See [docs/QUICKSTART_LOCAL.md](docs/QUICKSTART_LOCAL.md) (local), the cluster runbooks
(`docs/QUICKSTART_{S3DF,NERSC,LXPLUS}.md`, fronted by
[docs/CLUSTER_ABSTRACTION.md](docs/CLUSTER_ABSTRACTION.md)), and the dataset schema in
[docs/LUCID_DATASET.md](docs/LUCID_DATASET.md). The container
`ghcr.io/cider-ml/lucid:latest` ships PhotonSim + GENIE pre-built.

## Citation

If you use LUCiD, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

> O. Alterkait, C. Jesús-Valls, R. Matsumoto, P. de Perio, K. Terao,
> *End-to-end Differentiable Calibration and Reconstruction for Optical Particle Detectors*,
> arXiv:2602.24129 (2026).

## License

The license is being finalized and will be added shortly.
