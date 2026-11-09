# LUCiD

LUCiD (Light-based Unified Calibration and trackIng Differentiable simulation) is a
differentiable photon simulator for optical particle detectors: water Cherenkov tanks,
water-based liquid scintillator (WbLS) and ice. It is written in JAX, so the gradient of a loss
on the sensor output flows back through emission, propagation, scattering and sensor response.
Calibration and track reconstruction are then gradient-based fits on the same forward model.
Light emission comes from SIREN emitters, small neural networks trained on GEANT4 photon tables.

Units are metres, nanoseconds and MeV throughout.

The method and its results are in *End-to-end Differentiable Calibration and Reconstruction for
Optical Particle Detectors*, [arXiv:2602.24129](https://arxiv.org/abs/2602.24129). The project
has a [home page](../).

## Where to start

| I want to | Read |
|---|---|
| install it and see an event | [Install](getting-started/install.md), then the [quickstart](getting-started/quickstart.md) |
| run it on my detector | [Model your own detector](guides/model-your-own-detector.md) and the [bundled detectors](getting-started/detectors.md) |
| reconstruct a track | [Reconstruction](guides/reconstruction.md) |
| calibrate optical parameters | [Calibration](guides/calibration.md) |
| redraw the paper's figures | [Paper figures](guides/paper.md) |
| produce a labelled GEANT4 dataset | [Producing datasets](guides/production/index.md) |
| look up an argument, config key or output field | [Reference](reference/simulator.md) |

## Cite

If you use LUCiD, cite the paper above.
[`CITATION.cff`](https://github.com/CIDeR-ML/LUCiD/blob/main/CITATION.cff) has the reference.
