# Documentation

Welcome to the LUCiD docs. For the project overview, see the
[LUCiD home page](../); this section is the technical documentation — install,
concepts, guides, and reference.

**Light-based Unified Calibration and trackIng Differentiable simulation** — the first end-to-end
*differentiable* optical particle-detector simulator. Gradients flow through emission,
propagation, scattering, and sensor response, so **calibration** and **track reconstruction**
become gradient-based optimization in one JAX framework, with a **SIREN** surrogate (a small
neural net trained on GEANT4 photon tables) supplying the light emission. Units are **meters,
nanoseconds, MeV**. Eighteen detector configurations ship in-tree, from Super-Kamiokande to
IceCube — see the [bundled detectors](getting-started/detectors.md).

Accompanies the paper *"End-to-end Differentiable Calibration and Reconstruction for Optical
Particle Detectors"* — [arXiv:2602.24129](https://arxiv.org/abs/2602.24129).

## What can I do with it?

| I want to… | Start here |
|------------|-----------|
| See something work in a few minutes | [Install](getting-started/install.md) → [quickstart](getting-started/quickstart.md) (`examples/hello_simulate.py`) |
| Understand the forward model | Concepts: [the photon pipeline](concepts/photon-pipeline.md), [architecture](concepts/architecture.md) |
| Reconstruct a track | [Reconstruction](guides/reconstruction.md); `examples/seed_reconstruct.py`; `analysis/paper/utils/run_study.py` for many events |
| Calibrate optical parameters | [Calibration](guides/calibration.md); `examples/hello_calibrate.py` |
| Model a specific detector | [Model your own detector](guides/model-your-own-detector.md); [configuration reference](reference/config.md) |
| Produce a dataset (GEANT4/PhotonSim) | [Producing datasets](guides/production/index.md); [dataset schema](reference/dataset-schema.md) |

## Cite

If you use LUCiD, please cite the paper above; [`CITATION.cff`](https://github.com/CIDeR-ML/LUCiD/blob/main/CITATION.cff)
has the reference.
