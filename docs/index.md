# LUCiD

**Light-based Unified Calibration and trackIng Differentiable simulation** — the first
end-to-end *differentiable* optical particle-detector simulator. Gradients flow through
emission, propagation, scattering, and sensor response, so **calibration** and **track
reconstruction** become gradient-based optimization in a single framework.

Accompanies the paper *"End-to-end Differentiable Calibration and Reconstruction for Optical
Particle Detectors"* — [arXiv:2602.24129](https://arxiv.org/abs/2602.24129).

## What is it?

LUCiD is a [JAX](https://github.com/google/jax)-based simulation of optical photon transport in
particle detectors — water-Cherenkov tanks, scintillator (WbLS), and neutrino telescopes. The
entire forward model (emission → ray tracing → scatter/reflect/absorb → per-PMT charge & time) is
JIT-compiled and differentiable, so you can take gradients of detector observables with respect
to particle properties *and* detector parameters. Units are **meters, nanoseconds, MeV**.

## What can I do with it?

| I want to… | Start here |
|------------|-----------|
| See something work in a minute | [Getting started](getting-started/install.md) → `examples/hello_simulate.py` |
| Simulate & display an event | Concepts: [the photon pipeline](concepts/photon-pipeline.md); tutorial `00_quickstart` |
| Reconstruct a track | [Reconstruction](RECONSTRUCTION.md); `examples/hello_reconstruct.py`; `lucid-optimize` |
| Calibrate optical parameters | [Calibration](CALIBRATION.md); `examples/hello_calibrate.py` |
| Model a specific detector | Concepts: [geometry & configuration](concepts/geometry.md) |
| Produce a dataset (GEANT4/PhotonSim) | [dataset schema](LUCID_DATASET.md); `lucid-run-job` |

## The framework at a glance

- **One differentiable forward hub** — `setup_event_simulator` returns a JIT-compiled callable
  mapping (particle/source, detector params) → per-PMT (charge, time).
- **Diverse geometries** — cylinder (SK/HK/WCTE), sphere (JUNO), box, and neutrino-telescope
  strings (IceCube); water, WbLS, ice.
- **SIREN emitter** — a physics-informed surrogate replaces GEANT4 for Cherenkov/dE-dx emission
  inside the differentiable loop.
- **Gradient-based calibration & reconstruction** — a shared Gauss-Newton engine (`lucid.fitting`)
  with Fisher / Cramér-Rao uncertainties.

## Cite

If you use LUCiD, please cite the paper (see [`CITATION.cff`](https://github.com/CIDeR-ML/LUCiD/blob/main/CITATION.cff)):
Alterkait, Jesús-Valls, Matsumoto, de Perio, Terao, *arXiv:2602.24129* (2026).
