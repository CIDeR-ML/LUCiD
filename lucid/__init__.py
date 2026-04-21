"""LUCiD — Light-based Unified Calibration and trackIng Differentiable simulation.

Submodules are imported on demand:

    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams, ParticleParams
    from lucid.geometry import generate_detector

No eager imports live here so that lightweight tools (e.g.
`lucid.production.run_job`'s `--skip-lucid` path on a bare Python
environment) can import submodules without pulling in JAX / numpy /
h5py.
"""
