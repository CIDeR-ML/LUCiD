"""LUCiD — Light-based Unified Calibration and trackIng Differentiable simulation.

Submodules are imported on demand:

    from lucid.simulation import setup_event_simulator
    from lucid.detector_params import DetectorParams, ParticleParams, JointParams
    from lucid.geometry import generate_detector

No eager imports live here so that lightweight tools (e.g.
``lucid.production.run_job``'s ``--skip-lucid`` path on a bare Python
environment) can import submodules without pulling in JAX / numpy / h5py.
The convenience top-level names below still resolve — lazily, via PEP 562
``__getattr__`` — on first access, so ``lucid.DetectorParams`` and
``from lucid import DetectorParams`` both work without eagerly importing JAX
at ``import lucid`` time.
"""

__all__ = [
    'setup_event_simulator',
    'DetectorParams',
    'ParticleParams',
    'JointParams',
    'generate_detector',
]

# name -> submodule it lives in (resolved lazily on first attribute access)
_LAZY = {
    'setup_event_simulator': 'lucid.simulation',
    'DetectorParams': 'lucid.detector_params',
    'ParticleParams': 'lucid.detector_params',
    'JointParams': 'lucid.detector_params',
    'generate_detector': 'lucid.geometry',
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'lucid' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(__all__)
