"""Smoke-test: import every module in lucid/ without error."""
import importlib
import pytest


MODULES = [
    "lucid",
    "lucid.simulation",
    "lucid.simulation.optics",
    "lucid.simulation.photon_step",
    "lucid.simulation.sensor_response",
    "lucid.simulation.simulator",
    "lucid.simulation.config",
    "lucid.simulation.types",
    "lucid.geometry.detector_geometry",
    "lucid.sources.particle_model",
    "lucid.generate",
    "lucid.sources",
    "lucid.sources.siren_rays",
    "lucid.sources.calibration_sources",
    "lucid.sources.event_io",
    "lucid.losses",
    "lucid.detector_params",
    "lucid.utils",
    "lucid.overlap",
    "lucid.visualization",
    "lucid.geometry",
    "lucid.geometry.base",
    "lucid.geometry.detector",
    "lucid.geometry.cylinder",
    "lucid.geometry.sphere",
    "lucid.geometry.box",
    "lucid.propagation",
    "lucid.propagation.base",
    "lucid.propagation.cylinder",
    "lucid.propagation.sphere",
    "lucid.propagation.box",
    "lucid.siren",
    "lucid.siren.core",
    "lucid.optimization",
    "lucid.optimization.grid_search",
    "lucid.optimization.pipeline",
    "lucid.optimization.run",
    "lucid.wavelength",
    "lucid.wavelength.medium",
    "lucid.wavelength.spectrum",
    "lucid.wavelength.scattering",
    "lucid.gradient_analysis",
    "lucid.gradient_analysis.sweep",
    "lucid.gradient_analysis.plotting",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
