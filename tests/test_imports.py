"""Smoke-test: import every module in lucid/, and check the public surface is usable."""
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
    "lucid.geometry.registry",
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


def test_every_public_name_resolves_to_what_it_claims_to_be():
    """A name in `__all__` must be the OBJECT it advertises, not merely present.

    A submodule with the same name as an exported function (e.g. `gauss_newton`) can shadow it, so
    `from lucid.fitting import gauss_newton` returns the module. The shadowing is silent and
    import-order dependent, and tests that import from the submodule path do not catch it.
    """
    import types
    import lucid.fitting as F

    for name in F.__all__:
        assert hasattr(F, name), f'{name} is in __all__ but not on the package'

    shadowed = [n for n in F.__all__
                if isinstance(getattr(F, n), types.ModuleType) and n != 'report']
    assert not shadowed, (
        f'{shadowed} resolve to MODULES, not to the objects __all__ advertises — a submodule of '
        f'the same name has shadowed them')

    # `damped_matrix` is deliberately NOT here: it is reached as
    # `lucid.fitting.transforms.damped_matrix` and is not re-exported from the package.
    for name in ('gauss_newton', 'calibrate', 'closure', 'closure_data', 'fit',
                 'fit_track', 'crb', 'profile_gains', 'neyman_residual'):
        assert callable(getattr(F, name)), f'lucid.fitting.{name} is not callable'
