"""Wavelength-dependent physics for LUCiD.

Submodules:
    medium     — MediumProperties, make_medium, compute_effective_properties
    spectrum   — Cherenkov wavelength sampling
    scattering — Rayleigh and Mie/HG phase function samplers
"""

__all__ = [
    "MediumProperties",
    "make_medium",
    "compute_effective_properties",
    "load_qe_curve",
    "sample_cherenkov_wavelengths",
    "DEFAULT_WAVELENGTH_NM",
]

from lucid.wavelength.medium import MediumProperties, make_medium, compute_effective_properties, load_qe_curve
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths

# Default wavelength (nm) for padding and fallback — near Cherenkov peak, mid-grid.
DEFAULT_WAVELENGTH_NM = 400.0
