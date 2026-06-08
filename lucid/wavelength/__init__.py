"""Wavelength-dependent physics for LUCiD.

Submodules:
    medium        — MediumProperties, make_medium, compute_effective_properties
    optical_model — evaluate_optical_model: per-photon λ → optical-property seam
    spectrum      — Cherenkov wavelength sampling
    scattering    — Rayleigh and Mie/HG phase function samplers
"""
from lucid.wavelength.medium import MediumProperties, make_medium, compute_effective_properties, load_qe_curve
from lucid.wavelength.optical_model import evaluate_optical_model, OpticalArrays
from lucid.wavelength.spectrum import (
    sample_cherenkov_wavelengths, Monochromatic, PowerLaw, QEWeighted,
)

# Default wavelength (nm) for padding and fallback — near Cherenkov peak, mid-grid.
DEFAULT_WAVELENGTH_NM = 400.0
