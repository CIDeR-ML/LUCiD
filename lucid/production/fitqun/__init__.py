"""Generators for the inputs fiTQun's tuning chain consumes.

fiTQun reconstructs a water-Cherenkov event by comparing it against an
analytic prediction of the charge and time at every PMT. That prediction is
driven by five tuned tables, and this package produces the LUCiD-derived
inputs to each of them:

    cprofile   Cherenkov emission profile and its integral tables (from
               PhotonSim; particle + water only, no detector)
    chargepdf  photosensor charge response f(q | mu) (from LUCiD's digitizer)
    angular    photosensor angular response epsilon(cos eta)
    timepdf    direct-light time residual vs predicted charge
    scattable  indirect (scattered and reflected) light tables

The boundary is deliberate: each module writes the file the corresponding
stage of ``fiTQun/Utilities`` already reads, so the fitting code that turns
histograms into fiTQun's parametrised ``const/`` files stays where it is and
keeps being the thing the collaboration has validated. :mod:`rootio` is the
seam that lets a non-ROOT codebase write those files.

Nothing here is detector-specific by construction — the profile and charge
response are not, and the three that are take their geometry from the usual
LUCiD detector/physics config pair, so the same code serves SK and HK.
"""
# Submodules are imported lazily: the cluster fan-out runs on the submit host,
# where uproot and numpy need not be installed, and only needs `macros`.
import importlib

_SUBMODULES = ("angular", "angular_driver", "binning", "chargepdf", "cprofile",
               "isotropic_sample", "macros", "scattable_driver", "particles", "rootio", "scattable",
               "timepdf")

__all__ = list(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
