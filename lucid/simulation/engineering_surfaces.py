"""Shared parameter contract for effective JUNO engineering surfaces."""

import numpy as np


SURFACE_NAMES = (
    "support",
    "chimney_acrylic",
    "chimney_steel",
    "acrylic_bowl",
    "diffuser_ptfe",
    "diffuser_peek",
    "diffuser_fiber",
    "bottom_chimney_acrylic",
    "bottom_chimney_steel",
)


def engineering_surface_parameters(physics_config, required_codes=()):
    """Return validated R, T, and specular fractions by surface code.

    Parameters for enabled surface codes are mandatory.  Disabled codes get
    inert defaults so older detector configurations remain loadable.
    """
    required = set(required_codes)
    for code in required:
        name = SURFACE_NAMES[code]
        for suffix in ("reflection_rate", "fspec"):
            key = f"{name}_{suffix}"
            if key not in physics_config:
                raise ValueError(
                    f"Enabled engineering surface requires {key!r}"
                )
    reflectances = np.asarray([
        physics_config.get(f"{name}_reflection_rate", 0.0)
        for name in SURFACE_NAMES
    ], dtype=float)
    transmittances = np.asarray([
        physics_config.get(f"{name}_transmission_rate", 0.0)
        for name in SURFACE_NAMES
    ], dtype=float)
    fspecs = np.asarray([
        physics_config.get(f"{name}_fspec", 1.0)
        for name in SURFACE_NAMES
    ], dtype=float)
    if np.any(reflectances < 0.0) or np.any(transmittances < 0.0):
        raise ValueError("Engineering surface A/R/T rates must be non-negative")
    if np.any(reflectances + transmittances > 1.0 + 1e-12):
        raise ValueError("Engineering surface R + T must not exceed one")
    if np.any((fspecs < 0.0) | (fspecs > 1.0)):
        raise ValueError("Engineering surface fspec must lie in [0, 1]")
    return reflectances, transmittances, fspecs
