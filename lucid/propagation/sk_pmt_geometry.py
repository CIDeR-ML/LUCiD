"""Geometry definitions for the SK-like sphere-plus-torus PMT.

Lengths are in metres in the PMT-local frame, with +z pointing toward the
water. The sphere describes the front bulb; the torus describes the rim.
This module contains geometry definitions only, with no ray-intersection solver.
"""

# SK-like PMT geometry constants in metres.
SPHERE_CENTER_Z = -0.127
SPHERE_RADIUS = 0.315
TORUS_MAJOR_RADIUS = 0.104
TORUS_MINOR_RADIUS = 0.150
GEANT_PMT_RADIUS = 0.254
SPHERE_TO_TORUS_Z = (
    TORUS_MINOR_RADIUS * -SPHERE_CENTER_Z
    / (SPHERE_RADIUS - TORUS_MINOR_RADIUS)
)
BARREL_INACTIVE_BAND_Z = 0.020
