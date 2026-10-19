"""Differentiable SK-IV acrylic-cover transport.

The SK 20-inch PMT is covered by a 13 mm spherical acrylic shell.  The trace
contains the directly transmitted path plus two leading SK-like reflected
paths: reflection from the outer water/acrylic face and
reflection from the inner acrylic/water face followed by escape into the ID.
The simulator can either sample those paths or deliberately absorb them for
the historical ballistic diagnostic.
"""

from pathlib import Path
from typing import NamedTuple
import json
import os

import numpy as np

import jax.numpy as jnp

from lucid.wavelength.medium import (
    sk4_water_group_refractive_index,
    sk4_water_refractive_index,
)

from .sk_pmt_jax import intersect_sk20inch_pmt_jax


# SK-like cover geometry in metres.
SK_ACRYLIC_RADIUS = 0.3165
SK_ACRYLIC_THICKNESS = 0.013
SK_ACRYLIC_OUTER_RADIUS = SK_ACRYLIC_RADIUS + SK_ACRYLIC_THICKNESS / 2.0
SK_ACRYLIC_INNER_RADIUS = SK_ACRYLIC_RADIUS - SK_ACRYLIC_THICKNESS / 2.0
SK_ACRYLIC_CENTER_OFFSET = SK_ACRYLIC_RADIUS - 0.2125


class SKAcrylicOptics(NamedTuple):
    refractive_index: jnp.ndarray
    absorption_coefficient: jnp.ndarray
    water_refractive_index: jnp.ndarray
    water_group_refractive_index: jnp.ndarray


class SKAcrylicPMTTrace(NamedTuple):
    cover_intersects: jnp.ndarray
    cover_position: jnp.ndarray
    cover_time_distance: jnp.ndarray
    cover_transmission: jnp.ndarray
    intersects: jnp.ndarray
    active: jnp.ndarray
    distance: jnp.ndarray
    time_distance: jnp.ndarray
    position: jnp.ndarray
    normal: jnp.ndarray
    incident_direction: jnp.ndarray
    transmission: jnp.ndarray
    acrylic_distance: jnp.ndarray
    reflected: "SKAcrylicReflectionTransport"


class SKAcrylicReflectionTransport(NamedTuple):
    """First-order reflected branches at an SK-IV acrylic PMT cover."""

    outer_position: jnp.ndarray
    outer_direction: jnp.ndarray
    outer_time_distance: jnp.ndarray
    outer_probability: jnp.ndarray
    inner_exit_valid: jnp.ndarray
    inner_exit_position: jnp.ndarray
    inner_exit_direction: jnp.ndarray
    inner_exit_time_distance: jnp.ndarray
    inner_exit_probability: jnp.ndarray


class SKAcrylicExitTrace(NamedTuple):
    valid: jnp.ndarray
    position: jnp.ndarray
    direction: jnp.ndarray
    transmission: jnp.ndarray
    time_distance: jnp.ndarray
    acrylic_distance: jnp.ndarray


def load_sk4_acrylic_optics(wavelength_nm, table_path=None):
    """Interpolate a user-supplied acrylic optical table at one wavelength.

    Supply a JSON path explicitly or set ``LUCID_SK4_ACRYLIC_TABLE``.
    No detector optical table is bundled. The JSON must contain three
    equally sized one-dimensional arrays with at least two entries:
    ``wavelength_nm`` (positive, strictly increasing, in nanometres),
    ``refractive_index`` (positive), and ``absorption_coefficient_per_m``
    (nonnegative). All entries must be finite.

    Explicit paths take precedence over the environment variable. To preserve
    the SK comparison convention, wavelengths outside the supplied range use
    index 1.49 and zero absorption rather than extrapolation. This loader runs
    during simulator setup; it is not a fit of the table entries.
    """
    if table_path is None:
        table_path = os.environ.get("LUCID_SK4_ACRYLIC_TABLE")
        if not table_path:
            raise FileNotFoundError(
                "No acrylic optical table was configured. Pass table_path=... "
                "(sk_acrylic_table=... in setup_event_simulator) or set "
                "LUCID_SK4_ACRYLIC_TABLE.")
    path = Path(table_path).expanduser().resolve()
    with path.open() as handle:
        table = json.load(handle)
    keys = ("wavelength_nm", "refractive_index", "absorption_coefficient_per_m")
    try:
        arrays = tuple(np.asarray(table[key], dtype=float) for key in keys)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Acrylic optical table requires numeric arrays: " + ", ".join(keys)
        ) from error
    wavelength, refractive_index, absorption = arrays
    if (any(values.ndim != 1 for values in arrays)
            or wavelength.size < 2
            or any(values.shape != wavelength.shape for values in arrays)):
        raise ValueError(
            "Acrylic optical table arrays must be one-dimensional, have equal "
            "lengths, and contain at least two entries")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Acrylic optical table entries must be finite")
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Acrylic wavelengths must be positive and strictly increasing")
    if np.any(refractive_index <= 0) or np.any(absorption < 0):
        raise ValueError(
            "Acrylic refractive indices must be positive and absorption "
            "coefficients must be nonnegative")
    wavelength, refractive_index, absorption = (
        jnp.asarray(values, dtype=jnp.float32) for values in arrays)
    lam = jnp.asarray(wavelength_nm, dtype=jnp.float32)
    # Preserve the SK comparison fallback outside the supplied table range.
    in_range = (lam >= wavelength[0]) & (lam <= wavelength[-1])
    n_acrylic = jnp.where(
        in_range, jnp.interp(lam, wavelength, refractive_index), 1.49)
    absorption_coefficient = jnp.where(
        in_range, jnp.interp(lam, wavelength, absorption), 0.0)
    return SKAcrylicOptics(
        refractive_index=n_acrylic,
        absorption_coefficient=absorption_coefficient,
        water_refractive_index=sk4_water_refractive_index(lam),
        water_group_refractive_index=sk4_water_group_refractive_index(lam),
    )


def _first_positive_sphere_root(origin, direction, center, radius):
    offset = origin - center
    b = jnp.dot(offset, direction)
    c = jnp.dot(offset, offset) - radius * radius
    discriminant = b * b - c
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-12)
    near = -b - root
    far = -b + root
    distance = jnp.where(near > 1e-6, near, far)
    valid = (discriminant >= 0.0) & (distance > 1e-6)
    return distance, valid


def _refract_and_transmission(direction, incident_normal, n1, n2):
    """Return the transmitted direction and SK-IV Fresnel weight.

    ``incident_normal`` points into the incident medium.  SK-IV combined the
    Fresnel components with the incidence angle itself, rather than tracked
    polarization: R = Rp cos(theta)^2 + Rs sin(theta)^2.
    """
    cosine1 = jnp.clip(-jnp.dot(direction, incident_normal), 0.0, 1.0)
    eta = n1 / n2
    sine2_squared = eta * eta * jnp.maximum(1.0 - cosine1 * cosine1, 0.0)
    valid = sine2_squared < 1.0
    cosine2 = jnp.sqrt(jnp.maximum(1.0 - sine2_squared, 0.0) + 1e-12)
    transmitted = (
        eta * direction + (eta * cosine1 - cosine2) * incident_normal)
    transmitted = transmitted / (jnp.linalg.norm(transmitted) + 1e-12)

    rs = ((n1 * cosine1 - n2 * cosine2)
          / (n1 * cosine1 + n2 * cosine2 + 1e-12)) ** 2
    rp = ((n1 / (cosine1 + 1e-12) - n2 / (cosine2 + 1e-12))
          / (n1 / (cosine1 + 1e-12) + n2 / (cosine2 + 1e-12))) ** 2
    reflectivity = rp * cosine1 * cosine1 + rs * (1.0 - cosine1 * cosine1)
    transmission = jnp.where(valid, jnp.clip(1.0 - reflectivity, 0.0, 1.0), 0.0)
    return transmitted, transmission, valid


def _reflect(direction, incident_normal):
    """Specular reflection for a normal pointing into the incident medium."""
    cosine = jnp.clip(-jnp.dot(direction, incident_normal), 0.0, 1.0)
    reflected = direction + 2.0 * cosine * incident_normal
    return reflected / (jnp.linalg.norm(reflected) + 1e-12)


def trace_sk4_acrylic_to_pmt(
        ray_origin, ray_direction, pmt_position, pmt_axis, optics,
        *, barrel=False):
    """Trace a ray through the SK-IV cover and onto the curved PMT surface."""
    direction = ray_direction / (jnp.linalg.norm(ray_direction) + 1e-12)
    axis = pmt_axis / (jnp.linalg.norm(pmt_axis) + 1e-12)
    cover_center = pmt_position - SK_ACRYLIC_CENTER_OFFSET * axis
    starts_in_gap = (
        jnp.linalg.norm(ray_origin - cover_center)
        < SK_ACRYLIC_INNER_RADIUS - 1e-5)

    outer_distance, outer_valid = _first_positive_sphere_root(
        ray_origin, direction, cover_center, SK_ACRYLIC_OUTER_RADIUS)
    outer_position = ray_origin + outer_distance * direction
    outer_normal = (outer_position - cover_center) / SK_ACRYLIC_OUTER_RADIUS
    # GEANT defines SPCW/SPCT/SPCB as hemispheres whose domes face the ID.
    # The continuation behind the equator is handled separately by
    # SGACREDGE; treating the cover as a full sphere admits grazing rays
    # through acrylic that is not physically present.
    outer_front = jnp.dot(outer_position - cover_center, axis) >= 0.0
    acrylic_direction, first_transmission, first_valid = (
        _refract_and_transmission(
            direction, outer_normal,
            optics.water_refractive_index, optics.refractive_index))
    outer_reflected_direction = _reflect(direction, outer_normal)
    outer_shell_valid = outer_valid & outer_front & first_valid
    outer_reflection_probability = jnp.where(
        outer_shell_valid, 1.0 - first_transmission, 0.0)

    acrylic_distance, inner_valid = _first_positive_sphere_root(
        outer_position, acrylic_direction, cover_center,
        SK_ACRYLIC_INNER_RADIUS)
    inner_position = outer_position + acrylic_distance * acrylic_direction
    inner_normal = (inner_position - cover_center) / SK_ACRYLIC_INNER_RADIUS
    inner_front = jnp.dot(inner_position - cover_center, axis) >= 0.0
    gap_direction, second_transmission, second_valid = (
        _refract_and_transmission(
            acrylic_direction, inner_normal,
            optics.refractive_index, optics.water_refractive_index))

    # If the photon reflects at the inner acrylic/water boundary, follow it
    # back through the shell and through the outer acrylic/water boundary.
    # A further reflection at that last boundary is O(R^2) and is deliberately
    # left in the untracked remainder; the two retained branches dominate the
    # production-SK correction while keeping one continuation per LUCiD ray.
    inner_reflected_direction = _reflect(acrylic_direction, inner_normal)
    return_acrylic_distance, return_outer_valid = _first_positive_sphere_root(
        inner_position, inner_reflected_direction, cover_center,
        SK_ACRYLIC_OUTER_RADIUS)
    inner_exit_position = (
        inner_position + return_acrylic_distance * inner_reflected_direction)
    inner_exit_front = jnp.dot(
        inner_exit_position - cover_center, axis) >= 0.0
    inner_exit_normal = -(
        inner_exit_position - cover_center) / SK_ACRYLIC_OUTER_RADIUS
    inner_exit_direction, third_transmission, third_valid = (
        _refract_and_transmission(
            inner_reflected_direction, inner_exit_normal,
            optics.refractive_index, optics.water_refractive_index))

    pmt_hit = intersect_sk20inch_pmt_jax(
        inner_position, gap_direction, pmt_position, axis, barrel=barrel)
    direct_hit = intersect_sk20inch_pmt_jax(
        ray_origin, direction, pmt_position, axis, barrel=barrel)
    reaches_inner = (
        outer_shell_valid & inner_valid & inner_front)
    shell_valid = reaches_inner & second_valid
    survival = jnp.exp(
        -optics.absorption_coefficient * jnp.maximum(acrylic_distance, 0.0))
    cover_transmission = jnp.where(
        shell_valid,
        first_transmission * second_transmission * survival,
        0.0)
    return_survival = jnp.exp(
        -optics.absorption_coefficient
        * jnp.maximum(return_acrylic_distance, 0.0))
    inner_exit_valid = (
        reaches_inner & return_outer_valid & inner_exit_front & third_valid)
    inner_exit_probability = jnp.where(
        inner_exit_valid,
        first_transmission * survival * (1.0 - second_transmission)
        * return_survival * third_transmission,
        0.0)
    total_distance = outer_distance + acrylic_distance + pmt_hit.distance
    # The simulator converts this value with the water group velocity.
    # Use the acrylic phase index for the cover segment and the water group
    # index for water, encoding the delay as an equivalent water-path distance.
    time_distance = total_distance + acrylic_distance * (
        optics.refractive_index / optics.water_group_refractive_index - 1.0)
    cover_time_distance = outer_distance + acrylic_distance + acrylic_distance * (
        optics.refractive_index / optics.water_group_refractive_index - 1.0)
    inner_reflected_acrylic_distance = (
        acrylic_distance + return_acrylic_distance)
    inner_exit_time_distance = (
        outer_distance + inner_reflected_acrylic_distance
        + inner_reflected_acrylic_distance * (
            optics.refractive_index
            / optics.water_group_refractive_index - 1.0))
    intersects = jnp.where(
        starts_in_gap, direct_hit.intersects,
        shell_valid & pmt_hit.intersects)
    active = jnp.where(
        starts_in_gap, direct_hit.active,
        shell_valid & pmt_hit.active)
    return SKAcrylicPMTTrace(
        cover_intersects=(~starts_in_gap) & shell_valid,
        cover_position=jnp.where(
            starts_in_gap, ray_origin, inner_position),
        cover_time_distance=jnp.where(
            starts_in_gap, 0.0, cover_time_distance),
        cover_transmission=jnp.where(
            starts_in_gap, 1.0, cover_transmission),
        intersects=intersects,
        active=active,
        distance=jnp.where(
            starts_in_gap, direct_hit.distance, total_distance),
        time_distance=jnp.where(
            starts_in_gap, direct_hit.distance, time_distance),
        position=jnp.where(
            starts_in_gap, direct_hit.position, pmt_hit.position),
        normal=jnp.where(
            starts_in_gap, direct_hit.normal, pmt_hit.normal),
        incident_direction=jnp.where(
            starts_in_gap, direction, gap_direction),
        transmission=jnp.where(
            starts_in_gap, 1.0, cover_transmission),
        acrylic_distance=jnp.where(
            starts_in_gap, 0.0, acrylic_distance),
        reflected=SKAcrylicReflectionTransport(
            outer_position=jnp.where(
                starts_in_gap, ray_origin, outer_position),
            outer_direction=jnp.where(
                starts_in_gap, direction, outer_reflected_direction),
            outer_time_distance=jnp.where(
                starts_in_gap, 0.0, outer_distance),
            outer_probability=jnp.where(
                starts_in_gap, 0.0, outer_reflection_probability),
            inner_exit_valid=(~starts_in_gap) & inner_exit_valid,
            inner_exit_position=jnp.where(
                starts_in_gap, ray_origin, inner_exit_position),
            inner_exit_direction=jnp.where(
                starts_in_gap, direction, inner_exit_direction),
            inner_exit_time_distance=jnp.where(
                starts_in_gap, 0.0, inner_exit_time_distance),
            inner_exit_probability=jnp.where(
                starts_in_gap, 0.0, inner_exit_probability),
        ),
    )


def trace_sk4_acrylic_from_pmt(
        ray_origin, ray_direction, pmt_position, pmt_axis, optics):
    """Trace a PMT-reflected ray from the water gap back into ID water."""
    direction = ray_direction / (jnp.linalg.norm(ray_direction) + 1e-12)
    axis = pmt_axis / (jnp.linalg.norm(pmt_axis) + 1e-12)
    cover_center = pmt_position - SK_ACRYLIC_CENTER_OFFSET * axis

    gap_distance, inner_valid = _first_positive_sphere_root(
        ray_origin, direction, cover_center, SK_ACRYLIC_INNER_RADIUS)
    inner_position = ray_origin + gap_distance * direction
    inner_front = jnp.dot(inner_position - cover_center, axis) >= 0.0
    # At this interface the incident water is inside the inner sphere.
    inner_normal = -(inner_position - cover_center) / SK_ACRYLIC_INNER_RADIUS
    acrylic_direction, first_transmission, first_valid = (
        _refract_and_transmission(
            direction, inner_normal,
            optics.water_refractive_index, optics.refractive_index))

    acrylic_distance, outer_valid = _first_positive_sphere_root(
        inner_position, acrylic_direction, cover_center,
        SK_ACRYLIC_OUTER_RADIUS)
    outer_position = inner_position + acrylic_distance * acrylic_direction
    outer_front = jnp.dot(outer_position - cover_center, axis) >= 0.0
    # Acrylic occupies the inside of the outer boundary.
    outer_normal = -(outer_position - cover_center) / SK_ACRYLIC_OUTER_RADIUS
    water_direction, second_transmission, second_valid = (
        _refract_and_transmission(
            acrylic_direction, outer_normal,
            optics.refractive_index, optics.water_refractive_index))

    valid = (
        inner_valid & inner_front & first_valid
        & outer_valid & outer_front & second_valid)
    survival = jnp.exp(
        -optics.absorption_coefficient * jnp.maximum(acrylic_distance, 0.0))
    transmission = jnp.where(
        valid, first_transmission * second_transmission * survival, 0.0)
    distance = gap_distance + acrylic_distance
    time_distance = distance + acrylic_distance * (
        optics.refractive_index / optics.water_group_refractive_index - 1.0)
    return SKAcrylicExitTrace(
        valid=valid,
        position=outer_position,
        direction=water_direction,
        transmission=transmission,
        time_distance=time_distance,
        acrylic_distance=acrylic_distance,
    )


__all__ = [
    "SKAcrylicOptics",
    "SKAcrylicPMTTrace",
    "SKAcrylicReflectionTransport",
    "SKAcrylicExitTrace",
    "load_sk4_acrylic_optics",
    "trace_sk4_acrylic_to_pmt",
    "trace_sk4_acrylic_from_pmt",
]
