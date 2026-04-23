"""High-level helpers for photon-shotgun production runs.

Includes position-grid builders (uniform-in-cylinder, regular grid), source
construction for batched cases, and a small convenience for building the
simulator + generator pair for a given detector geometry.
"""
from typing import Optional, Tuple

import json
import numpy as np

import jax
import jax.numpy as jnp

from lucid.sources.shotgun_source import shotgun_source, stack_shotgun_sources


# ---------------------------------------------------------------------------
# Position / direction samplers
# ---------------------------------------------------------------------------

def read_detector_bounds(json_path: str) -> dict:
    """Extract axis-aligned detector dimensions from a geometry config JSON.

    For algorithmic configs the dimensions live directly in
    ``geometry_definitions`` (``radius``/``height``/``length``/``width``).
    For file-loaded cylinders the JSON only carries ``npz_file_path`` —
    we fall back to building the detector to read the dimensions off
    its scalars.

    Returns:
        cylinder → ``{type: 'cylinder', r, H}``
        sphere   → ``{type: 'sphere', r}``
        box      → ``{type: 'box', x, y, z}``
        (units: meters)
    """
    with open(json_path) as f:
        cfg = json.load(f)
    t = cfg.get('detector_type', 'cylinder').lower()
    geom = cfg.get('geometry_definitions', cfg)

    if t == 'cylinder':
        if 'npz_file_path' in geom:
            # File-loaded geometry — bounds live in the .npz, not the JSON.
            from lucid.geometry import generate_detector
            det = generate_detector(json_path)
            return dict(type='cylinder', r=float(det.r), H=float(det.H))
        return dict(type='cylinder',
                    r=float(geom['radius']), H=float(geom['height']))
    if t == 'sphere':
        return dict(type='sphere', r=float(geom['radius']))
    if t == 'box':
        return dict(type='box',
                    x=float(geom.get('x_size', geom['x'])),
                    y=float(geom.get('y_size', geom['y'])),
                    z=float(geom.get('z_size', geom['z'])))
    raise ValueError(f"Unknown detector_type in {json_path}: {t}")


def sample_positions_uniform(
    n: int, bounds: dict, *,
    fraction: float = 0.9, key=None,
) -> np.ndarray:
    """Uniformly sample ``n`` positions inside the detector volume.

    ``fraction`` shrinks the sampling region so positions stay away from the
    wall (e.g., 0.9 ⇒ 90 % of each dimension).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    if bounds['type'] == 'cylinder':
        r = bounds['r'] * fraction
        h = bounds['H'] * fraction
        k1, k2, k3 = jax.random.split(key, 3)
        rr = r * jnp.sqrt(jax.random.uniform(k1, (n,)))
        th = jax.random.uniform(k2, (n,), minval=0.0, maxval=2 * jnp.pi)
        zz = jax.random.uniform(k3, (n,), minval=-h / 2, maxval=h / 2)
        return np.asarray(jnp.stack([rr * jnp.cos(th), rr * jnp.sin(th), zz], axis=-1))
    if bounds['type'] == 'sphere':
        r_max = bounds['r'] * fraction
        k1, k2, k3 = jax.random.split(key, 3)
        u = jax.random.uniform(k1, (n,))
        ct = jax.random.uniform(k2, (n,), minval=-1.0, maxval=1.0)
        ph = jax.random.uniform(k3, (n,), minval=0.0, maxval=2 * jnp.pi)
        rr = r_max * jnp.cbrt(u)
        st = jnp.sqrt(1 - ct ** 2)
        return np.asarray(jnp.stack(
            [rr * st * jnp.cos(ph), rr * st * jnp.sin(ph), rr * ct], axis=-1))
    if bounds['type'] == 'box':
        lo = jnp.array([-bounds['x'], -bounds['y'], -bounds['z']]) * fraction / 2
        hi = jnp.array([bounds['x'], bounds['y'], bounds['z']]) * fraction / 2
        return np.asarray(jax.random.uniform(key, (n, 3), minval=lo, maxval=hi))
    raise ValueError(f"Unknown bounds type: {bounds['type']}")


def sample_directions_isotropic(n: int, key=None) -> np.ndarray:
    """Uniform directions on the unit sphere."""
    if key is None:
        key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (n, 3))
    return np.asarray(v / jnp.linalg.norm(v, axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Source batch construction
# ---------------------------------------------------------------------------

def build_case_sources(
    n_cases: int, n_photons: int,
    positions: np.ndarray, directions: np.ndarray,
    *, wavelength='cherenkov', intensity: float = 1.0,
):
    """Construct a list of ``n_cases`` ShotgunSource objects.

    ``positions`` / ``directions`` are shape ``(n_cases, 3)`` — one fixed
    (origin, direction) broadcast across the ``n_photons`` photons of each
    case. Wavelength is the same for every case (scalar or 'cherenkov').
    """
    if positions.shape != (n_cases, 3):
        raise ValueError(f"positions must be (n_cases, 3); got {positions.shape}")
    if directions.shape != (n_cases, 3):
        raise ValueError(f"directions must be (n_cases, 3); got {directions.shape}")
    sources = []
    for i in range(n_cases):
        sources.append(shotgun_source(
            positions[i], directions[i], n_photons=n_photons,
            wavelength=wavelength, intensity=intensity))
    return sources


def batched_source_iter(sources, chunk: int):
    """Yield ``stack_shotgun_sources`` chunks of size ``chunk``."""
    for i in range(0, len(sources), chunk):
        yield stack_shotgun_sources(sources[i:i + chunk])
