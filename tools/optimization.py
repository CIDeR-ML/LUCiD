from pandas.tests.tseries.offsets.test_business_day import offset
from tools.utils import generate_random_point_inside_cylinder

import jax.numpy as jnp
from jax import jit
from typing import Tuple

@jit
def get_initial_guess(
        charges: jnp.ndarray,
        detector_points: jnp.ndarray,
        intensity_scale: float = 5.0,
        key: jnp.ndarray = None,
        offset: float = 0.5
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate initial parameter guesses from detector hits.
    Assumes photons originate from center and form a cone to the detector wall.
    Only considers nonzero charge hits for calculations.

    Parameters
    ----------
    charges : jnp.ndarray
        Array of charge values for detector hits
    detector_points : jnp.ndarray
        Array of detector positions where hits occurred
    intensity_scale : float
        Scale factor for intensity estimation
    key : jnp.ndarray
        Key for random initial position generation
    offset : float
        Offset from the wall used for random initial position generation

    Returns
    -------
    Tuple containing:
        opening_angle : jnp.ndarray (scalar)
        position : jnp.ndarray (3,)
        direction : jnp.ndarray (3,)
        intensity : jnp.ndarray (scalar)
    """
    if key is not None:
        position = generate_random_point_inside_cylinder(key, offset=offset)
    else:
        position = jnp.zeros(3)

    # Get average direction by weighting detector positions by charge
    total_charge = jnp.sum(charges)
    charge_weights = charges / (total_charge + 1e-8)

    vectors_to_hits = detector_points - position
    vectors_to_hits = vectors_to_hits / (jnp.linalg.norm(vectors_to_hits, axis=1, keepdims=True) + 1e-8)

    # Get direction from vectors
    direction = jnp.sum(vectors_to_hits * charge_weights[:, None], axis=0)
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)

    # Calculate angles
    cos_angles = jnp.clip(jnp.dot(vectors_to_hits, direction), -1.0, 1.0)
    angles = jnp.arccos(cos_angles)

    # Weighted RMS angle
    opening_angle = jnp.degrees(jnp.sqrt(jnp.sum(angles ** 2 * charge_weights)))
    opening_angle = jnp.clip(opening_angle, 1.0, 80.0)

    # Intensity is just total charge scaled
    intensity = total_charge * intensity_scale

    return (
        opening_angle,
        position,
        direction,
        intensity
    )