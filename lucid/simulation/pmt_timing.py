"""Per-photoelectron PMT timing-response models.

These models act after optical propagation and before hit aggregation or
electronics digitization.  They change only photon timestamps; charge and
sensor assignment remain untouched.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


# SK-IV timing mixture copied from SKDetSim ``sgpmt.F``.  The original code
# draws one uniform number and applies the first cumulative threshold passed:
#
#   <0.01400  N(111.5, 7.0)
#   <0.02700  N( 39.0, 5.0)
#   <0.05075  N( 80.0,25.0)
#   <0.05100  N(-47.0, 8.0)
#   <0.06100  N( 15.0, 4.5)
#   otherwise 0
#
# Values are nanoseconds.  This is the SK_GEOMETRY>=3 / non-legacy-unitune
# branch used by the SK-IV laser sample.
SK4_COMPONENT_PROBABILITIES = jnp.asarray(
    [0.01400, 0.01300, 0.02375, 0.00025, 0.01000, 0.93900]
)
SK4_COMPONENT_MEANS_NS = jnp.asarray([111.5, 39.0, 80.0, -47.0, 15.0, 0.0])
SK4_COMPONENT_SIGMAS_NS = jnp.asarray([7.0, 5.0, 25.0, 8.0, 4.5, 0.0])
# Only the five explicit SKDetSim thresholds are needed. Leaving the final
# component implicit guarantees ``searchsorted`` always returns an in-bounds
# component even if the probabilities are represented in float32.
SK4_CUMULATIVE_PROBABILITIES = jnp.cumsum(SK4_COMPONENT_PROBABILITIES[:-1])

# Derive timing draws from a dedicated substream without advancing the key used
# later for QE, Gaussian TTS, or charge resolution. This makes paired runs with
# timing off/on differ only in time response, which is useful for validation and
# avoids a configuration option silently changing the charge realization.
_SK4_RNG_STREAM_ID = 0x534B3454  # ASCII-ish "SK4T"


def sample_sk4_time_offsets(key, shape) -> jnp.ndarray:
    """Sample the SK-IV ``sgpmt.F`` accepted-PE timing offset in ns.

    The generated distribution matches SKDetSim statistically; it does not try
    to reproduce SKDetSim's generator state or individual random draws.
    """
    timing_key = jax.random.fold_in(key, _SK4_RNG_STREAM_ID)
    component_key, gaussian_key = jax.random.split(timing_key)
    uniform = jax.random.uniform(component_key, shape=shape)
    component = jnp.searchsorted(
        SK4_CUMULATIVE_PROBABILITIES, uniform, side="right"
    )
    mean = SK4_COMPONENT_MEANS_NS[component]
    sigma = SK4_COMPONENT_SIGMAS_NS[component]
    offset = mean + sigma * jax.random.normal(gaussian_key, shape=shape)
    return offset


def apply_sk4_pmt_timing(times, key):
    """Apply the SK-IV timing mixture to finite, positive photon times."""
    times = jnp.asarray(times)
    offsets = sample_sk4_time_offsets(key, times.shape)
    valid = jnp.isfinite(times) & (times > 0)
    return jnp.where(valid, times + offsets, times), key


def identity_pmt_timing(times, key):
    """Return timestamps and RNG state unchanged."""
    return times, key


def get_pmt_timing_model(
    model: Optional[str],
) -> Callable[[jnp.ndarray, jax.Array], Tuple[jnp.ndarray, jax.Array]]:
    """Resolve a setup-time PMT timing model name.

    ``None`` and ``"none"`` preserve the historical LUCiD output and do not
    consume random numbers.  ``"sk4"`` selects the SK-IV ``sgpmt.F`` mixture.
    """
    if model is None or model == "none":
        return identity_pmt_timing
    if model == "sk4":
        return apply_sk4_pmt_timing
    raise ValueError(
        f"Unknown PMT timing model {model!r}; choices are None, 'none', 'sk4'"
    )
