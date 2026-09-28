"""A synthetic calibration problem for the fitting tests: no simulator, no detector.

Per-sensor charge is log-linear in three optical fields through a fixed response matrix per
source, times the per-PMT gains. Each test picks its own size and seed. (Not a test module: the
leading underscore keeps pytest from collecting it.)
"""
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import DetectorParams, _flatten_detector_params

FIELDS = ['scatter_length', 'absorption_length', 'qe']


def dp_true(ns):
    return DetectorParams.from_flat(
        scatter_length=30.0, absorption_length=80.0, qe=0.25,
        wall_reflection_rate=0.4, sensor_reflection_rate=0.2, qe_corrections=np.ones(ns))


def sources(ns, seed, b_spread, b_scale):
    """Two sources: a smooth response and a random one (``b_spread`` wide, ``b_scale`` bright)."""
    rng = np.random.default_rng(seed)
    ra = jnp.asarray(np.linspace(0.2, 1.0, ns)[:, None] * np.cos(np.arange(len(FIELDS)) + 1.0))
    rb = jnp.asarray(rng.standard_normal((ns, len(FIELDS))) * b_spread)
    return [{'resp': ra, 'scale': jnp.asarray(1.0)}, {'resp': rb, 'scale': jnp.asarray(b_scale)}]


def charge(source, dp):
    """Mean per-sensor charge at ``dp``, gains included."""
    f = _flatten_detector_params(dp)
    logs = jnp.stack([jnp.log(jnp.asarray(f[k]).reshape(())) for k in FIELDS])
    return jnp.exp(source['resp'] @ logs) * source['scale'] * jnp.asarray(f['qe_corrections'])
