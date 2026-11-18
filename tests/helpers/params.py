"""The two DetectorParams presets several test modules build by hand.

``lab_dp`` sets every optical scalar to a distinct round value, so a field read from the wrong
leaf shows up. ``calib_dp`` is the 50 m / 50 m / 0.2 water the wavelength and QE-sampling
simulator tests run at. ``leaf``, ``with_leaf`` and ``central_difference`` read, replace and
finite-difference one leaf by its flat name, for the gradient tests. ``on_deposits`` runs a
``lucid.simulation.hits`` function on hand-built deposits.
"""
import jax.numpy as jnp

from lucid.detector_params import DetectorParams


def lab_dp(num_sensors=10):
    """scatter 50 m, Mie 1000 m (g 0.9), absorption 100 m, wall 0.5, sensor 0.3, qe 0.2."""
    return DetectorParams.from_flat(
        scatter_length=50.0, mie_scatter_length=1000.0, g=0.9,
        wall_reflection_rate=0.5, sensor_reflection_rate=0.3,
        absorption_length=100.0, qe=0.2, qe_corrections=jnp.ones(num_sensors))


def calib_dp(num_sensors):
    """scatter 50 m, absorption 50 m, wall and sensor reflection 0.2, qe 0.2, no Mie."""
    return DetectorParams.from_flat(
        scatter_length=50.0, wall_reflection_rate=0.2, sensor_reflection_rate=0.2,
        absorption_length=50.0, qe=0.2, qe_corrections=jnp.ones(num_sensors))


def leaf(dp, field):
    """The DetectorParams (or gradient pytree) leaf ``field``, by its flat name, as a float."""
    from lucid.detector_params import _flatten_detector_params
    return float(_flatten_detector_params(dp)[field])


def with_leaf(dp, field, value):
    """``dp`` with the leaf ``field`` (flat name, e.g. ``'qe'``) replaced by ``value``."""
    from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
    flat = _flatten_detector_params(dp)
    if field not in flat:
        raise KeyError(field)
    return _nest_flat_kwargs({**flat, field: jnp.asarray(value, dtype=jnp.float32)})


def central_difference(f, dp, field, rel_step=0.01):
    """``(f(dp, field + h) - f(dp, field - h)) / 2h`` with ``h = rel_step * |field|``."""
    x = leaf(dp, field)
    h = rel_step * abs(x)
    return (float(f(with_leaf(dp, field, x + h))) - float(f(with_leaf(dp, field, x - h)))) / (2 * h)


def on_deposits(hits_fn, flat_weights, flat_indices, flat_times, num_sensors, *, qe,
                qe_corrections, key=None, tts=0.0, **options):
    """``hits_fn`` on hand-built deposits: response tts ``tts``, no segment index, none indirect."""
    response = DetectorParams.from_flat(tts=tts).response
    return hits_fn(flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections,
                   response, None, None, **options)
