"""The composed forward model on the tiny detector, against references computed outside it.

Each test runs the whole chain -- source, propagator (grid lookup, sensor intersection, first
hit), photon step, hit aggregation -- on the session simulators of ``tests/conftest.py``
(``tests/helpers/tiny.py``):

- the per-sensor charge of an isotropic source equals a float64 ray trace of the same rays;
- photons mirrored through the detector's z = 0 plane light the mirrored sensors, exactly;
- the gradient of the charge in the absorption length is the Beer-Lambert derivative, and the
  central difference of the forward.
"""
import jax
import jax.numpy as jnp
import numpy as np

from tests.helpers import tiny

N, QE = tiny.N, tiny.QE


def _first_sensor_hit(origins, directions, sensors, sensor_radius, radius, height):
    """Float64 ray trace: the sensor sphere each ray enters first before leaving the cylinder,
    or -1. The reference the simulator's deposit is held to."""
    o, d = np.asarray(origins, np.float64), np.asarray(directions, np.float64)
    # exit through the barrel (x^2 + y^2 = R^2) or a cap (z = +-H/2)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - radius ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t_barrel = np.where(a > 0, (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a), np.inf)
        t_cap = np.where(d[:, 2] != 0, (np.sign(d[:, 2]) * height / 2 - o[:, 2]) / d[:, 2], np.inf)
    t_exit = np.minimum(t_barrel, t_cap)
    # entry into each sensor sphere
    oc = o[:, None, :] - np.asarray(sensors, np.float64)[None]
    bb = np.einsum('nk,nsk->ns', d, oc)
    disc = bb * bb - (np.einsum('nsk,nsk->ns', oc, oc) - sensor_radius ** 2)
    t_in = np.where(disc > 0, -bb - np.sqrt(np.maximum(disc, 0.0)), np.inf)
    t_in = np.where(t_in > 0, t_in, np.inf)
    first = np.argmin(t_in, axis=1)
    hit = t_in[np.arange(len(o)), first] <= t_exit
    return np.where(hit, first, -1)


def test_isotropic_charge_is_a_direct_ray_trace(tiny_sim, tiny_detector):
    """Through transparent water with black walls, a ray deposits its weight I/N on the first PMT
    whose sphere it enters before the wall, so each PMT's charge is QE (I/N) x the number of
    such rays. The isotropic source is a deterministic Fibonacci lattice, so the reference traces
    exactly the rays the simulator propagates. It agrees to float32 rounding on every one of the
    582 PMTs (measured: max |difference| 2e-6 of a 0.4 charge unit), and the charge that is not
    on a PMT is the light the walls absorbed: the total is QE I x the covered fraction."""
    from lucid.sources import isotropic_source
    from lucid.sources.calibration_sources import get_isotropic_rays
    position = np.array([0.21, -0.13, 0.32])       # off every symmetry axis
    intensity = 2.0 * N                            # weight 2 per ray
    params = tiny.transparent(tiny_sim.det_geom.num_sensors)
    q = np.asarray(tiny_sim(isotropic_source(position, intensity=intensity), params,
                            jax.random.PRNGKey(0))[0])

    dirs, origins, _ = get_isotropic_rays(jnp.asarray(position, jnp.float32),
                                          jnp.float32(intensity), N, jax.random.PRNGKey(0))
    det = tiny_detector
    first = _first_sensor_hit(origins, dirs, det.all_points, float(det.S_radius), det.r, det.H)
    counts = np.bincount(first[first >= 0], minlength=len(q))
    expected = QE * (intensity / N) * counts
    assert counts.sum() > 0.1 * N                  # the trace has something to agree with
    np.testing.assert_allclose(q, expected, rtol=0, atol=1e-4)
    np.testing.assert_allclose(q.sum(), QE * intensity * np.mean(first >= 0), rtol=1e-5)


def test_mirrored_photons_light_the_mirrored_sensors(tiny_data_sim, tiny_detector):
    """The detector is symmetric under z -> -z (the caps are one layout at +-H/2, the barrel
    rings symmetric about z = 0), so photons mirrored through z = 0 give the charge pattern
    permuted by the sensor mirror map. Known photons in data mode make it exact: no sampling
    enters a transparent, black-walled, one-step forward."""
    pts = np.asarray(tiny_detector.all_points)
    index = {tuple(np.round(p, 6)): i for i, p in enumerate(pts)}
    mirror = np.array([index[tuple(np.round(p * [1, 1, -1], 6))] for p in pts])

    rng = np.random.default_rng(0)
    origins = rng.uniform(-0.5, 0.5, (N, 3))
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    params = tiny.transparent(len(pts))

    def charge(o, d):
        return np.asarray(tiny_data_sim(params, jax.random.PRNGKey(1), tiny.photon_data(o, d))[0])

    flip = np.array([1.0, 1.0, -1.0])
    q, q_mirrored = charge(origins, dirs), charge(origins * flip, dirs * flip)
    assert q.sum() > 0.1 * N * QE
    assert np.count_nonzero(q[:len(tiny_detector.barr_points)]) > 0    # barrel and caps both lit
    assert np.count_nonzero(q[len(tiny_detector.barr_points):]) > 0
    np.testing.assert_array_equal(q_mirrored, q[mirror])


def test_absorption_gradient_is_the_beer_lambert_derivative(tiny_sim, tiny_detector):
    """A pencil beam on a PMT a path d away through water of absorption length L carries
    Q = N QE exp(-d/L), so dQ/dL = Q d / L^2. The forward and the autodiff gradient through the
    whole simulator equal those closed forms to 1e-4 (measured 9e-6 and 2e-6), and the gradient
    equals the central difference of the forward at a fixed key. The step is 10 % of L: the
    float32 forward carries ~3e-5 relative noise from one L to the next (measured), which a
    smaller step would amplify, and the step's own truncation error is 8e-4."""
    from lucid.sources import laser_source
    target, centre, out = tiny.barrel_target(tiny_detector)
    path = np.hypot(centre[0], centre[1]) - float(tiny_detector.S_radius)
    source = laser_source(position=[0.0, 0.0, centre[2]], direction=out.tolist(),
                          intensity=N, fiber_NA=1e-6)
    length = 3.0

    def charge(absorption_length):
        params = tiny.transparent(tiny_sim.det_geom.num_sensors, sensor_reflection_rate=0.25,
                                  absorption_length=absorption_length)
        return tiny_sim(source, params, jax.random.PRNGKey(3))[0][target]

    q, dq = jax.value_and_grad(charge)(length)
    closed = N * QE * np.exp(-path / length)
    np.testing.assert_allclose(float(q), closed, rtol=1e-4)
    np.testing.assert_allclose(float(dq), closed * path / length ** 2, rtol=1e-4)
    h = 0.1 * length
    central = (float(charge(length + h)) - float(charge(length - h))) / (2 * h)
    np.testing.assert_allclose(float(dq), central, rtol=1e-2)
