"""`lucid.sources.data_event`: the padded photon buffer and the observed ``(charge, time)``.

pad_photon_data zero-fills the buffer and keeps the true count; an event larger than the buffer
is an error, not a truncation. simulate_data_event hands the simulator that buffer, adds the event's
start time to recorded hits, and reports no time where there is no charge.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.sources.data_event import pad_photon_data, simulate_data_event


def _raw(n):
    rng = np.random.default_rng(0)
    return {'photon_origins': rng.normal(size=(n, 3)), 'photon_directions': rng.normal(size=(n, 3)),
            'photon_times': rng.uniform(size=n), 'wavelengths': rng.uniform(300, 600, size=n)}


def test_pads_to_the_buffer_and_keeps_the_count():
    raw = _raw(5)
    pd, n = pad_photon_data(raw, 8)
    assert n == 5 and int(pd['N']) == 5
    for k in ('photon_origins', 'photon_directions', 'photon_times', 'wavelengths'):
        assert pd[k].shape[0] == 8
        np.testing.assert_allclose(pd[k][:5], raw[k], rtol=1e-6)
    # the tail is empty: no light, and unit directions along +z
    for k in ('photon_origins', 'photon_times', 'wavelengths'):
        assert not np.asarray(pd[k][5:]).any()
    np.testing.assert_array_equal(pd['photon_directions'][5:], np.tile([0., 0., 1.], (3, 1)))


def test_an_event_larger_than_the_buffer_raises():
    with pytest.raises(ValueError, match='nbuf=4'):
        pad_photon_data(_raw(5), 4)


def test_simulate_data_event_passes_the_buffer_and_zeroes_unlit_times():
    seen = {}

    def stub_sim(key, pd):
        seen['N'], seen['shape'] = int(pd['N']), pd['photon_origins'].shape
        return jnp.array([0., 2., 0., 1.]), jnp.array([5., 6., 7., 8.])

    charge, time = simulate_data_event(stub_sim, _raw(5), jax.random.PRNGKey(0), nbuf=8, t0=0.0)
    assert seen == {'N': 5, 'shape': (8, 3)}
    np.testing.assert_array_equal(charge, [0., 2., 0., 1.])
    np.testing.assert_array_equal(time, [0., 6., 0., 8.])


def test_t0_shifts_recorded_times_and_leaves_unlit_sensors_at_zero():
    def stub_sim(key, pd):
        return jnp.array([0., 2., 1.]), jnp.array([5., 6., 7.])

    _, time = simulate_data_event(stub_sim, _raw(5), jax.random.PRNGKey(0), nbuf=8, t0=-3.5)
    np.testing.assert_array_equal(time, [0., 2.5, 3.5])


def test_place_photons_moves_the_primary_vertex_onto_the_drawn_vertex_in_metres():
    from lucid.sources.data_event import place_photons
    raw = _raw(4)
    raw['photon_origins'][0] = [1., 2., 3.]                       # the primary vertex itself
    raw['photon_origins'][1] = [1., 3., 3.]                       # 1 m from it along +y
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])     # +y -> -x
    out = place_photons(raw, primary_vertex=[1., 2., 3.], rotation=R, vertex=[-4., 0., 5.])
    np.testing.assert_allclose(out['photon_origins'][0], [-4., 0., 5.])
    np.testing.assert_allclose(out['photon_origins'][1], [-5., 0., 5.])
    np.testing.assert_allclose(out['photon_directions'], raw['photon_directions'] @ R.T)
    np.testing.assert_array_equal(out['photon_times'], raw['photon_times'])
