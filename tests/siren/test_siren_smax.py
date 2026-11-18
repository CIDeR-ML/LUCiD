"""``lucid.siren.emitter.make_smax_fn``: the one s_max(E) evaluator, for the forward model (jax) and the
table builder's report (numpy, float64)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.siren.emitter import make_smax_fn

LOW = dict(a=3.1, b1=1.02, b2=0.81, E0=450.0)
HIGH = dict(a_hi=2.4, b1_hi=1.05, b2_hi=0.9, E0_hi=900.0)
PIECEWISE = dict(form='piecewise', params={**LOW, **HIGH, 'e_join_mev': 1000.0})


def two_power(E, a, b1, b2, E0):
    return a * E ** b1 / (1.0 + (E / E0) ** (b1 - b2))


@pytest.mark.parametrize('E', [10.0, 999.0, 1000.0, 2500.0])
def test_each_form_is_its_formula(E):
    assert make_smax_fn(dict(form='A*E^B', params=dict(A=2.0, B=0.9)))(E) == 2.0 * E ** 0.9
    assert make_smax_fn(dict(form='smooth_two_power', params=LOW))(E) == two_power(E, **LOW)
    piece = LOW if E < 1000.0 else {k[:-3]: v for k, v in HIGH.items()}
    assert make_smax_fn(PIECEWISE, np)(E) == two_power(E, **piece)


def test_csv_string_params_read_as_floats():
    as_strings = dict(form='piecewise', params={k: repr(v) for k, v in PIECEWISE['params'].items()})
    E = np.array([500.0, 1500.0])
    np.testing.assert_array_equal(make_smax_fn(as_strings, np)(E), make_smax_fn(PIECEWISE, np)(E))


def test_jax_piecewise_is_traceable_and_agrees_with_float64():
    E = np.array([100.0, 999.0, 1000.0, 3000.0])
    got = jax.jit(make_smax_fn(PIECEWISE))(jnp.asarray(E, jnp.float32))
    assert got.dtype == jnp.float32
    np.testing.assert_allclose(got, make_smax_fn(PIECEWISE, np)(E), rtol=1e-6)


def test_unknown_form_is_refused():
    with pytest.raises(ValueError, match='unknown smax form'):
        make_smax_fn(dict(form='cubic', params={}))
