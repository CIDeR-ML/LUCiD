"""Acrylic transport checks using invented, replaceable optical data.

The optical values below are synthetic test inputs, not measurements or
values derived from a private detector table. The cover geometry remains SK-IV.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.propagation.sk_acrylic import (
    SK_ACRYLIC_THICKNESS,
    load_sk4_acrylic_optics,
    trace_sk4_acrylic_from_pmt,
    trace_sk4_acrylic_to_pmt,
)


@pytest.fixture
def optical_table(tmp_path):
    path = tmp_path / "synthetic_optics.json"
    path.write_text(json.dumps({
        "wavelength_nm": [350.0, 450.0, 550.0],
        "refractive_index": [1.6, 1.5, 1.4],
        "absorption_coefficient_per_m": [0.02, 0.04, 0.06],
    }))
    return path


@pytest.fixture
def optics(optical_table):
    return load_sk4_acrylic_optics(400.0, optical_table)


def _trace(x, optics):
    return trace_sk4_acrylic_to_pmt(
        jnp.array([x, 0.0, 1.0]),
        jnp.array([0.0, 0.0, -1.0]),
        jnp.zeros(3),
        jnp.array([0.0, 0.0, 1.0]),
        optics,
    )


def test_user_table_interpolates_optical_values(optics):
    assert float(optics.refractive_index) == pytest.approx(1.55)
    assert float(optics.absorption_coefficient) == pytest.approx(0.03)


def test_environment_table_is_used_when_no_path_is_given(optical_table, monkeypatch):
    monkeypatch.setenv("LUCID_SK4_ACRYLIC_TABLE", str(optical_table))
    result = load_sk4_acrylic_optics(400.0)
    assert float(result.refractive_index) == pytest.approx(1.55)


def test_explicit_replacement_overrides_environment(optical_table, tmp_path, monkeypatch):
    monkeypatch.setenv("LUCID_SK4_ACRYLIC_TABLE", str(optical_table))
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps({
        "wavelength_nm": [300.0, 600.0],
        "refractive_index": [1.7, 1.7],
        "absorption_coefficient_per_m": [0.1, 0.1],
    }))
    result = load_sk4_acrylic_optics(400.0, replacement)
    assert float(result.refractive_index) == pytest.approx(1.7)
    assert float(result.absorption_coefficient) == pytest.approx(0.1)


def test_missing_input_explains_how_to_supply_table(monkeypatch):
    monkeypatch.delenv("LUCID_SK4_ACRYLIC_TABLE", raising=False)
    with pytest.raises(FileNotFoundError, match="LUCID_SK4_ACRYLIC_TABLE"):
        load_sk4_acrylic_optics(400.0)


@pytest.mark.parametrize("key,values,message", [
    ("wavelength_nm", [350.0, 350.0, 550.0], "strictly increasing"),
    ("wavelength_nm", [550.0, 450.0, 350.0], "strictly increasing"),
    ("wavelength_nm", [-350.0, 450.0, 550.0], "positive"),
    ("refractive_index", [1.5], "equal lengths"),
    ("refractive_index", [[1.5, 1.5, 1.5]], "one-dimensional"),
    ("refractive_index", [1.5, float("nan"), 1.5], "finite"),
    ("refractive_index", [1.5, 0.0, 1.5], "positive"),
    ("absorption_coefficient_per_m", [0.0, -0.1, 0.0], "nonnegative"),
])
def test_invalid_tables_are_rejected(optical_table, key, values, message):
    data = json.loads(optical_table.read_text())
    data[key] = values
    optical_table.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_sk4_acrylic_optics(400.0, optical_table)


def test_outside_table_range_retains_comparison_fallback(optical_table):
    for wavelength in [300.0, 600.0]:
        result = load_sk4_acrylic_optics(wavelength, optical_table)
        assert float(result.refractive_index) == pytest.approx(1.49)
        assert float(result.absorption_coefficient) == 0.0


def test_normal_ray_keeps_pmt_geometry_and_gets_cover_loss_and_delay(optics):
    trace = _trace(0.0, optics)
    assert bool(trace.intersects) and bool(trace.active)
    assert float(trace.distance) == pytest.approx(0.812, abs=2e-6)
    npt.assert_allclose(trace.position, [0.0, 0.0, 0.188], atol=2e-6)
    npt.assert_allclose(trace.incident_direction, [0.0, 0.0, -1.0], atol=2e-6)
    n_water = float(optics.water_refractive_index)
    n_cover = float(optics.refractive_index)
    reflectance = ((n_water - n_cover) / (n_water + n_cover)) ** 2
    expected_transmission = (1.0 - reflectance) ** 2 * np.exp(
        -float(optics.absorption_coefficient) * SK_ACRYLIC_THICKNESS)
    assert float(trace.transmission) == pytest.approx(expected_transmission, rel=2e-6)
    expected_delay = SK_ACRYLIC_THICKNESS * (
        float(optics.refractive_index)
        / float(optics.water_group_refractive_index)
        - 1.0)
    assert float(trace.time_distance - trace.distance) == pytest.approx(
        expected_delay, rel=2e-5)


def test_normal_reflected_ray_exits_cover_with_matching_transmission(optics):
    incoming = _trace(0.0, optics)
    outgoing = trace_sk4_acrylic_from_pmt(
        incoming.position, jnp.array([0.0, 0.0, 1.0]),
        jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), optics)
    assert bool(outgoing.valid)
    npt.assert_allclose(outgoing.position, [0.0, 0.0, 0.219], atol=2e-6)
    npt.assert_allclose(outgoing.direction, [0.0, 0.0, 1.0], atol=2e-6)
    assert float(outgoing.transmission) == pytest.approx(
        float(incoming.transmission), rel=2e-5)


def test_first_order_reflection_branches_are_physical_and_conserve_weight(optics):
    trace = _trace(0.20, optics)
    reflected = trace.reflected
    assert bool(reflected.inner_exit_valid)
    assert 0.0 < float(reflected.outer_probability) < 1.0
    assert 0.0 < float(reflected.inner_exit_probability) < 1.0
    assert float(
        trace.transmission
        + reflected.outer_probability
        + reflected.inner_exit_probability
    ) <= 1.0 + 1e-6
    # Both retained branches have returned from the cover toward the ID.
    assert float(reflected.outer_direction[2]) > 0.0
    assert float(reflected.inner_exit_direction[2]) > 0.0
    assert float(reflected.inner_exit_time_distance) > float(
        reflected.outer_time_distance)


def test_ray_already_in_water_gap_does_not_cross_cover_again(optics):
    trace = trace_sk4_acrylic_to_pmt(
        jnp.array([0.0, 0.0, 0.205]),
        jnp.array([0.0, 0.0, -1.0]),
        jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), optics)
    assert bool(trace.intersects) and bool(trace.active)
    assert not bool(trace.cover_intersects)
    assert float(trace.transmission) == pytest.approx(1.0)
    assert float(trace.distance) == pytest.approx(0.017, abs=2e-6)


def test_back_half_of_acrylic_sphere_is_not_a_cover(optics):
    trace = trace_sk4_acrylic_to_pmt(
        jnp.array([1.0, 0.0, -0.20]),
        jnp.array([-1.0, 0.0, 0.0]),
        jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), optics)
    assert not bool(trace.cover_intersects)
    assert not bool(trace.intersects)


@pytest.mark.parametrize("x", [0.10, 0.20, 0.24])
def test_forward_and_reverse_derivatives_match(x, optics):

    def observable(offset):
        trace = trace_sk4_acrylic_to_pmt(
            jnp.array([offset, 0.01, 1.0]),
            jnp.array([0.01, 0.0, -1.0]),
            jnp.zeros(3),
            jnp.array([0.0, 0.0, 1.0]),
            optics,
        )
        return (trace.time_distance + trace.transmission
                + trace.reflected.outer_probability
                + trace.reflected.inner_exit_probability
                + 0.01 * jnp.sum(trace.reflected.outer_direction)
                + 0.01 * jnp.sum(trace.reflected.inner_exit_direction)
                + 0.01 * jnp.sum(trace.position)
                + 0.01 * jnp.sum(trace.incident_direction))

    tangent = jnp.asarray(0.3)
    forward = jax.jvp(
        observable, (jnp.asarray(x),), (tangent,))[1]
    gradient = jax.grad(observable)(jnp.asarray(x))
    reverse = gradient * tangent
    assert np.isfinite(float(forward)) and np.isfinite(float(reverse))
    assert float(forward) == pytest.approx(float(reverse), rel=3e-5, abs=3e-6)
