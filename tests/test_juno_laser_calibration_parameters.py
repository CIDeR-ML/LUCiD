"""Tests for the restricted JUNO LS calibration parameterization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.juno_laser.config import PARAMETER_NAMES
from lucid.juno_laser.optimizer import (
    adam_stage,
    make_ensemble_forward,
    select_stage_coordinates,
)
from lucid.juno_laser.parameters import (
    CalibParams,
    decode_calibration_vector,
    encode_calibration_values,
    unpack_params,
)
from lucid.juno_laser.source import (
    JUNO_LS_PHASE_COEFFICIENTS,
    LSPHASE_SHAPE_NAMES,
    lsphase_polynomial_coefficients,
    set_diffuser_angular_shape,
)
from lucid.sources.diffuser_ball import diffuser_ball_source


def _fixed_params():
    """Return a fixed optical baseline for unit tests."""
    return CalibParams(
        n_ls=jnp.asarray(1.487, dtype=jnp.float32),
        scat_ls=jnp.asarray(87.0, dtype=jnp.float32),
        abs_ls=jnp.asarray(85.262, dtype=jnp.float32),
        n_water=jnp.asarray(1.347, dtype=jnp.float32),
        scat_water=jnp.asarray(941.0, dtype=jnp.float32),
        abs_water=jnp.asarray(17.1, dtype=jnp.float32),
        amp=jnp.asarray(1.0, dtype=jnp.float32),
    )


BOUNDS = {
    "scat_ls": (20.0, 400.0),
    "abs_ls": (20.0, 1200.0),
    **{name: (-3.0, 3.0) for name in LSPHASE_SHAPE_NAMES},
}


def _coordinates(first=4.0, second=4.0, shapes=(0.0, 0.0, 0.0)):
    """Return one five-coordinate optimizer vector for tests."""
    return jnp.asarray((first, second, *shapes), dtype=jnp.float32)


def test_five_parameter_vector_round_trip():
    """Lengths use log coordinates while three shapes remain linear."""
    assert PARAMETER_NAMES == (
        "scat_ls", "abs_ls", *LSPHASE_SHAPE_NAMES
    )
    values = {
        "scat_ls": 92.0,
        "abs_ls": 84.0,
        "lsphase_shape_0": 0.75,
        "lsphase_shape_1": -0.25,
        "lsphase_shape_2": 0.5,
    }
    encoded = encode_calibration_values(values)
    decoded = decode_calibration_vector(encoded)
    np.testing.assert_allclose(
        [decoded[name] for name in PARAMETER_NAMES],
        [92.0, 84.0, 0.75, -0.25, 0.5],
        rtol=1e-6,
    )


def test_unpack_only_changes_ls_lengths():
    """Refractive indices and water properties stay at the fixed baseline."""
    fixed = _fixed_params()
    theta = encode_calibration_values({
        "scat_ls": 100.0,
        "abs_ls": 70.0,
        "lsphase_shape_0": 1.2,
        "lsphase_shape_1": -0.4,
        "lsphase_shape_2": 0.3,
    })
    params, shape = unpack_params(theta, fixed)
    np.testing.assert_allclose(params.scat_ls, 100.0, rtol=2e-7)
    np.testing.assert_allclose(params.abs_ls, 70.0, rtol=2e-7)
    np.testing.assert_allclose(params.n_ls, fixed.n_ls)
    np.testing.assert_allclose(params.n_water, fixed.n_water)
    np.testing.assert_allclose(params.scat_water, fixed.scat_water)
    np.testing.assert_allclose(params.abs_water, fixed.abs_water)
    np.testing.assert_allclose(shape, (1.2, -0.4, 0.3))


def test_zero_shape_coordinates_preserve_nominal_lsphase():
    """The three-coordinate origin is the configured JUNOSW shape."""
    source = diffuser_ball_source([0.0, 0.0, 0.0])
    nominal = set_diffuser_angular_shape(source, jnp.zeros(3))
    nominal_weights = np.asarray(jax.nn.softmax(nominal.angular_logits))
    np.testing.assert_allclose(
        nominal_weights,
        np.asarray(jax.nn.softmax(source.angular_logits)),
        rtol=1e-6,
    )


def test_lsphase_coordinates_have_three_independent_shape_directions():
    """Centered-logit coordinates remove exactly the common-scale zero mode."""
    source = diffuser_ball_source([0.0, 0.0, 0.0])

    def weights(coordinates):
        varied = set_diffuser_angular_shape(source, coordinates)
        return jax.nn.softmax(varied.angular_logits)

    jacobian = np.asarray(jax.jacrev(weights)(jnp.zeros(3)))
    assert np.linalg.matrix_rank(jacobian, tol=1.0e-6) == 3
    np.testing.assert_allclose(np.sum(jacobian, axis=0), 0.0, atol=1e-7)


def test_nominal_logits_report_original_junosw_polynomial():
    """Gauge-fixed reporting recovers the four J26.5.1 coefficients."""
    source = diffuser_ball_source([0.0, 0.0, 0.0])
    coefficients = lsphase_polynomial_coefficients(source.angular_logits)
    np.testing.assert_allclose(
        coefficients, JUNO_LS_PHASE_COEFFICIENTS, rtol=2e-6, atol=2e-7
    )


def test_ensemble_subset_preserves_absolute_source_random_keys():
    """Held-out source subsets must reuse their full-ensemble fold indices."""
    sources = tuple(
        diffuser_ball_source([float(index), 0.0, 0.0])
        for index in range(3)
    )

    def forward(source, _params, key):
        random_value = jax.random.uniform(key, (), dtype=jnp.float32)
        value = source.center[0] + random_value
        return jnp.reshape(value, (1, 1)), None

    full = make_ensemble_forward(
        forward,
        sources,
        (1, 1),
        fixed_params=_fixed_params(),
    )
    subset = make_ensemble_forward(
        forward,
        sources[1:],
        (1, 1),
        fixed_params=_fixed_params(),
        source_indices=(1, 2),
    )
    theta = _coordinates()
    keys = jax.random.split(jax.random.PRNGKey(17), 2)
    full_values = full(theta, keys)
    subset_values = subset(theta, keys)
    np.testing.assert_allclose(subset_values[0], full_values[1])
    np.testing.assert_allclose(subset_values[1], full_values[2])


def test_optimizer_rejects_nonfinite_gradient():
    """Invalid gradients must fail loudly instead of becoming silent zeros."""
    def objective(theta, keys):
        del keys
        return jnp.sqrt(theta[2])

    theta0 = encode_calibration_values({
        "scat_ls": 87.0,
        "abs_ls": 85.262,
        **{name: 0.0 for name in LSPHASE_SHAPE_NAMES},
    })
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        adam_stage(
            objective,
            theta0,
            PARAMETER_NAMES,
            ("lsphase_shape_0",),
            bounds=BOUNDS,
            train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
            steps=1,
            learning_rate=1e-3,
        )


def test_optimizer_preserves_mixed_coordinate_semantics():
    """Adam reports physical lengths and a linear angular coordinate."""
    target = encode_calibration_values({
        "scat_ls": 90.0,
        "abs_ls": 80.0,
        **{name: 0.5 for name in LSPHASE_SHAPE_NAMES},
    })

    def objective(theta, keys):
        del keys
        return jnp.sum((theta - target) ** 2)

    start = encode_calibration_values({
        "scat_ls": 87.0,
        "abs_ls": 85.262,
        **{name: 1.0 for name in LSPHASE_SHAPE_NAMES},
    })
    result = adam_stage(
        objective,
        start,
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=2,
        learning_rate=1e-3,
    )
    assert result["values"].shape == (5,)
    assert result["coordinates"].shape == (5,)
    np.testing.assert_allclose(
        result["values"][:2],
        np.exp(result["coordinates"][:2]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result["values"][2:], result["coordinates"][2:]
    )


def test_optimizer_accepts_host_accumulated_value_and_gradient():
    """One host-summed gradient produces one ordinary Adam update."""
    target = _coordinates(4.1, 4.2, (0.1, -0.2, 0.3))

    def objective(theta, keys):
        del keys
        return jnp.sum((theta - target) ** 2)

    def host_value_and_grad(theta, keys):
        value, gradient = jax.value_and_grad(objective)(theta, keys)
        return float(value), np.asarray(gradient)

    common = dict(
        theta0=_coordinates(),
        fields=PARAMETER_NAMES,
        active_fields=PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=2,
        learning_rate=1e-3,
    )
    fused = adam_stage(objective, **common)
    split = adam_stage(
        objective,
        value_and_grad_fn=host_value_and_grad,
        **common,
    )
    np.testing.assert_allclose(
        split["last_coordinates"], fused["last_coordinates"], rtol=1e-7
    )
    np.testing.assert_allclose(
        split["loss_history"], fused["loss_history"], rtol=1e-7
    )


def test_optimizer_reports_timings_and_periodic_checkpoints():
    """Long stages expose reproducible recovery snapshots and timings."""
    snapshots = []

    def objective(theta, keys):
        del keys
        return jnp.sum(theta**2)

    result = adam_stage(
        objective,
        _coordinates(),
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=3,
        learning_rate=1e-3,
        checkpoint_every=2,
        checkpoint_callback=snapshots.append,
    )
    assert [item["step"] for item in snapshots] == [2, 3]
    assert snapshots[-1]["coordinates"].shape == (5,)
    assert result["evaluation_seconds"].shape == (4,)
    assert np.all(result["evaluation_seconds"] >= 0.0)
    assert result["stage_seconds"] >= 0.0


def test_optimizer_histories_describe_the_same_evaluated_states():
    """Every stored loss must correspond to the coordinates at that index."""
    target = _coordinates(4.5, 4.4, (0.6, -0.2, 0.3))

    def objective(theta, keys):
        del keys
        return jnp.sum((theta - target) ** 2)

    result = adam_stage(
        objective,
        _coordinates(),
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=2,
        learning_rate=0.1,
    )
    coordinates = result["coordinate_history"]
    expected = np.sum((coordinates - np.asarray(target)) ** 2, axis=1)
    np.testing.assert_allclose(result["loss_history"], expected, rtol=1e-6)
    assert len(result["loss_history"]) == 3
    assert result["best_step"] == int(np.argmin(expected))
    np.testing.assert_allclose(
        result["coordinates"], coordinates[result["best_step"]]
    )
    np.testing.assert_allclose(result["last_coordinates"], coordinates[-1])
    np.testing.assert_allclose(result["last_loss"], expected[-1])


def test_true_train_best_is_recorded_without_validation():
    """Train-best remains available when early stopping is disabled."""
    target = _coordinates(4.5, 4.4, (0.6, -0.2, 0.3))

    def objective(theta, keys):
        del keys
        return jnp.sum((theta - target) ** 2)

    result = adam_stage(
        objective,
        _coordinates(),
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=1,
        learning_rate=0.01,
    )
    assert result["best_step"] == 1
    assert result["best_loss"] < result["loss_history"][0]
    assert result["patience_reference_loss"] is None


def test_stage_selection_prefers_mc_validation_including_step_zero():
    """Without held-out data, a stage hands off the MC-validation best."""
    start = _coordinates()
    train_target = _coordinates(4.5, 4.4, (0.6, -0.2, 0.3))

    def train_objective(theta, keys):
        del keys
        return jnp.sum((theta - train_target) ** 2)

    def mc_validation_objective(theta, keys):
        del keys
        return jnp.sum((theta - start) ** 2)

    result = adam_stage(
        train_objective,
        start,
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=2,
        learning_rate=0.1,
        mc_validation_objective=mc_validation_objective,
        mc_validation_keys=jnp.ones((1, 2), dtype=jnp.uint32),
        validation_every=1,
    )
    assert result["best_mc_validation_step"] == 0
    np.testing.assert_allclose(select_stage_coordinates(result), start)
    assert not np.allclose(result["last_coordinates"], start)


def test_stage_selection_falls_back_to_train_best_not_last():
    """A closure without validation must hand off the true train best."""
    result = {
        "coordinates": np.asarray((1.0, 2.0, 3.0)),
        "last_coordinates": np.asarray((4.0, 5.0, 6.0)),
        "best_mc_validation_coordinates": None,
        "best_heldout_validation_coordinates": None,
    }
    np.testing.assert_allclose(
        select_stage_coordinates(result), result["coordinates"]
    )


def test_early_stopping_counts_mc_validation_evaluations():
    """Improving train loss cannot reset MC-validation patience."""
    start = _coordinates()
    target = _coordinates(5.0, 5.0, (1.0, 0.5, -0.5))

    def train_objective(theta, keys):
        del keys
        return jnp.sum((theta - target) ** 2)

    def flat_mc_validation(theta, keys):
        del theta, keys
        return jnp.asarray(1.0)

    result = adam_stage(
        train_objective,
        start,
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=10,
        learning_rate=0.01,
        mc_validation_objective=flat_mc_validation,
        mc_validation_keys=jnp.ones((1, 2), dtype=jnp.uint32),
        validation_every=2,
        early_stopping_patience=2,
    )
    assert result["early_stopping_validation"] == "mc"
    assert result["validation_evaluations"] == 3
    assert result["stale_validation_evaluations"] == 2
    assert len(result["loss_history"]) == 5
    assert result["loss_history"][-1] < result["loss_history"][0]


def test_heldout_checkpoint_has_priority_over_mc_validation():
    """True held-out selection overrides same-data MC validation."""
    result = {
        "coordinates": np.asarray((1.0, 1.0, 1.0)),
        "best_mc_validation_coordinates": np.asarray((2.0, 2.0, 2.0)),
        "best_heldout_validation_coordinates": np.asarray((3.0, 3.0, 3.0)),
    }
    np.testing.assert_allclose(
        select_stage_coordinates(result), (3.0, 3.0, 3.0)
    )


def test_heldout_validation_controls_early_stopping_when_available():
    """Same-data MC improvement cannot override a flat held-out loss."""
    start = _coordinates()
    target = _coordinates(5.0, 5.0, (1.0, 0.5, -0.5))

    def improving_objective(theta, keys):
        del keys
        return jnp.sum((theta - target) ** 2)

    def flat_heldout(theta, keys):
        del theta, keys
        return jnp.asarray(1.0)

    result = adam_stage(
        improving_objective,
        start,
        PARAMETER_NAMES,
        PARAMETER_NAMES,
        bounds=BOUNDS,
        train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
        steps=5,
        learning_rate=0.01,
        mc_validation_objective=improving_objective,
        mc_validation_keys=jnp.ones((1, 2), dtype=jnp.uint32),
        heldout_validation_objective=flat_heldout,
        heldout_validation_keys=jnp.full((1, 2), 2, dtype=jnp.uint32),
        validation_every=1,
        early_stopping_patience=1,
    )
    assert result["early_stopping_validation"] == "heldout"
    assert result["best_heldout_validation_step"] == 0
    assert result["best_mc_validation_step"] == 1
    np.testing.assert_allclose(select_stage_coordinates(result), start)


def test_early_stopping_requires_validation():
    """Patience cannot silently fall back to the train objective."""
    with pytest.raises(ValueError, match="requires a validation objective"):
        adam_stage(
            lambda theta, keys: jnp.sum(theta**2),
            _coordinates(),
            PARAMETER_NAMES,
            PARAMETER_NAMES,
            bounds=BOUNDS,
            train_keys=jnp.zeros((1, 2), dtype=jnp.uint32),
            steps=2,
            learning_rate=0.01,
            early_stopping_patience=1,
        )
