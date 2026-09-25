"""A learning-rate anneal must follow the DRIVER's iteration, not an optimiser's own counter.

`optax.scale_by_schedule` keeps the step count in its state. That is correct for a momentum
buffer and wrong for an anneal as soon as the driver can REJECT a step: `minimize` restores the
previous optimiser state on a rejection, which rewinds the counter, so the schedule spends the
same iteration twice and the anneal runs slow.

This is not hypothetical. The numpy step had exactly this defect; on a real reconstructed event
the rewind moved the fitted energy by 2.41 MeV. `scale_by_driver_schedule` fixes it by READING
`iteration` out of optax's extra arguments — which `minimize` supplies — instead of counting.

The tests below pin three things, and the middle one is the whole point:
  * the schedule tracks the driver's index when it is supplied;
  * a REJECTED step does not rewind it, which is what a stateful counter would do;
  * the internal counter still works when no driver supplies an index, so this composes with a
    plain optax loop that knows nothing about the convention.
"""
import numpy as np
import jax.numpy as jnp
import optax
import pytest

from lucid.fitting.transforms import scale_by_driver_schedule, annealed_learning_rate


def test_it_follows_the_iteration_it_is_given():
    """The basic contract: schedule evaluated at the driver's index, not at a call count."""
    tx = scale_by_driver_schedule(lambda s: float(s))
    st = tx.init(jnp.ones(2))
    # Call it in a deliberately scrambled order. A counter would return 0, 1, 2; reading the
    # supplied index must return 7, 3, 5.
    for it in (7, 3, 5):
        out, st = tx.update(jnp.ones(2), st, None, iteration=it)
        np.testing.assert_allclose(np.asarray(out), np.full(2, float(it)))


def test_a_repeated_iteration_does_not_advance_it():
    """THE DEFECT. A rejected step re-runs the same driver iteration; the anneal must not move.

    With a stateful counter the second call would return a LATER schedule value, which is the
    rewind-and-repeat that cost 2.41 MeV. Here the value is a function of the index alone, so
    calling twice at the same index is idempotent by construction.
    """
    tx = scale_by_driver_schedule(lambda s: 10.0 - s)
    st = tx.init(jnp.ones(1))
    first, st = tx.update(jnp.ones(1), st, None, iteration=4)
    again, st = tx.update(jnp.ones(1), st, None, iteration=4)
    np.testing.assert_allclose(np.asarray(first), np.asarray(again))
    assert float(np.asarray(first)[0]) == 6.0


def test_it_falls_back_to_counting_when_no_driver_supplies_an_index():
    """So it still composes with a plain optax loop rather than requiring this project's driver."""
    tx = scale_by_driver_schedule(lambda s: float(s))
    st = tx.init(jnp.ones(1))
    seen = []
    for _ in range(3):
        out, st = tx.update(jnp.ones(1), st, None)
        seen.append(float(np.asarray(out)[0]))
    assert seen == [0.0, 1.0, 2.0]


def test_it_is_stateless_in_the_sense_that_matters():
    """State exists only for the fallback counter; with an index supplied it cannot disagree.

    Asserted by feeding a FRESH state each call and requiring the same answer — if the schedule
    value depended on accumulated state, restoring an old state (which is what `minimize` does on
    a rejection) would change the result.
    """
    tx = scale_by_driver_schedule(lambda s: 2.0 ** s)
    fresh = tx.init(jnp.ones(1))
    a, _ = tx.update(jnp.ones(1), fresh, None, iteration=3)
    b, _ = tx.update(jnp.ones(1), tx.init(jnp.ones(1)), None, iteration=3)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------- the anneal itself

def test_the_anneal_ends_exactly_on_lr_final():
    """`transition_steps` must be steps-1, and this is where that bites.

    `optax.linear_schedule` divides by `transition_steps` while the loop being reproduced divides
    by `steps-1`. The obvious `linear_schedule(4.0, 1.5, steps)` ends at 1.5167, a 1.1% error on
    the final learning rate — four orders above the float32 agreement the module otherwise argues
    about, and invisible unless the endpoint is checked.
    """
    steps = 150
    sched = annealed_learning_rate(4.0, 1.5, steps)
    assert float(sched(0)) == pytest.approx(4.0)
    assert float(sched(steps - 1)) == pytest.approx(1.5, rel=1e-9)


def test_the_naive_transition_length_would_be_wrong():
    """The control on the test above: show the off-by-one actually produces the 1.5167 endpoint."""
    naive = optax.linear_schedule(4.0, 1.5, transition_steps=150)
    assert float(naive(149)) == pytest.approx(1.51667, rel=1e-4)
    assert float(naive(149)) != pytest.approx(1.5, rel=1e-6)


def test_the_anneal_does_not_run_past_its_end():
    """Unclamped, the ratio grows without bound and drives the rate NEGATIVE — descent to ascent.

    `optax.linear_schedule` holds `end_value` after the transition, so this asserts the property
    rather than an implementation; a hand-rolled `lr + (lr_final-lr)*step/(steps-1)` would fail it.
    """
    sched = annealed_learning_rate(4.0, 1.5, 10)
    for beyond in (10, 25, 1000):
        assert float(sched(beyond)) == pytest.approx(1.5)
        assert float(sched(beyond)) > 0


def test_no_final_rate_means_a_constant():
    """`lr_final=None` is the un-annealed configuration and must not depend on the iteration."""
    sched = annealed_learning_rate(4.0, None, 100)
    assert float(sched(0)) == 4.0
    assert float(sched(99)) == 4.0
