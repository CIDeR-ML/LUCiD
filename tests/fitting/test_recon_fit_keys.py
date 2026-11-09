"""Each event's fit draws from its own stream.

The fit averages its gradient and metric over ``nkeys`` keys split from the event's fit key,
``derive_event_keys(event_seed, job, event)['fit_key']``. Two properties follow and are checked
here on the analytic problem (no simulator): two events never share draws, so the same data at two
event indices fit to different key-averaged minima; and an event's fit depends on its index alone,
so it is bit-identical fitted alone or anywhere in any batch.
"""
import numpy as np

from tests.helpers.recon_analytic_model import AnalyticModel, _THETA_STAR, _U_START, _scale9

EVENT_SEED = 100003


def _fit_key(event):
    from lucid.sources.seed_utils import derive_event_keys
    return derive_event_keys(EVENT_SEED, 0, event)['fit_key']


def _fit_events(events):
    """Fit the same observables at each event index, in the given order."""
    from dataclasses import replace
    from lucid.fitting import FIT_RECIPE, fit_track
    fit = replace(FIT_RECIPE, nkeys=3, steps=20)
    model = AnalyticModel()
    start = _THETA_STAR + _scale9() * _U_START
    oc = ot = np.zeros(12)
    return {ev: np.asarray(fit_track(model, oc, ot, start, fit, key=_fit_key(ev))[0])
            for ev in events}


def test_no_two_events_share_a_fit_key():
    from lucid.fitting import FIT_RECIPE
    keys = [tuple(np.asarray(k).tolist()) for ev in range(20) for k in FIT_RECIPE.keys(_fit_key(ev))]
    assert len(set(keys)) == len(keys) == 20 * FIT_RECIPE.nkeys


def test_the_same_data_at_two_event_indices_fits_differently():
    """Shared keys would fit identical data identically; independent draws move the minimum."""
    fits = _fit_events([3, 5])
    assert np.max(np.abs(fits[3] - fits[5]) / _scale9()) > 1e-3


def test_an_event_fits_the_same_alone_or_in_any_batch():
    alone = _fit_events([3])[3]
    for batch in ([0, 1, 2, 3], [5, 3, 0], [3, 3]):
        np.testing.assert_array_equal(_fit_events(batch)[3], alone, err_msg=str(batch))
