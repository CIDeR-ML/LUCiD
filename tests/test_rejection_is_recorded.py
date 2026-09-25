"""A refused step must survive all the way to the stored record.

This exists because it did not. `n_rejected` was added to `minimize` and to `fit_track`, and a
100-event GPU campaign then reported "NOT RECORDED" for every event: the count was written into a
record builder the published config does not use, and the live one dropped it. Thirty-three
minutes on ten GPUs to discover a plumbing gap that costs nothing to test.

The chain is: `minimize` counts -> `fit_track` forwards -> the pipeline record carries ->
`_ATTR_KEYS` routes it to an HDF5 attribute -> the aggregator reads `attrs`. Every link is cheap
to check and the whole chain is worthless if any one is missing.
"""
import numpy as np
import pytest

from lucid.fitting.minimize import minimize
from lucid.fitting.transforms import damped_gauss_newton

P, N = 4, 8
TRUTH = np.array([1.0, 2.0, -1.0, 0.5])


class Prob:
    """Deterministic least squares that poisons the gradient at a chosen call."""

    def __init__(self, blow_at=None):
        self.A = np.random.default_rng(3).standard_normal((N, P))
        self.blow_at, self.n = blow_at, 0

    def grad_metric_loss(self, theta, step, refresh=True):
        self.n += 1
        r = self.A @ (np.asarray(theta, float) - TRUTH)
        g = self.A.T @ r
        if self.blow_at is not None and self.n == self.blow_at:
            g = g * np.nan
        return g, self.A.T @ self.A, float(r @ r)

    def accumulate(self, theta, dtheta):
        return np.asarray(theta) + np.asarray(dtheta)


def _run(blow_at):
    return minimize(Prob(blow_at=blow_at), TRUTH + 0.7, 8,
                    damped_gauss_newton(0.01, 0.1),
                    needs_metric=True, reject_nonfinite=True)


def test_minimize_counts_a_refused_step():
    assert _run(blow_at=4)['n_rejected'] >= 1


def test_it_reports_zero_when_nothing_is_refused():
    """The control. A counter that always fires is as useless as one that never does."""
    res = _run(blow_at=None)
    assert res['n_rejected'] == 0
    assert res['rejected_steps'] == ()


def test_the_refused_step_index_is_recorded():
    res = _run(blow_at=4)
    assert len(res['rejected_steps']) == res['n_rejected']
    assert all(isinstance(i, int) for i in res['rejected_steps'])


@pytest.mark.parametrize('module', ['run', 'run_study'])
def test_the_hdf5_writer_routes_it_to_an_ATTRIBUTE(module):
    """`_ATTR_KEYS` is a whitelist, and a scalar missing from it lands as a DATASET.

    Every reader here uses `attrs`, so an unlisted scalar reads as ABSENT rather than raising —
    which is how a run with no instrumentation looks identical to a run with no rejections. Both
    writers keep their own whitelist: `run` (local) and `run_study` (the SLURM worker).
    """
    import importlib
    keys = importlib.import_module(f'analysis.paper.utils.{module}')._ATTR_KEYS
    assert 'n_rejected' in keys


def test_the_published_record_builder_carries_it():
    """Guards the specific gap that cost a campaign: the LIVE builder must include the key.

    Checked on the source rather than by running the pipeline, which needs a detector, a ROOT file
    and a GPU. The point is that the key appears in the record the published config writes.
    """
    import inspect
    from analysis.paper.utils import pipeline
    src = inspect.getsource(pipeline)
    # The live builder is the one that also writes `best_iter_win`; there is more than one
    # builder in this module and only that one runs for the published configuration.
    live = [b for b in src.split('rec.update(') if 'best_iter_win' in b]
    assert live, 'could not locate the record builder that writes best_iter_win'
    assert any('n_rejected' in b for b in live), \
        'the live record builder does not carry n_rejected — a campaign will report NOT RECORDED'


def test_a_NONZERO_count_survives_a_real_hdf5_round_trip(tmp_path):
    """The link the rest of this file asserts about but never exercises.

    `test_the_hdf5_writer_routes_it_to_an_ATTRIBUTE` checks that `n_rejected` is IN `_ATTR_KEYS`,
    and `test_the_published_record_builder_carries_it` reads the builder's source text. Neither
    calls `_write_event`, so the dispatch itself — the conditional deciding attribute versus
    dataset — was never executed by any test. A break there would leave both of them passing.

    The value written here is deliberately NONZERO. Every artifact this project has round-tripped
    so far records `n_rejected = 0` (measured: 0 of 100 published events ever refuse a step), so a
    test built on the real data would have confirmed nothing — zero is also what an absent key
    reads back as once a caller applies a default. Only a nonzero value distinguishes "written and
    read back" from "never written at all", which is precisely the failure that cost the campaign.
    """
    import h5py
    from analysis.paper.utils.run import _write_event

    rec = {'ev': 7, 'n_rejected': 3, 'n_hit': 624,
           'fit_err': np.array([1.6, 1.01, -22.7, -0.02])}
    path = tmp_path / 'rt.h5'
    with h5py.File(path, 'w') as h:
        _write_event(h.create_group('ev0007'), rec)

    with h5py.File(path, 'r') as h:
        g = h['ev0007']
        assert 'n_rejected' in g.attrs, 'n_rejected did not reach the file as an attribute'
        assert int(g.attrs['n_rejected']) == 3, (
            f"n_rejected round-tripped as {g.attrs['n_rejected']!r}, not the 3 that was written")
        # The other half of the dispatch: a non-whitelisted key must land as a DATASET. Without
        # this, a writer that made EVERYTHING an attribute would satisfy the assertion above.
        assert 'fit_err' in g and 'fit_err' not in g.attrs, \
            'fit_err should be a dataset, not an attribute — the whitelist is not discriminating'
        np.testing.assert_allclose(np.asarray(g['fit_err']), rec['fit_err'])
