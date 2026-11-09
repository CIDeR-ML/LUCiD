"""lucid.fitting.analysis: the track-error definition every recon study reports, on a case worked
by hand, and the bootstrap against its closed forms."""
import numpy as np

from lucid.fitting import bootstrap, bootstrap_ci, track_errors


def test_track_errors_on_a_hand_worked_case():
    """Truth: E 1000 MeV at (1, 2, 3) m along +z, t0 5 ns. Fit: E 990 MeV at (1.03, 2.04, 3) m,
    direction at theta = 30 deg (phi = 90 deg), t0 4.5 ns, with its sin/cos pairs unnormalised as a
    Gauss-Newton iterate leaves them. By hand: vertex sqrt(3^2 + 4^2) = 5 cm, direction 30 deg,
    dE -10 MeV, dt0 -0.5 ns (signed). A trajectory (truth, fit) gives the rows in order."""
    truth = np.array([1000., 1., 2., 3., 0., 1., 0., 1., 5.])
    fit = np.array([990., 1.03, 2.04, 3., 2 * 0.5, 2 * np.sqrt(3) / 2, 3., 0., 4.5])
    true_dir = np.array([0., 0., 2.])                      # any length: the angle normalises
    np.testing.assert_allclose(track_errors(fit, truth, true_dir), [5., 30., -10., -0.5],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(track_errors(np.stack([truth, fit]), truth, true_dir),
                               [[0., 0., 0., 0.], [5., 30., -10., -0.5]], rtol=0, atol=1e-12)


def test_bootstrap_resamples_with_replacement_uniformly():
    """A resample of n events misses the sample minimum with probability (1 - 1/n)^n, so the
    replicate minimum equals the sample minimum in 1 - (1 - 1/n)^n of resamples (0.636 at n = 50;
    20000 replicates measure it to 0.004). Replicates are reproducible from the seed alone."""
    x = np.random.default_rng(3).normal(size=50)
    reps = bootstrap(x, np.min, n_boot=20_000, seed=0)
    assert abs(np.mean(reps == x.min()) - (1 - (1 - 1 / 50) ** 50)) < 0.015
    np.testing.assert_array_equal(reps, bootstrap(x, np.min, n_boot=20_000, seed=0))
    assert not np.array_equal(reps, bootstrap(x, np.min, n_boot=20_000, seed=1))


def test_bootstrap_errors_match_the_closed_forms():
    """For the mean of n events the bootstrap standard error is s / sqrt(n) (s with ddof 0), and
    the central 68.27% percentile interval of a near-normal statistic spans +-1 standard error:
    for the median of n normal draws that is sqrt(pi / 2) / sqrt(n)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    se = bootstrap(x, np.mean, n_boot=4000, seed=0).std()
    np.testing.assert_allclose(se, x.std() / np.sqrt(len(x)), rtol=0.05)
    lo, hi = bootstrap_ci(x, np.mean, ci=68.27, n_boot=4000, seed=0)
    np.testing.assert_allclose([lo, hi], x.mean() + np.array([-1, 1]) * x.std() / np.sqrt(len(x)),
                               atol=0.1 * x.std() / np.sqrt(len(x)))
    y = rng.normal(size=2000)
    lo, hi = bootstrap_ci(y, np.median, ci=68.27, n_boot=4000, seed=0)
    np.testing.assert_allclose((hi - lo) / 2, np.sqrt(np.pi / 2 / len(y)), rtol=0.15)
    assert lo < np.median(y) < hi
