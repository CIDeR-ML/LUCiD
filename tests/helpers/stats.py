"""Distribution checks shared by the sampler tests."""
import numpy as np


def ks_distance(samples, cdf):
    """Kolmogorov-Smirnov distance sup |F_n - F| between ``samples`` and the closed-form ``cdf``.

    Both one-sided gaps are taken at the jumps of the empirical CDF, so the value is exact, not a
    grid approximation. The 99.9 % critical value is about 1.95 / sqrt(n).
    """
    x = np.sort(np.asarray(samples, np.float64))
    F = cdf(x)
    n = len(x)
    return max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
