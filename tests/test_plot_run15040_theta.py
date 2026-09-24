"""Tests for compare/plot_run15040_theta.py normalization."""

import numpy as np
import pytest

from compare.plot_run15040_theta import (
    group_mean_sem,
    pmt_weighted_reference,
)


def test_weighted_reference_matches_global_per_pmt_mean():
    """The PMT-count-weighted reference is the global per-PMT mean."""
    theta = np.array([5.0, 5.0, 95.0])
    counts = np.array([1.0, 3.0, 8.0])

    mean, _, number = group_mean_sem(counts, theta)
    reference = pmt_weighted_reference(mean, number)

    # groups: [0,10) has 2 PMTs with mean 2; [90,100) has 1 PMT with 8
    assert reference == pytest.approx((1.0 + 3.0 + 8.0) / 3.0)
    assert reference != pytest.approx(np.nanmean(mean))  # flat = 5.0


def test_weighted_reference_ignores_empty_and_nan_groups():
    """Empty and NaN theta groups carry no weight."""
    theta = np.array([5.0, np.nan, 95.0])
    counts = np.array([2.0, 100.0, 4.0])

    mean, _, number = group_mean_sem(counts, theta)
    reference = pmt_weighted_reference(mean, number)

    assert reference == pytest.approx((2.0 + 4.0) / 2.0)


def test_weighted_reference_rejects_all_nan_input():
    """A curve without any finite group cannot be normalized."""
    with pytest.raises(ValueError, match="no finite theta group"):
        pmt_weighted_reference(
            np.array([np.nan, np.nan]), np.array([0, 0])
        )
