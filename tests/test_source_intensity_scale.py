"""Tests for the QE-weighted auto source-intensity scale."""

import json

import numpy as np
import pytest

from compare_lucid_sim.compare_sim_vs_real import (
    auto_source_intensity_scale,
    write_amplitude_intensity_scale,
)
from compare_lucid_sim.laser_compare_metrics import simulation_qe_factors


def _source(kind, total, events=10):
    """Build a minimal source cache with a per-channel PE total."""
    return {
        "kind": np.asarray(kind),
        "label": np.asarray(kind),
        "simulation_total_pe": np.asarray([total], dtype=np.float64),
        "simulation_events": np.asarray([events], dtype=np.float64),
    }


def test_qe_factors_weight_only_lucid():
    """LUCiD carries the equivalent QE; JUNOsw stays 1."""
    sources = [_source("lucid", 100.0), _source("junosw", 5.0)]
    factors = simulation_qe_factors(sources, True, 0.097)
    np.testing.assert_allclose(factors, [0.097, 1.0])
    factors = simulation_qe_factors(sources, False, 0.097)
    np.testing.assert_allclose(factors, [1.0, 1.0])


def test_qe_factors_reject_invalid_qe():
    """A non-positive QE factor is rejected."""
    with pytest.raises(ValueError):
        simulation_qe_factors([_source("lucid", 1.0)], True, 0.0)


def test_auto_scale_uses_qe_weighted_totals():
    """The auto scale matches the amplitude figure's QE convention."""
    sources = [_source("lucid", 100.0), _source("junosw", 5.0)]
    factors = simulation_qe_factors(sources, True, 0.097)
    scales = auto_source_intensity_scale(sources, factors)
    np.testing.assert_allclose(scales, [5.0 / 9.7, 1.0])
    plain = auto_source_intensity_scale(sources)
    np.testing.assert_allclose(plain, [0.05, 1.0])


def test_write_amplitude_intensity_scale(tmp_path):
    """The JSON record carries run, reference and per-source scales."""
    sources = [_source("lucid", 100.0), _source("junosw", 5.0)]
    factors = np.asarray([0.097, 1.0])
    output = tmp_path / "amplitude_intensity_scale.json"
    write_amplitude_intensity_scale(
        output, 15040, sources, np.asarray([0.5, 1.0]), factors, 100, 30.0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run"] == 15040
    assert payload["reference_source"] == "junosw"
    assert payload["common_valid_pmt_count"] == 100
    assert payload["sources"][0]["qe_factor"] == pytest.approx(0.097)
    assert payload["sources"][0]["source_intensity_scale"] == 0.5
    assert payload["sources"][1]["source_intensity_scale"] == 1.0