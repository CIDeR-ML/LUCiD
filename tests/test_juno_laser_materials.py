"""Regression tests for the JUNO 532 nm laser material baseline."""

import json
from pathlib import Path

import numpy.testing as npt

from lucid.detector_params import load_physics_config
from scripts.run_juno_laser_sim import (
    _lengths_at,
    _ls_optical_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "config" / "materials"
PHYSICS = ROOT / "config" / "JUNO_laser_physics_config.json"
EXPECTED_LS_ABSORPTION_M = 85.26179771369235


def test_laser_ls_absorption_matches_junosw_reference_jobs():
    """The monochromatic baseline must match the old-model job logs."""
    lengths = _lengths_at(
        MATERIALS,
        532.0,
        ("juno_laser_labls.json",),
    )

    npt.assert_allclose(
        lengths[0][2], EXPECTED_LS_ABSORPTION_M, rtol=0.0, atol=1e-12
    )


def test_laser_ls_provenance_is_self_consistent():
    """Physics config and transported material must share one LS value."""
    with PHYSICS.open() as stream:
        physics = json.load(stream)

    provenance = _ls_optical_provenance(
        physics, EXPECTED_LS_ABSORPTION_M
    )

    assert provenance["ls_optical_model"] == "old"
    assert provenance["ls_absorption_table"] == "ABSLENGTH_v1"
    assert provenance["ls_absorption_scale"] == "86.5/26.0"
    npt.assert_allclose(
        provenance["ls_absorption_length_532_m"],
        EXPECTED_LS_ABSORPTION_M,
        rtol=0.0,
        atol=1e-12,
    )


def test_production_physics_config_loads_the_same_scalar():
    """The public config loader must accept and retain the new baseline."""
    params, _, _ = load_physics_config(
        str(PHYSICS),
        num_sensors=17612,
        scalar_ref_wavelength=532.0,
    )

    npt.assert_allclose(
        float(params.absorption.absorption_length),
        EXPECTED_LS_ABSORPTION_M,
        rtol=1e-7,
    )
