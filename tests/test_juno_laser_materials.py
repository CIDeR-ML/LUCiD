"""Regression tests for the JUNO 532 nm laser material baseline."""

import json
from pathlib import Path

import numpy.testing as npt

from lucid.detector_params import load_physics_config
from lucid.juno_laser.parameters import medium_lengths_at


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "config" / "materials"
PHYSICS = ROOT / "config" / "JUNO_laser_physics_config.json"
EXPECTED_LS_ABSORPTION_M = 85.26179771369235


def test_laser_material_table_is_finite_at_532_nm():
    """The unscaled material table remains readable and physical."""
    lengths = medium_lengths_at(
        {"ls": MATERIALS / "juno_laser_labls.json"},
        532.0,
    )

    assert lengths["ls"]["abs"] > 0.0
    assert lengths["ls"]["scat"] > 0.0
    assert lengths["ls"]["n"] > 1.0


def test_laser_ls_provenance_is_self_consistent():
    """Physics config and transported material must share one LS value."""
    with PHYSICS.open() as stream:
        physics = json.load(stream)

    assert physics["ls_optical_model"] == "old"
    assert physics["ls_absorption_table"] == "ABSLENGTH_v1"
    assert physics["ls_absorption_scale"] == "86.5/26.0"
    npt.assert_allclose(
        physics["absorption_length"],
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
