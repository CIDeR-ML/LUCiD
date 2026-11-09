"""The production template pairs a detector's geometry and physics files.

``run_job.py`` turns one ``--detector`` string into
``config/{detector}_{geom,physics}_config.json`` by f-string, and a dataset
config that declares no ``detector`` of its own takes its digitizer and trigger
from the detector's physics file; ``lucid-run-job`` and the fan-out refuse a
detector that carries neither, and only the frozen ``*_WAND`` descriptions
carry them. So ``SK_WAND`` and ``HK_WAND`` are string-dispatched and required:
neither is dead, and nothing said so.

The pair is loaded here the way production loads it --- both halves of one
detector, through the same template --- and not as two independent files.
"""
import json
import os

import pytest

from lucid.detector_params import load_physics_config
from tests.helpers.paths import REPO

CONFIG_DIR = os.path.join(
    str(REPO), "config")

# The production detectors: the only ones whose physics config carries a
# readout, so the only ones a config with no "detector" key can run on.
WAND_DETECTORS = ("SK_WAND", "HK_WAND")


def _template(detector):
    """The pair of paths run_job.py:277-278 builds from one --detector string."""
    return (os.path.join(CONFIG_DIR, f"{detector}_geom_config.json"),
            os.path.join(CONFIG_DIR, f"{detector}_physics_config.json"))


@pytest.mark.parametrize("detector", WAND_DETECTORS)
def test_wand_pair_exists_under_the_template(detector):
    """Both halves resolve. A missing half is a detector no job can select."""
    for path in _template(detector):
        assert os.path.isfile(path), f"{detector}: {os.path.basename(path)} is missing"


@pytest.mark.parametrize("detector", WAND_DETECTORS)
def test_wand_physics_carries_the_readout_the_fanout_relies_on(detector):
    """A ``*_WAND`` physics config supplies its own digitizer and trigger.

    A dataset config with no ``detector`` key takes both from the detector;
    ``resolve_readout_blocks`` refuses a detector that carries neither (it used
    to fall back to the 'basic' digitizer with the trigger off --- a run that
    looked successful and produced an unusable dataset).
    """
    _, physics_path = _template(detector)
    with open(physics_path) as f:
        raw = json.load(f)
    assert raw.get("digitizer"), f"{detector} physics config declares no digitizer"
    assert raw.get("trigger"), f"{detector} physics config declares no trigger"
    # ... and both blocks parse the way run_job parses them: a windowed model
    # (not the 'basic' fallback) and a trigger with no key TriggerConfig drops
    from lucid.simulation.readout import Readout
    readout = Readout.from_blocks(raw["digitizer"], raw["trigger"])
    model = readout.digitizer
    assert model["model"] != "basic" and model["integration_window_ns"], model["model"]
    assert readout.trigger is not None


@pytest.mark.parametrize("detector", WAND_DETECTORS)
@pytest.mark.slow
def test_wand_pair_loads_as_one_detector(detector):
    """Geometry and physics load together, and the per-PMT vector matches.

    The two files are only correct as a pair: ``qe_corrections`` is sized by
    the geometry's sensor count, and the loader is the only place that can
    notice if the two have drifted apart.
    """
    from lucid.geometry.detector_geometry import DetectorGeometry

    geom_path, physics_path = _template(detector)
    geom = DetectorGeometry.from_config(geom_path)
    n_sensors = len(geom.sensor_points)
    assert n_sensors > 0

    params, _, _ = load_physics_config(physics_path, num_sensors=n_sensors)
    qe_c = params.per_pmt.qe_corrections
    assert qe_c.ndim in (0, 1)
    if qe_c.ndim == 1:
        assert qe_c.shape == (n_sensors,), (
            f"{detector}: qe_corrections is {qe_c.shape}, geometry has {n_sensors} sensors")


def test_the_two_wand_detectors_disagree_about_surface_optics():
    """HK_WAND still carries the pre-2026-09-28 placeholder reflectivities.

    SK_WAND carries the measured 0.05 wall / 0.25 sensor; HK_WAND carries
    0.2/0.2, so the HK and SK legs of one WAND campaign do not share a surface
    model. This test asserts what is true today rather than what should be, so
    that fixing HK_WAND is a deliberate act with a number in its commit message
    and not a silent edit.
    """
    values = {}
    for detector in WAND_DETECTORS:
        with open(_template(detector)[1]) as f:
            raw = json.load(f)
        values[detector] = (raw.get("wall_reflection_rate"),
                            raw.get("sensor_reflection_rate"))
    assert values["SK_WAND"] == (0.05, 0.25), values["SK_WAND"]
    assert values["HK_WAND"] == (0.2, 0.2), (
        f"HK_WAND reflectivities moved to {values['HK_WAND']} — if that is the "
        f"measured pair, update this test in the same commit")
