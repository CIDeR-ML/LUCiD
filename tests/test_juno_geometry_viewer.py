"""Deterministic tests for the browser JUNO geometry payload."""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt

from lucid.geometry.juno_structures import (
    AcrylicBowlLayout,
    ChimneyGeometry,
)
from scripts.view_juno_geometry import build_geometry_payload


def _detector(chimney=None, bowls=None):
    """Return a minimal detector carrying optional engineering structures."""
    return SimpleNamespace(
        all_points=np.empty((0, 3), dtype=float),
        S_radius=0.254,
        support_rods=None,
        chimney=chimney,
        acrylic_bowls=bowls,
        r_inner=17.7,
        r_acrylic=17.824,
        r_outer=19.434,
    )


def test_viewer_payload_contains_hollow_chimney_segments():
    """The viewer receives both annular chimney segments in metres."""
    chimney = ChimneyGeometry.junosw_j26(outer_limit_m=19.434)

    manifest, payloads = build_geometry_payload(
        _detector(chimney=chimney)
    )

    assert payloads == {}
    assert manifest["chimney_segment_count"] == 2
    segments = manifest["chimney_segments"]
    assert [item["name"] for item in segments] == [
        "Chimney Acrylic",
        "Chimney steel",
    ]
    npt.assert_allclose(
        [
            [
                item["inner_radius"],
                item["outer_radius"],
                item["z_min"],
                item["z_max"],
            ]
            for item in segments
        ],
        [
            [0.400, 0.524, 17.824, 18.424],
            [0.400, 0.405, 18.424, 19.434],
        ],
    )


def test_viewer_payload_places_radial_bowl_frusta():
    """Bowl instances are centred halfway along their outward radial axes."""
    bowls = AcrylicBowlLayout(
        copy_numbers=np.array([12, 13], dtype=np.int32),
        attachment_points_m=np.array([
            [17.824, 0.0, 0.0],
            [0.0, -17.824, 0.0],
        ]),
        outward_axes=np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]),
        base_radii_m=np.array([0.45, 0.45]),
        tip_radii_m=np.array([0.20, 0.20]),
        lengths_m=np.array([0.14, 0.14]),
    )

    manifest, payloads = build_geometry_payload(
        _detector(bowls=bowls)
    )

    assert manifest["acrylic_bowl_count"] == 2
    [group] = manifest["acrylic_bowl_groups"]
    assert group["name"] == "Acrylic bowls"
    assert group["count"] == 2
    assert group["stride"] == 9
    npt.assert_allclose(group["tip_ratio"], 0.20 / 0.45)
    rows = np.frombuffer(
        payloads[group["url"]], dtype="<f4"
    ).reshape(group["count"], group["stride"])
    npt.assert_allclose(
        rows[:, :3],
        [
            [17.894, 0.0, 0.0],
            [0.0, -17.894, 0.0],
        ],
        atol=1e-6,
    )
    npt.assert_allclose(rows[:, 3:6], bowls.outward_axes)
    npt.assert_allclose(rows[:, 6], bowls.base_radii_m)
    npt.assert_allclose(rows[:, 7], bowls.lengths_m / 2.0)
    npt.assert_array_equal(rows[:, 8], bowls.copy_numbers)


def test_viewer_payload_uses_empty_structure_groups_when_disabled():
    """A detector without the components produces no viewer layers."""
    manifest, payloads = build_geometry_payload(_detector())

    assert manifest["chimney_segments"] == []
    assert manifest["chimney_segment_count"] == 0
    assert manifest["acrylic_bowl_groups"] == []
    assert manifest["acrylic_bowl_count"] == 0
    assert payloads == {}
