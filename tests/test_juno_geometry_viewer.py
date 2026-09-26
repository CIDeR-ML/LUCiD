"""Deterministic tests for the browser JUNO geometry payload."""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt

from lucid.geometry.juno_structures import (
    AcrylicBowlLayout,
    ChimneyGeometry,
)
from lucid.geometry.juno_diffuser import DiffuserHardware
from scripts.view_juno_geometry import (
    build_geometry_payload,
    parse_vector3,
)


def _detector(chimney=None, bowls=None, diffuser_hardware=None):
    """Return a minimal detector carrying optional engineering structures."""
    return SimpleNamespace(
        all_points=np.empty((0, 3), dtype=float),
        S_radius=0.254,
        support_rods=None,
        chimney=chimney,
        acrylic_bowls=bowls,
        diffuser_hardware=diffuser_hardware,
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


def test_viewer_payload_places_radial_curved_bowls():
    """Viewer instances use the same curved profile as photon transport."""
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
    profile = np.asarray(group["profile"], dtype=float)
    assert profile.shape == (33, 2)
    npt.assert_allclose(profile[[0, -1], 0], [0.0, 1.0])
    npt.assert_allclose(
        profile[[0, 16, -1], 1],
        [0.450013840033, 0.298417111001, 0.200006485152],
        atol=1e-10,
    )
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
    npt.assert_allclose(rows[:, 6], 0.450013840033, atol=1e-7)
    npt.assert_allclose(rows[:, 7], bowls.lengths_m / 2.0)
    npt.assert_array_equal(rows[:, 8], bowls.copy_numbers)


def test_viewer_payload_uses_empty_structure_groups_when_disabled():
    """A detector without the components produces no viewer layers."""
    manifest, payloads = build_geometry_payload(_detector())

    assert manifest["chimney_segments"] == []
    assert manifest["chimney_segment_count"] == 0
    assert manifest["acrylic_bowl_groups"] == []
    assert manifest["acrylic_bowl_count"] == 0
    assert manifest["diffuser_hardware"] is None
    assert payloads == {}


def test_viewer_payload_places_source_attached_diffuser_hardware():
    """The viewer uses the runtime source pose and transport dimensions."""
    detector = _detector(
        diffuser_hardware=DiffuserHardware.junosw_j26_ls()
    )

    manifest, payloads = build_geometry_payload(
        detector,
        source_position=[1.0, -2.0, 17.1],
        source_axis=[0.0, 3.0, 4.0],
    )

    assert payloads == {}
    hardware = manifest["diffuser_hardware"]
    npt.assert_allclose(hardware["center"], [1.0, -2.0, 17.1])
    npt.assert_allclose(hardware["axis"], [0.0, 0.6, 0.8])
    assert hardware["parts"] == [
        {
            "name": "PTFE diffuser ball",
            "kind": "sphere",
            "color": "#f5f3e7",
            "radius": 0.010,
            "center_axial": 0.0,
        },
        {
            "name": "PEEK tube",
            "kind": "cylinder",
            "color": "#c9833e",
            "radius": 0.002,
            "half_length": 0.004,
            "center_axial": 0.010,
        },
        {
            "name": "PE fibre",
            "kind": "cylinder",
            "color": "#45b7b0",
            "radius": 0.0005,
            "half_length": 0.100,
            "center_axial": 0.110,
        },
    ]


def test_viewer_vector_parser_requires_three_finite_values():
    """Source pose CLI vectors use an explicit comma-separated contract."""
    npt.assert_allclose(parse_vector3("0,0,17.1"), [0.0, 0.0, 17.1])

    for invalid in ("0,0", "0,0,1,2", "0,nan,1", "zero,0,1"):
        try:
            parse_vector3(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid vector: {invalid}")
