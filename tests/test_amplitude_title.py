"""Tests for compare/laser_compare_plotting.py amplitude-page title."""

from compare.laser_compare_plotting import _amplitude_channel_title


def test_title_contains_dark_and_scale_lines():
    """The title lists the per-channel dark rate and source scales."""
    title = _amplitude_channel_title(
        93,
        "Hamamatsu",
        4.7913e-05,
        ["LUCiD (basis)", "JUNOsw (J26.3.2)"],
        [0.0151643, 1.0],
        "QE note",
    )
    lines = title.splitlines()
    assert lines[0] == ("Channel 93 | Hamamatsu | "
                        "PE/event timing spectrum")
    assert lines[1] == "Dark added to MC=4.7913e-05 PE/event/ns"
    assert lines[2] == ("Source intensity scale: "
                        "LUCiD (basis)=0.0151643, JUNOsw (J26.3.2)=1")
    assert lines[3] == "QE note"
    assert "Effective gates" not in title
    assert "prompt=" not in title


def test_title_formats_zero_and_unit_cleanly():
    """A unit scale renders as 1 and a zero dark rate renders as 0."""
    title = _amplitude_channel_title(
        1, "NNVT", 0.0, ["JUNOsw (J26.3.2)"], [1.0], ""
    )
    assert "Dark added to MC=0 PE/event/ns" in title
    assert "JUNOsw (J26.3.2)=1" in title