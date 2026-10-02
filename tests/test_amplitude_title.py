"""Tests for compare/laser_compare_plotting.py amplitude-page title."""

from compare_lucid_sim.laser_compare_plotting import (
    _amplitude_channel_title,
    short_source_label,
)


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
                        "LUCiD(basis)=0.0151643, JUNOsw=1")
    assert lines[3] == "QE note"
    assert "Effective gates" not in title
    assert "prompt=" not in title


def test_title_formats_zero_and_unit_cleanly():
    """A unit scale renders as 1 and a zero dark rate renders as 0."""
    title = _amplitude_channel_title(
        1, "NNVT", 0.0, ["JUNOsw (J26.3.2)"], [1.0], ""
    )
    assert "Dark added to MC=0 PE/event/ns" in title
    assert "JUNOsw=1" in title


def test_short_source_label_mapping():
    """Titles drop the JUNOsw version and compact LUCiD variants."""
    assert short_source_label("JUNOsw (J26.3.2)") == "JUNOsw"
    assert short_source_label("LUCiD (chimney off)") == "LUCiD(chimney off)"
    assert short_source_label("LUCiD") == "LUCiD"


def test_title_wraps_the_scale_line_for_multiple_lucid_sources():
    """With two LUCiD sources the scale list wraps to a second line."""
    title = _amplitude_channel_title(
        48,
        "NNVT",
        0.0,
        ["LUCiD (AllOn)", "LUCiD (chimney off)", "JUNOsw (J26.3.2)"],
        [0.15, 0.17, 1.0],
        "QE note",
    )
    lines = title.splitlines()
    assert "J26.3.2" not in title
    assert len(lines) == 5
    assert lines[2] == ("Source intensity scale: LUCiD(AllOn)=0.15, "
                        "LUCiD(chimney off)=0.17,")
    assert lines[3] == " " * 24 + "JUNOsw=1"
    assert lines[4] == "QE note"
    assert all(len(line) <= 68 for line in lines)
