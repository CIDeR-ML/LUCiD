"""CLI tests for the standalone TTS-versus-IRF check script."""

from pathlib import Path

import pytest

from compare_lucid_sim import plot_tts_vs_irf


def test_cli_requires_irf_h5():
    """The check refuses to run without the measured IRF kernels."""
    with pytest.raises(SystemExit):
        plot_tts_vs_irf.parse_args(["--lucid-sim-dir", "/x"])


def test_cli_parses_check_flags(tmp_path):
    """The check inherits the pipeline CLI and adds the panel switch."""
    csv_path = tmp_path / "ch.csv"
    csv_path.write_text("channel\n3\n")
    args = plot_tts_vs_irf.parse_args(
        [
            "--lucid-sim-dir", "/x",
            "--irf-h5", "kernels.h5",
            "--no-time-difference-panels",
            "--channel-selection", str(csv_path),
        ]
    )
    assert args.irf_h5 == Path("kernels.h5")
    assert not args.time_difference_panels
    assert args.channel_selection == csv_path
    assert args.aligned_data.name == "aligned_data_ch.npz"


def test_cli_panels_default_on():
    """The difference panels stay enabled by default."""
    args = plot_tts_vs_irf.parse_args(
        ["--lucid-sim-dir", "/x", "--irf-h5", "kernels.h5"]
    )
    assert args.time_difference_panels