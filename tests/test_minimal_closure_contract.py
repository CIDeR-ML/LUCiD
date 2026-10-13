"""Contract tests for the reviewable minimal JUNO closure."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

from lucid.juno_laser.config import JunoLaserConfig, PARAMETER_NAMES
from lucid.juno_laser.irf import make_identity_response_forward
from scripts.run_juno_full_geometry_closure import optimization_stages


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_juno_full_geometry_closure.py"


def test_only_two_positive_ls_parameters_are_fitted():
    """The single Adam stage fits only log scattering and absorption."""
    assert PARAMETER_NAMES == ("scat_ls", "abs_ls")
    args = SimpleNamespace(joint_steps=17)
    assert optimization_stages(args) == (
        ("joint", 17, ("scat_ls", "abs_ls")),
    )


def test_configuration_is_self_consistent():
    """The committed configuration specifies the ideal minimal source."""
    config = JunoLaserConfig.from_file(
        ROOT / "config" / "JUNO_laser_config.json"
    )
    assert config.data["source"]["model"] == "isotropic"
    assert config.calibration["parameters"] == list(PARAMETER_NAMES)


def test_identity_response_does_not_transform_the_forward():
    """No physical response kernel can enter the minimal forward."""
    forward = lambda theta, keys: (theta, keys)
    assert make_identity_response_forward(forward) is forward


def test_cli_does_not_expose_removed_model_controls():
    """Removed physics and nuisance controls are rejected by the CLI."""
    help_result = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = (
        "model-profile",
        "background-rate",
        "event-exposure",
        "event-time-offset",
        "importance-normalization",
        "optimization-plan",
        "stage1-steps",
        "heldout-source",
    )
    assert all(item not in help_result.stdout for item in forbidden)


def test_submit_minimal_five_dry_run_is_worktree_local():
    """The only submit target uses this checkout and fixed five sources."""
    result = subprocess.run(
        [
            "make",
            "-n",
            "-C",
            str(ROOT / "synthetic_closure"),
            "submit-minimal-five",
            "OUTDIR=/tmp/lucid-minimal-review",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert "0,0,0 0,0,9 0,0,15 0,0,16.1 0,0,17.1" in output
    assert str(ROOT / "run" / "run_juno_full_geometry_closure.sh") in output
    assert "MODEL_PROFILE" not in output
    assert "BACKGROUND_RATE" not in output
    assert "sbatch" in output
