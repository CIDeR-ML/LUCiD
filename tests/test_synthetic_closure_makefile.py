'''Contract tests for the synthetic-closure submission Makefile.'''

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKE_DIR = PROJECT_ROOT / "synthetic_closure"


def _dry_run(target: str, *variables: str) -> str:
    '''Return the command expanded by a non-executing Make invocation.'''
    command = [
        "make",
        "-n",
        "-C",
        str(MAKE_DIR),
        target,
        "OUTDIR=/tmp/lucid-closure-contract",
        *variables,
    ]
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_submit_one_uses_group_account_and_worker() -> None:
    '''The one-source target must use the shared account and worker script.'''
    output = _dry_run("submit-one")
    assert "sudo -u#35905 sbatch" in output
    assert "SOURCES='0,0,17.1'" in output
    assert "N_EVENTS='100'" in output
    assert "JOINT_STEPS='300'" in output
    assert "AD_FD_RELATIVE_STEP='0.001'" in output
    assert "CLOSURE_MODE='full'" in output
    assert "run/run_juno_full_geometry_closure.sh" in output


def test_submit_five_preserves_requested_nested_sources() -> None:
    '''The five-source target must preserve all source coordinates.'''
    output = _dry_run("submit-five")
    sources = "0,0,0 0,0,9 0,0,15 0,0,16.1 0,0,17.1"
    assert f"SOURCES='{sources}'" in output
    assert "DATA_N_PHOTONS='400000'" in output
    assert "FIT_N_PHOTONS='100000'" in output
    assert "K='32'" in output
    assert "OBJECTIVE_EXECUTION='per-source'" in output
    assert "N_EVENTS='100'" in output
    assert "JOINT_STEPS='300'" in output
    assert "AD_FD_RELATIVE_STEP='0.001'" in output


def test_closure_worker_disables_xla_command_buffer() -> None:
    '''The closure opt-in must append the ROCm command-buffer workaround.'''
    command = [
        "bash",
        "-c",
        (
            "unset XLA_FLAGS; "
            "export LUCID_DISABLE_XLA_COMMAND_BUFFER=1; "
            "source run/env.sh; printf '%s' \"$XLA_FLAGS\""
        ),
    ]
    output = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--xla_gpu_enable_command_buffer=" in output


def test_closure_worker_pins_python_313_before_environment() -> None:
    '''The GPU worker must not inherit an ambiguous generic Python.'''
    worker = (
        PROJECT_ROOT / "run" / "run_juno_full_geometry_closure.sh"
    ).read_text()
    pin = "CLOSURE_PYTHON:-/opt/gentoo/usr/bin/python3.13"
    assert pin in worker
    assert worker.index(pin) < worker.index('source "$WORK/run/env.sh"')


def test_worker_uses_process_isolated_generation_and_fit() -> None:
    '''Each data source and the fit must execute in separate processes.'''
    worker = (
        PROJECT_ROOT / "run" / "run_juno_full_geometry_closure.sh"
    ).read_text()
    assert 'for index in "${!SOURCE_ITEMS[@]}"' in worker
    assert "--phase generate" in worker
    assert "--phase fit" in worker
    assert "--observation-input" in worker
    assert "--source-index" in worker
    assert worker.count(
        '"$LUCID_PY" -u scripts/run_juno_full_geometry_closure.py'
    ) == 2


def test_submit_fit_only_reuses_existing_observation_fragments() -> None:
    '''Fit-only diagnostics must not regenerate expensive synthetic data.'''
    observation_root = "/tmp/lucid-existing-observations"
    output = _dry_run(
        "submit-fit-only",
        f"OBSERVATION_ROOT={observation_root}",
    )
    assert "CLOSURE_MODE='fit-only'" in output
    assert f"OBSERVATION_ROOT='{observation_root}'" in output
    assert "FIT_ENSEMBLE='1'" in output
    assert "N_EVENTS='2'" in output
    assert "JOINT_STEPS='128'" in output


def test_submit_five_fit_only_uses_all_sources_and_split_objective() -> None:
    '''The five-source fit-only target reuses data without GPU fusion.'''
    observation_root = "/tmp/lucid-five-source-observations"
    output = _dry_run(
        "submit-five-fit-only",
        f"OBSERVATION_ROOT={observation_root}",
    )
    sources = "0,0,0 0,0,9 0,0,15 0,0,16.1 0,0,17.1"
    assert f"SOURCES='{sources}'" in output
    assert "CLOSURE_MODE='fit-only'" in output
    assert "OBJECTIVE_EXECUTION='per-source'" in output
    assert "N_EVENTS='2'" in output
    assert "JOINT_STEPS='128'" in output


def test_worker_renders_closure_diagnostics_after_success() -> None:
    '''A completed closure must automatically produce its diagnostic PDF.'''
    worker = (
        PROJECT_ROOT / "run" / "run_juno_full_geometry_closure.sh"
    ).read_text()
    assert "scripts/plot_juno_closure_result.py" in worker
    assert "closure_diagnostics.pdf" in worker


def test_submit_one_fisher_uses_formal_transport_configuration() -> None:
    '''The one-source Fisher job uses the formal optical baseline.'''
    output = _dry_run(
        "submit-one-fisher",
        "FISHER_OUTDIR=/tmp/lucid-fisher-contract",
    )
    assert "SOURCES='0,0,17.1'" in output
    assert "N_PHOTONS='100000'" in output
    assert "N_EVENTS='100'" in output
    assert "K='32'" in output
    assert "run_juno_synthetic_fisher.sh" in output
