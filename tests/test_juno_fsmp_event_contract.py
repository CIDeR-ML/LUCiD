"""FSMP event/posterior-step and calibration CLI contract tests."""

from pathlib import Path
import re
import runpy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lucid.juno_laser.data import load_eventwise_pe_observation
from lucid.juno_laser.time_grid import TimeGrid


ROOT = Path(__file__).resolve().parents[1]


def _write_fsmp_table(path, *, omit_event20_step0=False):
    """Write two physical events with two posterior samples each."""
    rows = [
        (10, 0, 0, [1.0]),
        (10, 1, 0, []),
        (10, 0, 1, [2.0]),
        (10, 1, 1, []),
        (20, 0, 1, [3.0]),
        (20, 1, 1, []),
    ]
    if not omit_event20_step0:
        rows.extend(((20, 0, 0, []), (20, 1, 0, [])))
    table = pa.table(
        {
            "eid": pa.array([row[0] for row in rows], type=pa.int64()),
            "ch": pa.array([row[1] for row in rows], type=pa.int32()),
            "offset": pa.array(np.zeros(len(rows)), type=pa.float64()),
            "step": pa.array([row[2] for row in rows], type=pa.int32()),
            "PEt": pa.array(
                [row[3] for row in rows], type=pa.list_(pa.float64())
            ),
            "nu_lc": pa.array(np.ones(len(rows)), type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_fsmp_loader_selects_one_step_and_keeps_zero_pe_events(tmp_path):
    """Posterior samples are not counted as independent laser exposures."""
    path = tmp_path / "fsmp.parquet"
    _write_fsmp_table(path)
    observation = load_eventwise_pe_observation(
        path,
        n_sensors=2,
        time_grid=TimeGrid(0.0, 4.0, 1.0),
        posterior_step=0,
        event_exposure=np.asarray((1.5, 2.5)),
        event_time_offset_ns=np.asarray((0.1, -0.2)),
    )
    assert observation.n_events == 2
    assert observation.n_pe == 1
    assert observation.posterior_step == 0
    np.testing.assert_array_equal(observation.event_labels, (10, 20))
    np.testing.assert_array_equal(observation.event_ids, (0,))
    np.testing.assert_array_equal(observation.pe_times_ns, (1.0,))
    np.testing.assert_array_equal(observation.event_exposures, (1.5, 2.5))
    np.testing.assert_allclose(
        observation.event_time_offsets_ns, (0.1, -0.2)
    )


def test_fsmp_loader_rejects_missing_prescribed_step(tmp_path):
    """Every retained physical event must provide the selected state."""
    path = tmp_path / "fsmp.parquet"
    _write_fsmp_table(path, omit_event20_step0=True)
    with pytest.raises(ValueError, match="missing for physical event IDs"):
        load_eventwise_pe_observation(
            path,
            n_sensors=2,
            time_grid=TimeGrid(0.0, 4.0, 1.0),
            posterior_step=0,
            event_exposure=1.0,
        )


def test_fsmp_loader_requires_rows_that_expose_zero_pe_events(tmp_path):
    """Sparse PE-only tables cannot define the physical exposure ledger."""
    path = tmp_path / "fsmp.parquet"
    _write_fsmp_table(path)
    with pytest.raises(ValueError, match="one row per channel"):
        load_eventwise_pe_observation(
            path,
            n_sensors=3,
            time_grid=TimeGrid(0.0, 4.0, 1.0),
            posterior_step=0,
            event_exposure=1.0,
        )


def test_fsmp_loader_rejects_invalid_event_exposure(tmp_path):
    """Exposure is explicit and cannot be inferred from ``nu_lc``."""
    path = tmp_path / "fsmp.parquet"
    _write_fsmp_table(path)
    with pytest.raises(ValueError, match="event exposure"):
        load_eventwise_pe_observation(
            path,
            n_sensors=2,
            time_grid=TimeGrid(0.0, 4.0, 1.0),
            posterior_step=0,
            event_exposure=np.asarray((1.0,)),
        )


def test_fsmp_loader_rejects_invalid_event_time_offset(tmp_path):
    """Event timing metadata must follow the physical event ledger."""
    path = tmp_path / "fsmp.parquet"
    _write_fsmp_table(path)
    with pytest.raises(ValueError, match="event time offset"):
        load_eventwise_pe_observation(
            path,
            n_sensors=2,
            time_grid=TimeGrid(0.0, 4.0, 1.0),
            posterior_step=0,
            event_exposure=1.0,
            event_time_offset_ns=np.asarray((0.0,)),
        )


def test_calibration_job_only_passes_current_cli_options():
    """The Slurm command and Python parser must remain interface-compatible."""
    script = (
        ROOT / "run" / "run_juno_laser_calibrate_PEtime.sh"
    ).read_text()
    command = script.split(
        '"$PY" -u scripts/run_juno_laser_calibrate.py', maxsplit=1
    )[1]
    shell_options = set(re.findall(r"--[a-z0-9-]+", command))
    module = runpy.run_path(
        str(ROOT / "scripts" / "run_juno_laser_calibrate.py")
    )
    parser = module["build_parser"]()
    cli_options = set(parser._option_string_actions)
    assert shell_options <= cli_options
    assert "--prompt-max-ns" not in shell_options
    assert "--stage1-charge-weight" not in shell_options
