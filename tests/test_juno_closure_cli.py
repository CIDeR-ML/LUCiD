"""Deterministic contracts for the synthetic-closure orchestration."""

from types import SimpleNamespace

import numpy as np

from lucid.juno_laser.config import PARAMETER_NAMES
from lucid.juno_laser.eventwise import EventwisePeObservation
from lucid.juno_laser.irf import IrfKernelBank
from lucid.juno_laser.time_grid import TimeGrid
from scripts.run_juno_full_geometry_closure import (
    checkpoint_metadata,
    checkpoint_writer,
    evaluate_reference_objective_losses,
    optimization_stages,
    partition_source_indices,
    read_observation_fragments,
    write_observations,
)


def test_source_chunks_preserve_arbitrary_absolute_indices():
    """Chunking must not invent a fixed source manifest or renumber keys."""
    assert partition_source_indices((0, 3, 8, 11, 20), 2) == (
        (0, 3),
        (8, 11),
        (20,),
    )
    assert partition_source_indices((0, 3, 8), 0) == ((0, 3, 8),)


def test_joint_only_plan_has_one_all_parameter_stage():
    """The gradient qualification plan must optimize all five coordinates."""
    args = SimpleNamespace(
        optimization_plan="joint-only",
        joint_steps=17,
        stage1_steps=3,
        stage2_steps=4,
        stage3_steps=5,
    )
    assert optimization_stages(args) == (("joint", 17, PARAMETER_NAMES),)


def test_reference_losses_evaluate_truth_on_each_objective():
    """Truth losses are explicit train, validation and held-out values."""
    def objective(theta, keys):
        return np.sum((np.asarray(theta) - np.asarray(keys)) ** 2)

    losses = evaluate_reference_objective_losses(
        objective,
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 0.0]),
        heldout_objective=objective,
        heldout_keys=np.asarray([2.0, 2.0]),
    )
    assert losses == {
        "truth_train_loss_per_pe": 1.0,
        "truth_mc_validation_loss_per_pe": 5.0,
        "truth_heldout_loss_per_pe": 1.0,
    }


def test_checkpoint_metadata_matches_automatic_stage_selection():
    """The reported marker describes the checkpoint used downstream."""
    result = {
        "best_step": 3,
        "best_loss": 2.0,
        "best_coordinates": np.zeros(1),
        "best_mc_validation_step": 5,
        "best_mc_validation_loss": 1.8,
        "best_mc_validation_coordinates": np.ones(1),
        "best_heldout_validation_step": None,
        "best_heldout_validation_loss": None,
        "best_heldout_validation_coordinates": None,
    }
    assert checkpoint_metadata(result) == {
        "kind": "mc-validation-best",
        "step": 5,
        "loss_per_pe": 1.8,
    }


def test_staged_plan_remains_an_explicit_optional_strategy():
    """The historical three-stage plan remains available by request."""
    args = SimpleNamespace(
        optimization_plan="staged",
        joint_steps=17,
        stage1_steps=3,
        stage2_steps=4,
        stage3_steps=5,
    )
    stages = optimization_stages(args)
    assert [item[0] for item in stages] == [
        "ls_lengths",
        "diffuser_shape",
        "joint",
    ]
    assert [item[1] for item in stages] == [3, 4, 5]


def test_checkpoint_writer_atomically_replaces_recovery_snapshot(tmp_path):
    """Periodic snapshots retain coordinates required for a fresh restart."""
    write = checkpoint_writer(tmp_path, "joint")
    state = {
        "step": 4,
        "coordinates": np.arange(5, dtype=np.float32),
        "values": np.arange(5, dtype=np.float32) + 1.0,
        "loss": 2.5,
        "gradient": np.ones(5, dtype=np.float32),
        "moment1": np.ones(5, dtype=np.float32),
        "moment2": np.ones(5, dtype=np.float32),
        "best_coordinates": np.arange(5, dtype=np.float32),
        "best_loss": 2.0,
        "best_step": 3,
    }
    write(state)
    path = tmp_path / "checkpoint_joint.npz"
    assert path.is_file()
    with np.load(path) as saved:
        assert int(saved["step"]) == 4
        np.testing.assert_array_equal(
            saved["coordinates"], state["coordinates"]
        )


def test_observation_fragments_roundtrip_by_absolute_source_index(tmp_path):
    """Independent source files must reassemble without renumbering keys."""
    grid = TimeGrid(0.0, 4.0, 2.0)
    irf = IrfKernelBank(
        kernels=np.asarray([[0.25, 0.75], [0.6, 0.4]]),
        bin_width_ns=2.0,
    )
    args = SimpleNamespace(
        time_bin_ns=2.0,
        time_min_ns=0.0,
        time_max_ns=4.0,
        truncate_sigma=5.0,
        event_exposure=1.0,
        event_time_offset_ns=0.0,
        n_events=1,
        seed=42,
        data_n_photons=128,
        data_ensemble_size=1,
        data_importance_normalization="config",
        fit_n_photons=64,
        k=4,
        intensity=1000.0,
        optimization_plan="joint-only",
        background_rate_hz=0.0,
    )
    paths = []
    records = (
        (8, (0.0, 0.0, 17.1), 3.0),
        (3, (0.0, 0.0, 9.0), 1.0),
    )
    for source_index, position, pe_time in records:
        observation = EventwisePeObservation(
            event_ids=np.asarray([0], dtype=np.int32),
            pmt_ids=np.asarray([source_index % 2], dtype=np.int32),
            pe_times_ns=np.asarray([pe_time], dtype=np.float32),
            n_events=1,
            n_sensors=2,
            time_grid=grid,
        )
        prediction = np.full((2, 2), source_index, dtype=np.float32)
        path = tmp_path / f"source_{source_index}.h5"
        write_observations(
            path,
            (observation,),
            (prediction,),
            (position,),
            (source_index,),
            args,
            irf,
        )
        paths.append(path)
    positions = (records[1][1], records[0][1])
    observations, predictions = read_observation_fragments(
        paths,
        positions,
        (3, 8),
        grid,
        2,
        args,
        irf,
    )
    assert [item.pe_times_ns.tolist() for item in observations] == [
        [1.0],
        [3.0],
    ]
    assert [float(item[0, 0]) for item in predictions] == [3.0, 8.0]
