"""How a dataset config resolves its detector, readout and primary source (``lucid-run-job`` and
the fan-out share the rules).

Before, a missing ``digitizer`` / ``trigger`` block gave the ``basic`` digitizer and no trigger,
an unreadable physics config was swallowed, a config's ``detector`` silently replaced
``--detector``, and a mistyped ``primary_source`` ran as a particle gun.
"""
import json
from pathlib import Path

import pytest

from lucid.production.dataprod import fanout, run_job
from tests.helpers.paths import REPO

CONFIGS = Path(REPO) / "lucid/production/dataprod/configs"
SOLAR = CONFIGS / "Solar/01_e_low_energy.json"


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def test_readout_blocks_come_from_the_dataset_config_then_the_detector(tmp_path):
    physics = _write(tmp_path / "p.json", {"digitizer": {"model": "ski"},
                                           "trigger": {"window_ns": 200.0, "n_thr": 25}})
    assert run_job.resolve_readout_blocks({}, physics) == (
        {"model": "ski"}, {"window_ns": 200.0, "n_thr": 25})
    # a key in the dataset config wins, null included (the trigger opt-out)
    assert run_job.resolve_readout_blocks(
        {"digitizer": "basic", "trigger": None}, physics) == ("basic", None)
    # with both keys set the physics config is not read at all
    assert run_job.resolve_readout_blocks(
        {"digitizer": {"model": "hk"}, "trigger": None}, str(tmp_path / "absent.json")) == (
        {"model": "hk"}, None)


def test_a_missing_readout_block_raises_instead_of_running_basic(tmp_path):
    bare = _write(tmp_path / "bare.json", {})
    with pytest.raises(ValueError, match="no 'digitizer' block"):
        run_job.resolve_readout_blocks({"trigger": None}, bare)
    with pytest.raises(ValueError, match="no 'trigger' block"):
        run_job.resolve_readout_blocks({"digitizer": "basic"}, bare)
    with pytest.raises(ValueError, match='"digitizer": null names no model'):
        run_job.resolve_readout_blocks({"digitizer": None, "trigger": None}, bare)
    # a broken physics config is an error, not "no block"
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(ValueError):
        run_job.resolve_readout_blocks({}, str(broken))
    with pytest.raises(OSError):
        run_job.resolve_readout_blocks({}, str(tmp_path / "absent.json"))


@pytest.mark.parametrize("detector", ["SK_like", "HK"])
def test_a_generic_detector_supplies_no_readout(detector):
    """The generic physics configs carry no digitizer or trigger: a GeV config on one raises."""
    cfg = json.loads((CONFIGS / "GeV/02_mu.json").read_text())
    with pytest.raises(ValueError, match="no 'digitizer' block"):
        run_job.resolve_readout_blocks(cfg, run_job.physics_config_path(detector))


def _main(tmp_path, cfg, detector):
    return run_job.main(["--config", _write(tmp_path / "cfg.json", cfg), "--detector", detector,
                         "--output-dir", str(tmp_path / "out"), "--job-id", "1",
                         "--skip-photonsim", "--skip-genie"])


def test_run_job_refuses_a_config_without_readout_before_photonsim(tmp_path, capsys):
    cfg = {"name": "t", "material": "water", "particles": [{"pdg": 13}],
           "energy_distribution": "uniform"}
    assert _main(tmp_path, cfg, "SK_like") == run_job.EXIT_CONFIG
    assert "no 'digitizer' block" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_run_job_refuses_a_detector_other_than_the_one_the_config_declares(tmp_path, capsys):
    cfg = json.loads(SOLAR.read_text())
    assert cfg["detector"] == "HK"
    assert _main(tmp_path, cfg, "SK_WAND") == run_job.EXIT_CONFIG
    assert "declares detector 'HK', but --detector is 'SK_WAND'" in capsys.readouterr().err


def test_supernova_needs_its_selection_stated(tmp_path, capsys):
    cfg = {"name": "t", "material": "water", "primary_source": "supernova",
           "supernova": {"orderings": ["NMO"]}, "digitizer": {"model": "hk"}, "trigger": None}
    assert _main(tmp_path, cfg, "HK") == run_job.EXIT_CONFIG
    assert "needs {\"selection\"" in capsys.readouterr().err
    with pytest.raises(ValueError, match='needs "n"'):
        run_job._read_min_physics_hits({"selection": {"mode": "min_physics_hits"}}, default=3)


def test_primary_source_is_one_of_the_known_sources(tmp_path):
    base = {"name": "t", "material": "water", "particles": [{"pdg": 13}],
            "energy_distribution": "uniform"}
    for src in (None, "particles"):
        cfg = dict(base) if src is None else dict(base, primary_source=src)
        assert run_job.primary_source(run_job._load_config(_write(tmp_path / "a.json", cfg))) \
            == "particles"
    with pytest.raises(ValueError, match="primary_source is 'bom'"):
        run_job._load_config(_write(tmp_path / "b.json", dict(base, primary_source="bom")))
    vertex = {"particles": [{"pdg": 13}], "energy_distribution": "uniform"}
    pile = {"name": "t", "material": "water", "pile_up": True,
            "vertices": [vertex, dict(vertex, primary_source="supernova")]}
    with pytest.raises(ValueError, match="Pile-up vertex 1: primary_source is 'supernova'"):
        run_job._load_config(_write(tmp_path / "c.json", pile))


def _user_paths(tmp_path):
    image = tmp_path / "lucid.sif"
    image.touch()
    up = tmp_path / "user_paths.sh"
    up.write_text(f'export CLUSTER="slurm"\nexport LUCID_IMAGE_PATH="{image}"\n'
                  f'export OUTPUT_BASE_PATH="{tmp_path / "prod"}"\n'
                  'export SLURM_PARTITION="milano"\nexport SLURM_ACCOUNT="a:b"\n'
                  'export APPTAINER_BINDS="/sdf"\n')
    return str(up)


def test_fanout_passes_a_self_contained_config_its_own_detector(tmp_path, capsys):
    """``-D SK_WAND`` over every block (``submit_all_configs.sh``): the HK config keeps HK, so
    the job's ``--detector`` matches the config and run_job's mismatch check does not trip."""
    rc = fanout.main(["-c", str(SOLAR), "-D", "SK_WAND", "-t",
                      "--user-paths", _user_paths(tmp_path)])
    assert rc == 0
    subs = list((tmp_path / "prod").rglob("submit_job_*"))
    assert subs and all("--detector HK " in p.read_text() for p in subs)
    assert all("/HK/" in str(p) for p in subs)


def test_fanout_refuses_a_config_whose_readout_does_not_resolve(tmp_path, capsys):
    rc = fanout.main(["-c", str(CONFIGS / "GeV/02_mu.json"), "-D", "SK_like", "-t",
                      "--user-paths", _user_paths(tmp_path)])
    assert rc == 1
    assert "no 'digitizer' block" in capsys.readouterr().err
    assert fanout.main(["-c", str(CONFIGS / "GeV/02_mu.json"), "-D", "SK_WAND", "-t",
                        "--user-paths", _user_paths(tmp_path)]) == 0
