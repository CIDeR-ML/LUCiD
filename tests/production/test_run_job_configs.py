"""Phase-5 reconciliation — production subsystem present + well-formed.

A full PhotonSim→dataset run needs the external PhotonSim/GENIE binaries ($PHOTONSIM_BIN),
so this validates what runs without them: the lucid-run-job CLI parses and every shipped
dataprod config is well-formed. That every production module imports is
``tests/test_imports.py::test_import``, which walks all of ``lucid/`` including the
``__init__``-less ``lucid/production`` tree.
"""
import json
import os
import glob
import pytest
from tests.helpers.paths import REPO

ROOT = str(REPO)
CONFIGS = os.path.join(ROOT, 'lucid/production/dataprod/configs')


def test_run_job_cli_help():
    """argparse --help exits cleanly (no JIT / no PhotonSim)."""
    from lucid.production.dataprod.run_job import main
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0


def test_run_job_requires_a_detector(tmp_path, capsys):
    """No detector default: a job whose config declares no ``detector`` and that gets no
    ``--detector`` stops before PhotonSim, unless it is ``--skip-lucid`` (PhotonSim only, which
    simulates no detector). A config that declares one needs no flag."""
    from lucid.production.dataprod.run_job import EXIT_CONFIG, _parse_args, main
    argv = ['--config', os.path.join(CONFIGS, 'GeV/02_mu.json'),
            '--output-dir', str(tmp_path / 'out'), '--job-id', '1', '--skip-photonsim']
    assert main(argv) == EXIT_CONFIG
    assert "declares no 'detector'; pass --detector" in capsys.readouterr().err
    assert not (tmp_path / 'out').exists()
    assert _parse_args(argv + ['--skip-lucid']).detector is None
    # Solar declares HK: no flag, and the run gets past the detector and readout checks to the
    # missing PhotonSim output
    solar = ['--config', os.path.join(CONFIGS, 'Solar/01_e_low_energy.json'),
             '--output-dir', str(tmp_path / 'solar'), '--job-id', '1', '--skip-photonsim']
    assert main(solar) == 20
    assert 'output_job_000001.root does not exist' in capsys.readouterr().err


def test_dataprod_configs_well_formed():
    from lucid.production.dataprod.run_job import _load_config, _readout
    # configs live in block subdirectories (GeV/, SN/, Solar/, Test/) as NN_name.json.
    # GeV is trimmed to the production set (01-06); with SN/Solar/Test that's 9 blocks.
    files = sorted(glob.glob(os.path.join(CONFIGS, '*', '[0-9][0-9]_*.json')))
    assert len(files) >= 9, f"expected the block dataprod configs, found {len(files)}"
    for f in files:
        with open(f) as fh:
            cfg = json.load(fh)
        assert 'name' in cfg, f"{f}: missing 'name'"
        assert 'material' in cfg, f"{f}: missing 'material'"
        # every config drives a particle gun, GENIE, pile-up, a bomb, or a supernova burst
        assert any(k in cfg for k in ('particles', 'genie', 'vertices', 'bomb', 'supernova')), \
            f"{f}: no primary source (particles/genie/vertices/bomb/supernova)"
        # the config loads and its readout resolves as run_job does it, on the detector the
        # fan-out would pick (the config's own, else -D SK_WAND / HK_WAND), so a missing block,
        # a typo'd key or mode or primary_source fails here instead of silently defaulting
        _load_config(f)
        for detector in ([cfg['detector']] if 'detector' in cfg else ['SK_WAND', 'HK_WAND']):
            _readout(cfg, detector)


def test_unknown_selection_mode_raises():
    """A typo'd selection.mode silently reversed the selection it asked for.

    The fall-through default is None (real trigger) on the particles path and
    3 hits on the supernova path, so `{"mode": "triggered"}` ran the opposite
    of what the config said on one of the two.
    """
    from lucid.production.dataprod.run_job import _read_min_physics_hits

    assert _read_min_physics_hits({"selection": {"mode": "trigger"}},
                                  default=None) is None
    assert _read_min_physics_hits({"selection": {"mode": "min_physics_hits",
                                                 "n": 7}}, default=3) == 7
    # No selection block at all: the caller's default.
    assert _read_min_physics_hits({"supernova": {}}, default=3) == 3

    with pytest.raises(ValueError, match="selection.mode"):
        _read_min_physics_hits({"selection": {"mode": "triggered"}}, default=None)
    with pytest.raises(ValueError, match="unknown key"):
        _read_min_physics_hits({"selection": {"mode": "min_physics_hits",
                                              "hits": 3}}, default=None)
    # the retired supernova-block knob is refused, not ignored
    with pytest.raises(ValueError, match="no longer read"):
        _read_min_physics_hits({"supernova": {"min_physics_hits": 5}}, default=3)
