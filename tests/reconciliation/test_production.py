"""Phase-5 reconciliation — production subsystem present + well-formed.

A full PhotonSim→v3 run needs the external PhotonSim/GENIE binaries ($PHOTONSIM_BIN),
so this validates what runs without them: the entry point + helpers import, the
lucid-run-job CLI parses, and every shipped dataprod config is well-formed.
"""
import json
import os
import glob
import pytest

ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'
CONFIGS = os.path.join(ROOT, 'lucid/production/configs')


def test_production_modules_import():
    import importlib
    for m in ('run_job', 'generate_macro', 'run_genie', 'verify_output'):
        importlib.import_module(f'lucid.production.{m}')
    for m in ('cluster', 'htcondor', 'nersc', 'verify', 'user_paths', 'dataprod_fanout'):
        importlib.import_module(f'lucid.production.cluster_common.{m}')
    from lucid.production.run_job import main
    assert callable(main)


def test_run_job_cli_help():
    """argparse --help exits cleanly (no JIT / no PhotonSim)."""
    from lucid.production.run_job import main
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0


def test_dataprod_configs_well_formed():
    files = sorted(glob.glob(os.path.join(CONFIGS, 'dataprod_*.json')))
    assert len(files) >= 18, f"expected the dataprod_01..18 configs, found {len(files)}"
    for f in files:
        with open(f) as fh:
            cfg = json.load(fh)
        assert 'name' in cfg, f"{f}: missing 'name'"
        assert 'material' in cfg, f"{f}: missing 'material'"
        # every config drives either particle-gun, GENIE, or pile-up
        assert any(k in cfg for k in ('particles', 'genie', 'vertices', 'bombs')), \
            f"{f}: no primary source (particles/genie/vertices/bombs)"


def test_run_job_uses_modular_event_generation():
    """run_job must pull the data-gen fns from event_generation/seed_utils, NOT the
    recon event_io (which keeps its own generate_events_* + pad_photon_data)."""
    src = open(os.path.join(ROOT, 'lucid/production/run_job.py')).read()
    assert 'from lucid.sources.event_generation import' in src
    assert 'from lucid.sources.event_io import generate_events' not in src
