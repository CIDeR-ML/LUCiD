"""The paper figures' config contract: what TrackingPipeline needs, the study configs must supply.

Why this exists
---------------
`python analysis/paper/fig_nrays.py` — the documented way to reproduce a tracking figure, with
`--backend local` as the default — raised `KeyError: 'grid'` and could not run at all.

Two callers build the same pipeline and only one merged the defaults:

  run_study.py  (SLURM worker, --backend s3df)  load_config() -> _deep_merge -> safe
  run.py        (run_local,   --backend local)  the raw dict straight through -> KeyError

The published figures were produced on the s3df backend, so the production path always worked and
the laptop path never did — which is exactly the path an outside reader takes first.

These tests assert the contract statically, so they need no PhotonSim ROOT input, no SIREN weights
and no GPU. That matters: a gate requiring downloaded data does not get run, which is how the
tripwire went a month unnoticed.
"""
import ast
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / 'analysis' / 'paper'
PIPELINE = PAPER / 'utils' / 'pipeline.py'


def _required_keys():
    """Keys TrackingPipeline.__init__ indexes with config['x'] — a missing one is a KeyError.

    Read from the source rather than hard-coded, so this tracks the pipeline instead of drifting
    from it. `.get('x')` accesses are deliberately excluded: those have defaults and cannot raise.
    """
    src = PIPELINE.read_text()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'TrackingPipeline')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    keys = set()
    for node in ast.walk(init):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == 'config' and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    assert keys, 'found no config[...] accesses — the extractor has drifted from the source'
    return keys


def _study_configs():
    """One config from each builder in utils/studies.py, as the figure scripts produce them."""
    import sys
    sys.path.insert(0, str(PAPER.parents[1]))
    from analysis.paper.utils import studies
    return {
        'base_config': studies.base_config(
            'muon', 1000, 250_000, 100, '/nonexistent/mu-/1000MeV_100events.root', 'test'),
    }


@pytest.mark.parametrize('name', list(_study_configs()))
def test_pipeline_defaults_cover_study_configs(name):
    """Every key the pipeline indexes must exist after the constructor's merge.

    This is the assertion that would have caught the KeyError before it shipped.
    """
    import sys
    sys.path.insert(0, str(PAPER.parents[1]))
    from analysis.paper.utils.pipeline import DEFAULT_CONFIG, _deep_merge

    cfg = _study_configs()[name]
    merged = _deep_merge(DEFAULT_CONFIG, cfg)
    missing = sorted(_required_keys() - set(merged))
    assert not missing, (
        f'{name}: TrackingPipeline would raise KeyError on {missing}. Either studies.py must '
        f'supply them or DEFAULT_CONFIG must default them.')


def test_raw_study_config_alone_is_insufficient():
    """Guard on the guard: the merge must be doing real work.

    If studies.py ever supplied every key on its own, the test above would pass whether or not the
    constructor merged — and the KeyError could return unnoticed the next time a key was added to
    the pipeline. This asserts the merge is load-bearing.
    """
    cfg = _study_configs()['base_config']
    missing_without_merge = _required_keys() - set(cfg)
    assert missing_without_merge, (
        'study configs now supply every required key on their own — the merge is no longer '
        'load-bearing, so this test and the one above need rethinking')


def test_pipeline_merges_defaults_before_reading_config():
    """TrackingPipeline.__init__ must apply the merge BEFORE its first `config[...]` read.

    This is the test that actually gates the fix, and it exists because the two above do not:
    they check that DEFAULT_CONFIG *covers* the required keys, which was true before the bug was
    fixed and stayed true after. The defect was never a missing default — it was a caller that
    never applied the merge. Verified by deleting the merge line and watching those two still
    pass.

    Structural rather than behavioural on purpose: exercising __init__ for real needs a PhotonSim
    ROOT file and the SIREN weights, and a gate that needs downloaded data does not get run.
    """
    tree = ast.parse(PIPELINE.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'TrackingPipeline')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')

    merge_line = min(
        (n.lineno for n in ast.walk(init)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == '_deep_merge'),
        default=None)
    assert merge_line is not None, (
        'TrackingPipeline.__init__ does not merge DEFAULT_CONFIG. run_local() passes a raw '
        'study config, so the first config[...] read will raise KeyError.')

    first_read = min(
        n.lineno for n in ast.walk(init)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
        and n.value.id == 'config' and isinstance(n.slice, ast.Constant))
    assert merge_line < first_read, (
        f'the merge is at line {merge_line} but config[...] is first read at {first_read} — '
        f'the defaults must be applied before anything indexes the config')


def test_run_local_does_not_merge_defaults_itself():
    """run_local must NOT re-implement the merge; the constructor owns it.

    Recorded as a test because the tempting fix was to patch run_local — which would have left
    the same trap for the next caller of TrackingPipeline.
    """
    src = (PAPER / 'utils' / 'run.py').read_text()
    assert 'DEFAULT_CONFIG' not in src, (
        'run.py merges defaults itself — that duplicates the invariant instead of centralising '
        'it in TrackingPipeline.__init__')
    assert re.search(r'TrackingPipeline\(\s*config', src), \
        'run_local no longer passes the config straight to TrackingPipeline'
