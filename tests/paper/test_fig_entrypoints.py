"""Smoke tests for the 14 paper figure entry points and the two recipe modules.

This is L2 of the light gate written down as pytest, so CI runs every figure's
imports and argument parser without any fetched data. 26 of the 36 ``analysis/paper`` modules are referenced
by no other test, including 13 of the 14 figures and both ``calib_run.py`` and
``calibration.py``: an import error in any of them reaches a reader only when
they try to make the figure.
"""
import runpy
import sys
import warnings
from pathlib import Path

import pytest
from tests.helpers.paths import REPO

FIGURES = sorted(p.stem for p in (REPO / 'analysis' / 'paper').glob('fig_*.py'))

@pytest.fixture(scope='module', autouse=True)
def _repo_on_path():
    """``analysis`` is a source tree, not an installed package."""
    sys.path.insert(0, str(REPO))
    yield
    sys.path.remove(str(REPO))


def test_the_figure_set_is_the_published_fourteen():
    """analysis/paper ships exactly 14 fig_*.py.

    The contract in ``analysis/paper/README.md`` is one command per figure, so
    the count is part of it: a fifteenth figure has to appear here before it
    can quietly go uncovered.
    """
    assert len(FIGURES) == 14, f'expected 14 fig_*.py, found {len(FIGURES)}: {FIGURES}'


@pytest.mark.parametrize('mod_name', FIGURES)
def test_figure_help_runs_as_a_module_without_executing(mod_name, monkeypatch, capsys):
    """``python -m analysis.paper.<fig> --help`` prints usage and exits 0, doing no work.

    Run through ``runpy`` with ``run_name='__main__'``, which is what ``python -m`` does, so this
    also needs the ``__main__`` guard and a ``main()`` that takes no required argument: a missing
    guard returns without SystemExit, a required argument raises TypeError. argparse raises
    SystemExit from parse_args, the first statement after the parser is built in all 14 scripts,
    so nothing downstream of it runs. Running the module executes its imports, so an import
    error fails here too (there is no separate import-only test).
    """
    monkeypatch.setattr(sys, 'argv', [f'{mod_name}.py', '--help'])
    with warnings.catch_warnings():
        # runpy warns when the module is already in sys.modules (an earlier test imported it)
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(f'analysis.paper.{mod_name}', run_name='__main__', alter_sys=True)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith('usage:'), f'{mod_name} --help printed no usage line'


@pytest.mark.slow
def test_build_shared_constructs_at_a_reduced_photon_budget():
    """``calib_run.build_shared`` builds the shared calibration forward.

    Called at 1/1000 of the published photon budget and the smallest legal K,
    so this asserts the construction path, not the fit.
    """
    from analysis.paper.utils import calib_run, calibration
    sim = calib_run.build_shared(n_ph=1000, k=2, grid=calibration.CALIB['grid'])
    assert callable(sim)
    sources = calibration._laser_sources(1e6)
    assert len(sources) == 8, f'the canonical layout is 1 laser + 7 isotropic, got {len(sources)}'


def test_the_calibration_recipe_is_exactly_run_seeds_arguments():
    """``CALIB`` is passed as ``run_seed(**CALIB)``: every key is a parameter and every
    recipe parameter of ``run_seed`` is set, so the published run takes no default."""
    import inspect
    from analysis.paper.utils import calib_run, calibration
    params = inspect.signature(calib_run.run_seed).parameters
    recipe_params = {n for n, p in params.items()
                     if p.kind is p.KEYWORD_ONLY and p.default is inspect.Parameter.empty}
    assert set(calibration.CALIB) == recipe_params


def test_the_calibration_recipe_runs_the_published_step_count():
    """analysis/paper/README.md (Calibration figures): "The paper's run is 1200 Gauss-Newton
    steps per seed". calibration.py gives the reason: with the leg-bounded deposit the
    diffuse-reflection pair reaches truth only by ~1200 steps. Three seeds (README: "3 seeds")."""
    from analysis.paper.utils import calibration
    assert calibration.CALIB['steps'] == 1200
    assert tuple(calibration.CALIB_SEEDS) == (0, 1, 2)


def test_calibration_and_studies_name_the_same_detector():
    """The calibration figures and the reconstruction studies share one geometry.

    ``calibration.py`` calls itself the counterpart to ``studies.py``; the claim
    is only true while both sit on SK_like, and neither reads the other.
    """
    from analysis.paper.utils import calibration, studies
    cfg = studies.base_config('muon', 1000, 5000, 1, 'unused.root', 'smoke')
    assert Path(calibration.GEOM).name == Path(cfg['geom_config']).name
