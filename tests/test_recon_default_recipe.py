"""`DEFAULT_RECIPE` must stay splattable into `fit_track` — minus the one key that is not its own.

The recipe used to live in `lucid/fitting/sweep.py`, a characterisation driver with no caller in
the repo and no published number behind it. The driver was removed; the recipe moved to
`lucid.fitting.recon`, beside the function it configures, because one thing did consume it:
`tutorials/track_optimization.ipynb` imports it and nothing else from that module.

The contract is not "these are fit_track's defaults". Two keys make it more specific than that:

  * `trust=3.0` PINS what `fit_track`'s signature leaves as `'auto'` — so the recipe is a
    deliberate configuration, not a restatement, and comparing the two for equality would be
    wrong.
  * `time_weight` is NOT a `fit_track` argument. It configures the MODEL's time term, so a caller
    has to split the dict. The tutorial does exactly that:
    `fit_track(..., **{k: v for k, v in RECIPE.items() if k != 'time_weight'})`.

Both halves are load-bearing and fail in opposite directions. If a `fit_track` parameter is
renamed, the splat raises TypeError in a shipped tutorial that CI never executes. If someone
"tidies" `time_weight` into the signature, the tutorial's filter becomes wrong in the other
direction — it would then be dropping a real argument, silently reverting the time weight to a
default. Neither shows up anywhere else: notebooks are not run by the suite.
"""
import inspect

import pytest

from lucid.fitting.recon import DEFAULT_RECIPE, fit_track

FIT_TRACK_PARAMS = set(inspect.signature(fit_track).parameters)
MODEL_ONLY = {'time_weight'}


@pytest.mark.parametrize('key', sorted(set(DEFAULT_RECIPE) - MODEL_ONLY))
def test_every_recipe_knob_is_a_real_fit_track_parameter(key):
    """The splat must not raise. Checked per key so a failure names the offender."""
    assert key in FIT_TRACK_PARAMS, (
        f"DEFAULT_RECIPE['{key}'] is not a fit_track parameter, so "
        f"fit_track(**{{k: v for k, v in RECIPE.items() if k != 'time_weight'}}) raises TypeError "
        f"in tutorials/track_optimization.ipynb, which no test executes")


def test_time_weight_is_still_NOT_a_fit_track_parameter():
    """The other direction, and the one that fails silently rather than loudly.

    The tutorial filters `time_weight` out. If it became a real `fit_track` argument, that filter
    would start DROPPING a live setting instead of removing a foreign one, and the fit would
    quietly run at the default time weight.
    """
    assert 'time_weight' not in FIT_TRACK_PARAMS, (
        'time_weight is now a fit_track parameter, so the tutorial is silently dropping it; '
        'either stop filtering it there or take it out of DEFAULT_RECIPE')


def test_the_recipe_is_a_configuration_not_a_copy_of_the_defaults():
    """Guards the comment as much as the code.

    `trust` is pinned at 3.0 against a signature default of 'auto'. If someone ever makes the two
    agree, the recipe stops being a deliberate choice and the docstring above becomes wrong — so
    this asserts the disagreement rather than assuming it.
    """
    sig = inspect.signature(fit_track).parameters
    assert DEFAULT_RECIPE['trust'] == 3.0
    assert sig['trust'].default == 'auto', (
        "fit_track's trust default is no longer 'auto'; DEFAULT_RECIPE pinning 3.0 may now be "
        'redundant, and the reason this recipe exists needs restating')


def test_the_tutorial_still_imports_it_from_here():
    """The move is only complete if the consumer follows it.

    Read as text -- notebooks are JSON and the suite does not execute them, so this is the only
    cheap way to notice that a shipped tutorial still points at the deleted module.
    """
    from pathlib import Path
    nb = Path(__file__).resolve().parents[1] / 'tutorials' / 'track_optimization.ipynb'
    if not nb.exists():
        pytest.skip('tutorial not present in this checkout')
    src = nb.read_text()
    assert 'lucid.fitting.sweep' not in src, (
        'the tutorial still imports lucid.fitting.sweep, which no longer exists')
    assert 'DEFAULT_RECIPE' in src, 'the tutorial no longer imports the recipe at all'
