"""`DEFAULT_RECIPE` must stay splattable into `fit_track` — minus the one key that is not its own.

`tutorials/track_optimization.ipynb` calls
`fit_track(..., **{k: v for k, v in RECIPE.items() if k != 'time_weight'})`, and the suite does
not run notebooks, so these tests are the only check on that call. The recipe is not a copy of
`fit_track`'s defaults:

  * `trust=3.0` pins what the signature leaves as `'auto'`, so it is a deliberate configuration.
  * `time_weight` configures the model's time term and is NOT a `fit_track` argument. Renaming a
    `fit_track` parameter makes the splat raise TypeError; adding `time_weight` to the signature
    makes the tutorial's filter silently drop it, so the fit runs at the default time weight.
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
    """`trust` is pinned at 3.0 against a signature default of 'auto'.

    If the two ever agree, the recipe is no longer a deliberate choice and the module docstring is wrong.
    """
    sig = inspect.signature(fit_track).parameters
    assert DEFAULT_RECIPE['trust'] == 3.0
    assert sig['trust'].default == 'auto', (
        "fit_track's trust default is no longer 'auto'; DEFAULT_RECIPE pinning 3.0 may now be "
        'redundant, and the reason this recipe exists needs restating')


def test_the_tutorial_still_imports_it_from_here():
    """The tutorial imports the recipe from here, not from the removed `lucid.fitting.sweep`.

    Read as text because the suite does not execute notebooks.
    """
    from pathlib import Path
    nb = Path(__file__).resolve().parents[1] / 'tutorials' / 'track_optimization.ipynb'
    if not nb.exists():
        pytest.skip('tutorial not present in this checkout')
    src = nb.read_text()
    assert 'lucid.fitting.sweep' not in src, (
        'the tutorial still imports lucid.fitting.sweep, which no longer exists')
    assert 'DEFAULT_RECIPE' in src, 'the tutorial no longer imports the recipe at all'
