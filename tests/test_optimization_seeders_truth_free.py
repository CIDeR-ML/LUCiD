"""The direction and energy seeders: they take a truth argument, and must not use it.

`tests/test_optimization_seeder.py` covers the POSITION seeder. This file covers the other two:

    hierarchical_direction_search_cone(pred, position, t0, hits, times, charge, true_data, ...)
    energy_scan_optimization(pred, position, theta, phi, t0, hits, times, charge, true_data, ...)

In both, `true_data` is an unused leftover parameter (scoring uses only `counts_loss` on the
observed charge). It must stay unused: a seeder that reads truth is an oracle.

The seeders need only ``pred(track, key) -> (log_w, times, indices, total_charge)``, so a
closed-form charge pattern peaked at a known direction stands in for the photon forward and
lets these run in CI.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from lucid.optimization.utils.functions import (
    hierarchical_direction_search_cone, energy_scan_optimization,
)

NS = 48
TRUE_DIR = np.array([0.48, 0.36, 0.8])
TRUE_DIR = TRUE_DIR / np.linalg.norm(TRUE_DIR)
TRUE_E = 900.0
POSITION = np.array([0.5, -0.3, 1.0])
T0 = 2.0
_rng = np.random.default_rng(3)
SENSOR_DIR = _rng.standard_normal((NS, 3))
SENSOR_DIR /= np.linalg.norm(SENSOR_DIR, axis=1, keepdims=True)


def _charge(direction, energy):
    """A Cherenkov-ish ring: brightest where the sensor sits near the track direction."""
    cos = SENSOR_DIR @ np.asarray(direction, float)
    return (energy / TRUE_E) * (10.0 * np.exp(3.0 * cos) + 1.0)


OBSERVED = jnp.asarray(_charge(TRUE_DIR, TRUE_E))


def _stub_pred(track, key):
    """(log_w, flat_times, flat_indices, total_charge) — only the last is read."""
    th, ph = float(track.theta), float(track.phi)
    d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return None, None, None, jnp.asarray(_charge(d, float(track.energy)))


def _direction(true_data, levels=3, initial_div=8):
    return hierarchical_direction_search_cone(
        _stub_pred, jnp.asarray(POSITION), T0, None, None, OBSERVED,
        true_data, TRUE_E, levels, initial_div, 90.0, 0.5, 0)


def _energy(true_data, n_steps=9):
    th = float(np.arccos(TRUE_DIR[2]))
    ph = float(np.arctan2(TRUE_DIR[1], TRUE_DIR[0]))
    return energy_scan_optimization(
        _stub_pred, jnp.asarray(POSITION), th, ph, T0, None, None, OBSERVED,
        true_data, TRUE_E * 1.3, TRUE_E * 0.5, n_steps, 0)


def _unit(res):
    t, p = res['best_theta'], res['best_phi']
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


class TestDirectionSeeder:
    def test_it_finds_the_direction_the_charge_pattern_came_from(self):
        """If it could not, truth-independence below would be a statement about a broken search."""
        got = _unit(_direction(None))
        ang = np.degrees(np.arccos(np.clip(got @ TRUE_DIR, -1, 1)))
        assert ang < 12.0, f'seed direction is {ang:.1f} deg off the pattern it was given'

    def test_the_truth_argument_changes_nothing(self):
        a = _direction(None)
        b = _direction(('nonsense', np.full(NS, 1e6)))
        assert a['best_theta'] == b['best_theta']
        assert a['best_phi'] == b['best_phi']
        assert a['best_loss'] == b['best_loss']

    def test_the_observed_charge_does_matter(self):
        """The control for the test above: something the seeder DOES read must move the answer."""
        global OBSERVED
        keep = OBSERVED
        try:
            OBSERVED = jnp.asarray(_charge(-TRUE_DIR, TRUE_E))       # ring flipped
            flipped = _unit(_direction(None))
        finally:
            OBSERVED = keep
        ang = np.degrees(np.arccos(np.clip(flipped @ TRUE_DIR, -1, 1)))
        assert ang > 90.0, 'reversing the charge ring did not move the seed'


class TestEnergySeeder:
    def test_it_finds_the_energy_the_charge_pattern_came_from(self):
        res = _energy(None)
        assert abs(res['best_energy'] - TRUE_E) / TRUE_E < 0.15, res['best_energy']

    def test_the_truth_argument_changes_nothing(self):
        a = _energy(None)
        b = _energy(('nonsense', np.full(NS, 1e6)))
        assert a['best_energy'] == b['best_energy']
        assert a['best_loss'] == b['best_loss']


def test_the_truth_parameters_are_genuinely_dead():
    """`true_data` is not referenced in either seeder's source at all.

    The behavioural tests only cover today's inputs; this fails as soon as the parameter is read.
    """
    import ast
    import inspect
    for fn in (hierarchical_direction_search_cone, energy_scan_optimization):
        tree = ast.parse(inspect.getsource(fn))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert 'true_data' not in names, (
            f'{fn.__name__} now reads `true_data`. If that is deliberate, it must be justified: '
            f'a seeder that reads truth is an oracle, and no downstream metric can detect it.')
