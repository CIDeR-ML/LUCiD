"""The photon step has finite value, gradient and Hessian at the 0/0 firing conditions.

``make_photon_iteration_update_factors_safe`` dropped a ``jax.custom_vjp`` whose backward pass
scrubbed NaN cotangents. Its docstring justifies the drop with "verified NaN-free without the
backstop across ~3·10⁴ grad/Hessian evals over random + on-surface stress geometries". That
verification was ``hessian_probe/recon_nan_thorough.py`` on the ``geometry-v2`` branch, and all 31
scripts in that directory ``sys.path``-insert and ``chdir`` to ``/sdf/group/neutrino/omara/
LUCiD_hessian`` before importing ``lucid`` — so they measured another checkout, and nothing in this
repository has ever checked the claim. This file is the promotion of that script: the same stress
geometries, the same value/gradient/Hessian sweep, run against this tree.

Two tiers, because the original needed a GPU-hour:

* The default tier calls the step directly at the exact conditions the dropped backstop existed for
  — a photon sitting ON the surface it just hit (``surface_distance == 0``, so the surface-distance
  norm is a 0/0), normal incidence, grazing incidence, and a scatter length short enough that the
  free-path branch is taken instead. Seconds, no detector.
* The ``--slow`` tier is the original end-to-end sweep: SK-like geometry, random tracks over the
  fiducial volume plus the five on-surface/grazing stress geometries, value + gradient + Hessian of
  a scalar in the per-photon charges, over many keys at two bounce counts.

A failure here is information, not a regression: it means the eps-inside-sqrt floors are
insufficient and ``custom_vjp`` cannot be dropped after all.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.simulation.photon_step import make_photon_iteration_update_factors_safe
from lucid.simulation.reflection import ScalarReflection
from tests.helpers.paths import REPO

_STEP = make_photon_iteration_update_factors_safe()

# Optical scalars held fixed; only the geometry varies across the cases below. Plain Python floats
# and tuples, not `jnp` arrays: nothing in this module may build a device array at import, or
# merely collecting it initialises a JAX backend for a run that then deselects it (T11).
_MIE_SCATTER_LENGTH = 1.0e9       # m, i.e. no Mie channel
_G = 0.0                          # dimensionless Henyey-Greenstein asymmetry
_ABSORPTION_LENGTH = 100.0        # m
_LAM = 400.0                      # nm
_SPEED_OF_LIGHT = 0.2253          # m/ns
_REFL = ScalarReflection(wall_rate=0.25, sensor_rate=0.25)

_N = (0.0, 0.0, -1.0)             # inward-facing surface normal at the hit point, dimensionless
_P = (0.5, 0.5, 2.0)              # the photon's position, m
_N_KEYS = 8                       # PRNG draws per case; the step samples a free path and an angle


def _surface_distance(position, hit_position):
    """The simulator's own surface-distance expression (``simulator.py:627``): eps INSIDE the
    sqrt, then the 1e-6 offset removed, so it is exactly 0 for a photon on its hit point. It is
    copied here because it is inline there; if that line changes, change this one."""
    return jnp.sqrt(jnp.sum((hit_position - position) ** 2) + 1e-12) - 1e-6


def _loss(position, hit_position, direction, normal, hit_sensor, key, scatter_length):
    """Every differentiable output of one step, summed — so no NaN channel can hide.

    The step is fed the surface distance computed from the two points, as the simulator does,
    so the 0/0 of the norm at ``hit_position == position`` is on the tape.

    Parameters
    ----------
    position, hit_position, direction, normal : (3,) float arrays
        Photon position and its hit point in metres, its unit direction, the unit surface
        normal at the hit point.
    hit_sensor : bool
        Sensor (specular) or wall (diffuse) reflection branch.
    key : PRNGKey
    scatter_length : float
        Rayleigh mean free path, metres.

    Returns
    -------
    float
        A scalar in mixed units (metres + ns + dimensionless); only the finiteness of its
        derivatives matters here, not its scale.
    """
    new_pos, new_dir, new_time, detect_prob, refl_atten, cont_factor, logp, _ = _STEP(
        position, direction, 0.0, _surface_distance(position, hit_position),
        normal, scatter_length, _MIE_SCATTER_LENGTH, _G, _REFL,
        _ABSORPTION_LENGTH, hit_sensor, _LAM, key, _SPEED_OF_LIGHT)
    return (jnp.sum(new_pos) + jnp.sum(new_dir) + new_time
            + detect_prob + refl_atten + cont_factor + logp)


# (label, surface_distance m, direction, normal, hit_sensor, scatter_length m)
#
# `surface_distance == 0.0` is the condition the dropped backstop existed for: a reflected photon
# that is still exactly on the surface it just hit, so |hit_position - position| == 0 and the
# surface-distance norm differentiates to 0/0. The hit point is placed at position + dd * direction. The grazing and normal-incidence rows add the
# reflection branch's own degeneracies, and the last row forces the scatter branch instead so the
# free-path side of every `jnp.where` is covered too.
CASES = [
    ('on-surface, normal incidence, sensor', 0.0, (0.0, 0.0, 1.0), _N, True, 50.0),
    ('on-surface, normal incidence, wall', 0.0, (0.0, 0.0, 1.0), _N, False, 50.0),
    ('on-surface, grazing, sensor', 0.0, (0.999, 0.0, 0.045), _N, True, 50.0),
    ('on-surface, grazing, wall', 0.0, (0.999, 0.0, 0.045), _N, False, 50.0),
    ('on-surface, direction in the surface', 0.0, (1.0, 0.0, 0.0), _N, True, 50.0),
    ('near-surface, grazing', 1e-7, (0.999, 0.0, 0.045), _N, False, 50.0),
    ('scatter branch, short free path', 2.0, (0.0, 0.0, 1.0), _N, True, 1e-3),
]


@pytest.mark.parametrize('label,dd,direction,normal,hit_sensor,sl', CASES)
def test_value_and_gradient_are_finite(label, dd, direction, normal, hit_sensor, sl):
    """Value and reverse-mode gradient wrt the photon position and its hit point are finite at
    every firing condition."""
    direction, normal = jnp.asarray(direction), jnp.asarray(normal)

    def f(position, hit_position, key):
        return _loss(position, hit_position, direction, normal, hit_sensor, key, sl)

    position = jnp.asarray(_P)
    hit = position + dd * direction
    for i in range(_N_KEYS):
        key = jax.random.PRNGKey(30000 + i)
        value = float(f(position, hit, key))
        assert np.isfinite(value), f'{label}: non-finite value {value}'
        g_pos, g_hit = jax.grad(f, argnums=(0, 1))(position, hit, key)
        assert np.isfinite(np.asarray(g_pos)).all(), f'{label}: non-finite d/d(position) {g_pos}'
        assert np.isfinite(np.asarray(g_hit)).all(), f'{label}: non-finite d/d(hit) {g_hit}'


@pytest.mark.parametrize('label,dd,direction,normal,hit_sensor,sl', CASES)
def test_hessian_is_finite(label, dd, direction, normal, hit_sensor, sl):
    """The Hessian is the half ``custom_vjp`` blocked, so it is the half that has to be checked."""
    direction, normal = jnp.asarray(direction), jnp.asarray(normal)

    def f(x, key):
        return _loss(x[:3], x[3:], direction, normal, hit_sensor, key, sl)

    position = jnp.asarray(_P)
    x = jnp.concatenate([position, position + dd * direction])
    for i in range(_N_KEYS):
        h = np.asarray(jax.hessian(f)(x, jax.random.PRNGKey(30000 + i)))
        assert np.isfinite(h).all(), f'{label}: non-finite Hessian\n{h}'


def test_forward_mode_is_available():
    """The reason the backstop was dropped: ``custom_vjp`` blocks ``jvp``/``jacfwd``, which the
    calibration and recon Jacobians need. If something reintroduces it, this fails first."""
    def f(hit_position):
        return _loss(jnp.asarray(_P), hit_position, jnp.asarray([0.0, 0.0, 1.0]),
                     jnp.asarray(_N), True, jax.random.PRNGKey(30000), 50.0)

    hit = jnp.asarray(_P) + jnp.asarray([0.0, 0.0, 1.5])
    primal, tangent = jax.jvp(f, (hit,), (jnp.asarray([0.0, 0.0, 1.0]),))
    assert np.isfinite(float(primal))
    assert np.isfinite(float(tangent))


# ---------------------------------------------------------------------------------------------
# The end-to-end sweep, promoted from hessian_probe/recon_nan_thorough.py.
#
# Gated on --slow by its mark. The control showing that an eps-outside norm really does NaN at
# zero (so these tests can detect the bug) is in test_sensor_deposit_nan_gradient.py.
# ---------------------------------------------------------------------------------------------

_GEOM = os.path.join(REPO, 'config', 'SK_like_geom_config.json')
_PHYS = os.path.join(REPO, 'config', 'SK_like_physics_config.json')

# The original swept NT=40 random tracks x NKR=150 keys plus 5 stress geometries x NKS=400 keys at
# K in {8, 16}: ~3e4 gradient evaluations and a GPU-hour. These are the same geometries and the
# same quantities at a size that fits a `--slow` CPU run; raise them to the originals to reproduce
# the docstring's number exactly.
_N_RANDOM_TRACKS = 4
_N_KEYS_RANDOM = 3
_N_KEYS_STRESS = 4
_N_KEYS_HESSIAN = 1
_K_VALUES = (8,)
_N_PHOTONS = 2_000


@pytest.mark.slow
@pytest.mark.requires_emitter('muon', 'water')
def test_no_nan_over_the_stress_sweep():
    """No NaN in value, gradient or Hessian over random + on-surface/grazing track geometries.

    This is the claim ``make_photon_iteration_update_factors_safe``'s docstring makes, measured on
    this tree rather than on the checkout ``hessian_probe/`` chdir'd into.
    """
    from lucid.fitting.recon import track_from_vec9, vec9_from_track
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator

    det = generate_detector(_GEOM)
    n_sensors = len(det.all_points)
    radius = float(getattr(det, 'R', 16.9))          # m
    height = float(getattr(det, 'H', 36.2))          # m

    rng = np.random.default_rng(123)
    weights = jnp.asarray(np.random.default_rng(0).standard_normal(n_sensors))

    def random_tracks(n):
        out = []
        for _ in range(n):
            r = (radius - 0.3) * np.sqrt(rng.uniform())
            phi = rng.uniform(0.0, 2.0 * np.pi)
            position = [r * np.cos(phi), r * np.sin(phi),
                        rng.uniform(-(height / 2 - 0.3), height / 2 - 0.3)]
            d = rng.standard_normal(3)
            out.append((float(rng.uniform(200.0, 1500.0)), position,
                        (d / np.linalg.norm(d)).tolist()))
        return out

    # The five geometries that fire the 0/0: on the wall, on a cap, down the axis into a cap, and
    # a ray that grazes the wall for most of its length.
    stress = [('center', [2.0, -1.0, 3.0], [0.2, 0.1, 0.97]),
              ('near-wall', [radius - 0.5, 0.0, 0.0], [1.0, 0.0, 0.02]),
              ('near-cap', [0.0, 0.0, height / 2 - 0.5], [0.05, 0.05, 1.0]),
              ('axis-long', [0.0, 0.0, -height / 2 + 1.0], [0.0, 0.0, 1.0]),
              ('wall-graze', [0.0, 0.0, 0.0], [0.999, 0.0, 0.045])]

    n_checked = {'value': 0, 'grad': 0, 'hessian': 0}
    nan_at = []

    for k_bounces in _K_VALUES:
        predict = setup_event_simulator(
            _GEOM, _N_PHOTONS, source='particle', temperature=0.1, K=k_bounces,
            physics_config=_PHYS, default_detector_params=True, particle='muon',
            wavelength_mode=True, n_cap=80, n_angular=120, n_height=80)

        def loss(t9, key):
            """Weighted sum of the per-photon charges — element 3 of the per-photon return."""
            return jnp.sum(weights * predict(track_from_vec9(t9), key)[3])

        grad_fn = jax.jit(jax.grad(loss))
        hess_fn = jax.jit(jax.hessian(loss))

        jobs = ([('random', e, p, d, _N_KEYS_RANDOM) for e, p, d in random_tracks(_N_RANDOM_TRACKS)]
                + [(label, 1050.0, p, d, _N_KEYS_STRESS) for label, p, d in stress])

        for label, energy, position, direction, n_keys in jobs:
            t9 = jnp.asarray(vec9_from_track(energy, position, direction, 0.0), float)
            for i in range(n_keys):
                key = jax.random.PRNGKey(30000 + i)
                n_checked['value'] += 1
                if not np.isfinite(float(loss(t9, key))):
                    nan_at.append(f'value  K={k_bounces} {label} key={i}')
                n_checked['grad'] += 1
                if not np.all(np.isfinite(np.asarray(grad_fn(t9, key)))):
                    nan_at.append(f'grad   K={k_bounces} {label} key={i}')
            for i in range(min(n_keys, _N_KEYS_HESSIAN)):
                key = jax.random.PRNGKey(30000 + i)
                n_checked['hessian'] += 1
                if not np.all(np.isfinite(np.asarray(hess_fn(t9, key)))):
                    nan_at.append(f'hess   K={k_bounces} {label} key={i}')

    assert not nan_at, (
        f'{len(nan_at)} non-finite result(s) out of '
        f"{n_checked['value']} value / {n_checked['grad']} gradient / {n_checked['hessian']} "
        'Hessian evaluations — the eps-inside-sqrt floors are insufficient and the dropped '
        'custom_vjp backstop is still needed:\n  ' + '\n  '.join(nan_at[:20]))
