"""Fast unit tests for the pluggable spatial absorption field (``lucid.simulation.fields``)
and its opaque ``field_params`` leaf in ``DetectorParams``.

These exercise the field representations, the gauge/normamp wrappers, and the
``field_params``-is-opaque contract (normalize pass-through, save/load → None) WITHOUT building a
detector or running a sim — so they stay in the default (fast) suite. The end-to-end
byte-identical-when-off check lives in the slow companion ``test_absorption_field_e2e.py``.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.simulation.fields import make_field, _reference_set
from lucid import detector_params as dp

R, H = 5.0, 10.0
PTS = jnp.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [3.0, -1.0, -2.0], [-2.0, 1.5, 4.0]])

IDENTITY_AT_INIT = ["uniform", "poly", "grid"]     # init params ⇒ field ≡ 1
NONIDENTITY_AT_INIT = ["siren", "siren_grid"]      # SIREN has no zeroable constant mode
ALL_KINDS = IDENTITY_AT_INIT + NONIDENTITY_AT_INIT


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_forward_finite(kind):
    apply, init = make_field(kind, R=R, H=H)
    out = apply(init, PTS)
    assert out.shape == (PTS.shape[0],)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(out > 0))           # multiplicative correction must stay positive


@pytest.mark.parametrize("kind", IDENTITY_AT_INIT)
def test_identity_at_init(kind):
    apply, init = make_field(kind, R=R, H=H)
    np.testing.assert_allclose(np.asarray(apply(init, PTS)), 1.0, atol=1e-6)


@pytest.mark.parametrize("kind", NONIDENTITY_AT_INIT)
def test_siren_is_not_identity_at_init(kind):
    # documents the contract: a SIREN field perturbs absorption (~10%) at init, by design
    apply, init = make_field(kind, R=R, H=H)
    assert not np.allclose(np.asarray(apply(init, PTS)), 1.0, atol=1e-3)


@pytest.mark.parametrize("kind", ["poly", "grid", "siren", "siren_grid"])
def test_field_params_grad_live(kind):
    apply, init = make_field(kind, R=R, H=H)

    def loss(fp):
        return jnp.sum((apply(fp, PTS) - 1.3) ** 2)

    g = jax.grad(loss)(init)
    flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)])
    assert bool(jnp.all(jnp.isfinite(flat)))
    assert float(jnp.sum(jnp.abs(flat))) > 0.0      # gradient genuinely flows to the leaf


@pytest.mark.parametrize("kind", ["poly", "grid"])
def test_field_params_jacfwd(kind):
    # forward-mode must work (the engine dropped custom_vjp specifically to allow jacfwd). Only the
    # low-dim reps are checked: jacfwd over a SIREN weight pytree is O(n_weights) JVPs (impractical,
    # and not a real use case — SIREN field weights are reverse-mode optimized).
    apply, init = make_field(kind, R=R, H=H)
    j = jax.jacfwd(lambda fp: apply(fp, PTS))(init)
    flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(j)])
    assert bool(jnp.all(jnp.isfinite(flat)))


# --- gauge / normamp -------------------------------------------------------

def test_gauge_pins_mean_over_own_refset():
    # the wrapper pins the mean over ITS fixed ref set (seed 12345) — assert there, not on a
    # fresh sample (finite-sample residual ~2% otherwise).
    apply, init = make_field("siren_grid", R=R, H=H, gauge=True)
    ref = _reference_set(R, H, 2048, 12345)
    assert float(jnp.mean(apply(init, ref))) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("kind", ["uniform", "poly"])
def test_gauge_noop_on_self_gauging_reps(kind):
    # uniform/poly carry no free constant mode ⇒ gauge leaves them at identity
    apply, init = make_field(kind, R=R, H=H, gauge=True)
    np.testing.assert_allclose(np.asarray(apply(init, PTS)), 1.0, atol=1e-5)


def test_normamp_structure_and_amplitude():
    apply, init = make_field("siren_grid", R=R, H=H, normamp=True, init_amp=0.05)
    assert set(init.keys()) == {"w", "log_amp"}
    np.testing.assert_allclose(float(jnp.exp(init["log_amp"])), 0.05, rtol=1e-5)
    # the physical claim: deviation is unit-RMS-times-amp over the VOLUME. Measure over an
    # INDEPENDENT ref set (seed 999 ≠ the normamp's own seed 777) so this isn't tautological —
    # ~few-% finite-sample residual is expected.
    ref = _reference_set(R, H, 4000, 999)
    dev_rms = float(jnp.sqrt(jnp.mean((apply(init, ref) - 1.0) ** 2)))
    assert dev_rms == pytest.approx(0.05, rel=0.10)


def test_normamp_amp_only_scales_not_shape():
    # init_amp must scale the magnitude ONLY — the spatial PATTERN (deviation / amp) is invariant.
    a1, i1 = make_field("siren_grid", R=R, H=H, normamp=True, init_amp=0.05)
    a2, i2 = make_field("siren_grid", R=R, H=H, normamp=True, init_amp=0.10)
    shape1 = (a1(i1, PTS) - 1.0) / jnp.exp(i1["log_amp"])
    shape2 = (a2(i2, PTS) - 1.0) / jnp.exp(i2["log_amp"])
    np.testing.assert_allclose(np.asarray(shape1), np.asarray(shape2), rtol=1e-5)


@pytest.mark.parametrize("kind", ["poly", "grid", "siren", "siren_grid"])
def test_normamp_grad_finite(kind):
    # regression for the eps-OUTSIDE-sqrt NaN: at a zero/identity init (poly, grid) the deviation
    # variance is exactly 0, so an outside-the-sqrt floor leaves d√var = NaN on step 0. The
    # eps-inside-sqrt fix must keep the gradient finite for every rep, including the zero-init ones.
    apply, init = make_field(kind, R=R, H=H, normamp=True)

    def loss(fp):
        return jnp.sum((apply(fp, PTS) - 1.2) ** 2)

    g = jax.grad(loss)(init)
    flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)])
    assert bool(jnp.all(jnp.isfinite(flat)))


# --- opaque-leaf contract in DetectorParams --------------------------------

def _params_with(field_params, n_sensors=16):
    bmin, _ = dp.default_bounds(n_sensors)
    return bmin._replace(absorption=bmin.absorption._replace(field_params=field_params))


def test_field_params_none_adds_no_leaves():
    # the byte-identical contract: field_params=None ⇒ unchanged flat leaf set + pytree leaf count
    bmin, _ = dp.default_bounds(16)
    assert bmin.absorption.field_params is None
    assert "field_params" not in dp._FLAT_FIELDS
    n_with = len(jax.tree_util.tree_leaves(bmin))
    n_pop = len(jax.tree_util.tree_leaves(_params_with(jnp.zeros(8))))
    assert n_pop == n_with + 1     # exactly one extra leaf when populated, zero when None


@pytest.mark.parametrize("fp_kind", ["array", "normamp_dict"])
def test_normalize_passthrough(fp_kind):
    _, init = make_field("poly", R=R, H=H)
    if fp_kind == "array":
        fp = init + 0.01
    else:
        _, fp = make_field("siren_grid", R=R, H=H, normamp=True)
    bmin, bmax = dp.default_bounds(16)
    params = _params_with(fp)
    norm = dp.normalize_params(params, bmin, bmax)        # must NOT raise on structure mismatch
    deno = dp.denormalize_params(norm, bmin, bmax)
    # field_params passes through un-normalized on BOTH normalize and denormalize, recovered exactly
    for tree in (norm.absorption.field_params, deno.absorption.field_params):
        for a, b in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(fp)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # a normal scalar still round-trips through normalize→denormalize
    np.testing.assert_allclose(np.asarray(deno.scattering.scatter_length),
                               np.asarray(params.scattering.scatter_length), rtol=1e-5)


def test_normalize_none_unchanged():
    bmin, bmax = dp.default_bounds(16)
    norm = dp.normalize_params(bmin, bmin, bmax)
    assert norm.absorption.field_params is None


def test_save_load_drops_field_params(tmp_path):
    # B1 honest behavior: field_params is NOT persisted — round-trips to None, scalars preserved
    _, fp = make_field("poly", R=R, H=H)
    params = _params_with(fp + 0.05)
    path = str(tmp_path / "det.json")
    dp.save_detector_params(params, path)
    loaded = dp.load_detector_params(path, num_sensors=16)
    assert loaded.absorption.field_params is None
    np.testing.assert_allclose(np.asarray(loaded.absorption.absorption_length),
                               np.asarray(params.absorption.absorption_length), rtol=1e-5)
