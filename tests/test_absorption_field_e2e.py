"""Slow end-to-end checks for the spatial absorption field, through ``setup_event_simulator``.

Gated slow (builds a detector + JIT-compiles a propagator + runs a sim). Verifies the two
contract properties that only show up at the sim level:

* **byte-identical-when-off** — a ``poly`` field at its zero (identity) init produces the same
  charges as the homogeneous (``absorption_field=None``) engine.
* **B3 NaN-guard** — configuring a field but leaving ``DetectorParams.field_params=None`` (the
  config-loaded default) does NOT feed ``None`` into the field (which would NaN); it falls back to
  the field's identity init.

The fast field/unit contract lives in ``test_absorption_field.py``.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
SK_LIKE_GEOM = os.path.join(CONFIG, "SK_like_geom_config.json")
SK_LIKE_PHYS = os.path.join(CONFIG, "SK_like_physics_config.json")
KEY = jax.random.PRNGKey(0)
NPHOT, K_VAL = 50_000, 4

pytestmark = pytest.mark.slow


def _build(absorption_field):
    from lucid.simulation import setup_event_simulator
    return setup_event_simulator(
        SK_LIKE_GEOM, n_photons=NPHOT, temperature=None, K=K_VAL,
        is_calibration=True, physics_config=SK_LIKE_PHYS,
        default_detector_params=True, wavelength_mode=False,
        absorption_field=absorption_field,
    )


def test_poly_identity_matches_homogeneous_and_no_nan():
    from lucid.sources import isotropic_source
    src = isotropic_source(position=[0., 0., 0.], intensity=1e8)

    c_off, _ = _build(None)(src, KEY)
    c_poly, _ = _build("poly")(src, KEY)         # field_params=None → B3 guard → poly zeros (identity)

    # B3: a configured field with an unseeded (None) leaf must not NaN
    assert bool(jnp.all(jnp.isfinite(c_poly)))
    assert float(jnp.sum(c_poly)) > 0
    # byte-identical-when-off: identity poly ⇒ field_mult ≡ 1.0 exactly ⇒ exp(-d/L·1)==exp(-d/L),
    # so this is BIT-exact, not merely close. Assert equality to catch any deviation leaking into
    # the identity path (the precise failure this test guards).
    np.testing.assert_array_equal(np.asarray(c_poly), np.asarray(c_off))


def test_field_rejects_non_cylinder_geometry():
    # blocker-2 guard: the field encoding is cylindrical (r, θ, z); a sphere detector (JUNO) must
    # raise a clear error rather than AttributeError on the missing .H or silently mis-encode.
    from lucid.simulation import setup_event_simulator
    juno_geom = os.path.join(CONFIG, "JUNO_geom_config.json")
    juno_phys = os.path.join(CONFIG, "JUNO_physics_config.json")
    with pytest.raises(ValueError, match="cylinder"):
        setup_event_simulator(
            juno_geom, n_photons=NPHOT, temperature=None, K=K_VAL,
            is_calibration=True, physics_config=juno_phys,
            default_detector_params=True, wavelength_mode=False,
            absorption_field="poly")


def test_uniform_is_exactly_the_off_path():
    # 'uniform' is mapped to no field at all (same code path) ⇒ exactly equal to None
    from lucid.sources import isotropic_source
    src = isotropic_source(position=[0., 0., 0.], intensity=1e8)
    c_off, _ = _build(None)(src, KEY)
    c_uni, _ = _build("uniform")(src, KEY)
    np.testing.assert_array_equal(np.asarray(c_uni), np.asarray(c_off))
