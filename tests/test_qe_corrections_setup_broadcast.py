"""Pin the simulator-setup broadcast behavior for scalar qe_corrections.

Every shipped ``*_physics_config.json`` sets ``qe_corrections: 1.0`` as a
scalar. ``setup_event_simulator`` must broadcast that to ``(num_sensors,)``
so downstream indexing (``qe_corrections[flat_indices]``) is well-defined.
"""
import os
import pytest

jnp = pytest.importorskip('jax.numpy')
from lucid.simulation import setup_event_simulator
from lucid.utils import base_dir_path


@pytest.mark.slow
def test_scalar_qe_corrections_broadcasts_at_setup():
    simulate = setup_event_simulator(
        base_dir_path() + 'config/SK_like_geom_config.json',
        n_photons=0,
        K=4,
        is_data=True,
        temperature=0.0,
        apply_smearing=False,
        physics_config=base_dir_path() + 'config/SK_like_physics_config.json',
        default_detector_params=True,
    )
    dp = simulate.default_detector_params
    # After setup, scalar qe_corrections must have been broadcast.
    assert dp.qe_corrections.ndim == 1, \
        f"Expected qe_corrections to be 1-D after setup, got ndim={dp.qe_corrections.ndim}"
    # All entries are the scalar placeholder value (1.0) by default.
    assert float(jnp.max(dp.qe_corrections)) == pytest.approx(1.0)
    assert float(jnp.min(dp.qe_corrections)) == pytest.approx(1.0)
