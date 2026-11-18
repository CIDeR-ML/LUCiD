"""``setup_event_simulator`` returns a ``Simulator`` record, not a function with attributes stamped on.

Production reads what the simulator was built from (geometry, medium, reflection model, baked detector
params) to write provenance and to decide whether to run the scintillation pass. Those reads used to be
``getattr(sim, 'medium', None)``, which silently dropped scintillation on a callable without the
attribute. The record makes every field present on every branch, and frozen.
"""
import dataclasses
import json

import jax
import jax.numpy as jnp
import pytest

from lucid.simulation.simulator import Simulator, setup_event_simulator
from tests.helpers.paths import REPO


def _record():
    return Simulator(fn=lambda x, k: x * 2.0, det_geom=None, reflection_model='scalar_mix',
                     default_detector_params=None, emitter=None)


def test_calling_the_record_calls_its_function():
    assert float(_record()(jnp.float32(3.0), None)) == 6.0


def test_the_record_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _record().reflection_model = 'fresnel'


def test_the_record_hashes_by_identity_and_can_be_a_static_jit_argument():
    """DetectorParams holds arrays, so a field-wise hash would raise; jit needs a hashable static."""
    a, b = _record(), _record()
    assert hash(a) != hash(b) and a != b and a == a
    f = jax.jit(lambda s, x: s(x, None), static_argnums=0)
    assert float(f(a, jnp.float32(1.5))) == 3.0


@pytest.mark.slow
@pytest.mark.parametrize('baked', [True, False])
def test_every_branch_records_what_it_was_built_from(baked):
    geom = str(REPO / 'config/SK_like_geom_config.json')
    phys = str(REPO / 'config/SK_like_physics_config.json')
    sim = setup_event_simulator(geom, n_photons=0, K=4, source='photons', temperature=0.0,
                                charge_resolution=None, physics_config=phys,
                                default_detector_params=baked)
    assert isinstance(sim, Simulator)
    assert sim.det_geom.medium.material == json.load(open(geom))['material']
    assert sim.det_geom.num_sensors == len(sim.det_geom.detector.all_points)
    assert isinstance(sim.reflection_model, str) and sim.reflection_model
    assert (sim.default_detector_params is not None) == baked


@pytest.mark.parametrize('procs', [(), ('cherenkov', 'mie'), ('Cherenkov',)])
def test_an_emission_override_outside_the_two_processes_raises_before_any_build(procs):
    with pytest.raises(ValueError, match='emission_processes'):
        setup_event_simulator('no/such/geometry.json', source='particle', emission_processes=procs)


@pytest.mark.slow
def test_the_emission_override_replaces_the_materials_processes():
    """The two JUNO figures predicted one process at a time by monkeypatching make_medium; the
    override must land where the track mode reads it (det_geom.medium), and only there."""
    geom = str(REPO / 'config/SK_like_geom_config.json')
    phys = str(REPO / 'config/SK_like_physics_config.json')
    kw = dict(n_photons=0, K=4, source='photons', temperature=0.0, physics_config=phys)
    own = setup_event_simulator(geom, **kw)
    over = setup_event_simulator(geom, emission_processes=['scintillation'], **kw)
    assert own.det_geom.medium.emission_processes == ('cherenkov',)
    assert over.det_geom.medium.emission_processes == ('scintillation',)
    assert over.det_geom.medium._replace(emission_processes=('cherenkov',)) == own.det_geom.medium
