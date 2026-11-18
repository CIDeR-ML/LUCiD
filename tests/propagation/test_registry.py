"""The geometry registry, and detector bounds checks under jit and vmap.

Verifies:
- Registry lookup, case insensitivity, error handling
- Registering a class is enough for generate_detector to build it from a JSON config
- bounds_check works inside jax.jit and jax.vmap (it is called inside the JIT-compiled
  propagation loop)

That generate_detector builds the right class with the right sensors is
tests/propagation/test_geometry.py.
Small detectors are built directly here, so the module runs by default.
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry import (get_detector_class, generate_detector, register_detector,
                            Detector, Cylinder, Sphere, Box)
from lucid.geometry import registry
from lucid.geometry.registry import _REGISTRY


class TestRegistryLookup:
    def test_all_types_registered(self):
        assert sorted(_REGISTRY) == ['box', 'cylinder', 'sphere', 'string']

    def test_case_insensitive(self):
        for name, cls in (('cylinder', Cylinder), ('sphere', Sphere), ('box', Box)):
            for spelled in (name, name.capitalize(), name.upper()):
                assert get_detector_class(spelled) is cls

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown detector type"):
            get_detector_class('hexagon')


def test_registered_class_is_built_by_generate_detector(tmp_path, monkeypatch):
    """A class outside the library, once registered, is built from a JSON naming its type."""
    monkeypatch.setattr(registry, '_REGISTRY', dict(registry._REGISTRY))

    @register_detector('toy_ring')
    class ToyRing(Detector):
        def __init__(self, radius, n_sensors, sensor_radius):
            super().__init__(n_sensors, sensor_radius)
            self.r = radius
            self.place_photosensors()

        @classmethod
        def from_config(cls, geometry, config_dir):
            return cls(geometry['radius'], geometry['n_sensors'], geometry['sensor_radius'])

        def place_photosensors(self):
            phi = 2 * np.pi * np.arange(self.n_sensors) / self.n_sensors
            self.all_points = self.r * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], 1)

    cfg = tmp_path / 'toy_geom_config.json'
    cfg.write_text(json.dumps({'detector_type': 'Toy_Ring', 'material': 'water',
                               'geometry_definitions': {'radius': 2.0, 'n_sensors': 8,
                                                        'sensor_radius': 0.1}}))
    det = generate_detector(str(cfg))
    assert type(det) is ToyRing
    assert det.S_radius == 0.1
    npt.assert_allclose(np.linalg.norm(det.all_points, axis=1), 2.0)


def _small(kind):
    return {'cylinder': lambda: Cylinder(2.0, 4.0, 200, 0.04),      # r 2, caps at z = +-2
            'sphere': lambda: Sphere(1.8, 200, 0.05),
            'box': lambda: Box(6.0, 12.0, 5.0, 200, 0.075)}[kind]()  # half 3 x 6 x 2.5


# Inside, far outside, and a point just past each shape's surface along +x (2.0, 1.8, 3.0).
_PTS = {'cylinder': [[0., 0., 0.], [100., 100., 100.], [1.99, 0., 0.], [2.01, 0., 0.]],
        'sphere': [[0., 0., 0.], [100., 100., 100.], [1.79, 0., 0.], [1.81, 0., 0.]],
        'box': [[0., 0., 0.], [100., 100., 100.], [2.99, 0., 0.], [3.01, 0., 0.]]}


@pytest.mark.parametrize("kind", ['cylinder', 'sphere', 'box'])
def test_bounds_check_under_jit_and_vmap(kind):
    det = _small(kind)
    pts = jnp.array(_PTS[kind])
    want = [True, False, True, False]
    npt.assert_array_equal(jax.jit(det.bounds_check)(pts), want)
    single = jax.vmap(lambda p: det.bounds_check(p[None])[0])
    npt.assert_array_equal(single(pts), want)
