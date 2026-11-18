"""The tiny detector: the composed forward model at a size the default lane can afford.

A 1 m radius, 2 m tall water cylinder with 582 PMTs of 4 cm radius (``tiny_geom_config.json``
beside this file). Building and compiling a simulator on it takes seconds on a CPU, against about
a minute for SK_like, so the session fixtures in ``tests/conftest.py`` build each one once and the
tests that need the whole forward (photon source -> transport -> deposit -> hits) run before every
merge. The 4 cm radius is WCTE_like's, so a soft (temperature > 0) deposit here reads the overlap
table the default lane builds for WCTE_like anyway (``lucid.overlap`` caches it per radius; a new
radius costs about a minute on a fresh checkout).

Every simulator here runs the step-function deposit (``temperature=None``), which is bounded to
the leg the photon travels, so a photon deposits on a PMT exactly when its path passes through the
PMT's sphere before it leaves the detector. That is what makes the charge
computable directly by a ray trace (``test_forward_composed.py``).
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

GEOM = str(Path(__file__).with_name('tiny_geom_config.json'))
N = 4096          # rays per calibration event
QE = 0.2


def simulator(n_photons=N, **setup):
    """A calibration simulator on the tiny detector, one transport step unless ``K`` is given."""
    from lucid.simulation import setup_event_simulator
    kw = dict(temperature=None, K=1, source='light_source', wavelength_mode=False)
    return setup_event_simulator(GEOM, n_photons, **{**kw, **setup})


def data_simulator(**setup):
    """A given-photons simulator: photons given at call time, charge = QE x deposited weight."""
    from lucid.simulation import setup_event_simulator
    kw = dict(K=1, source='photons', detection='expected', temperature=None, wavelength_mode=False)
    return setup_event_simulator(GEOM, 0, **{**kw, **setup})


def transparent(num_sensors, **flat):
    """Water with no scattering or absorption, black walls and non-reflecting PMTs, QE 0.2.

    Any of those can be overridden by its flat ``DetectorParams`` name.
    """
    from lucid.detector_params import DetectorParams
    base = dict(scatter_length=1e9, mie_scatter_length=1e9, absorption_length=1e9,
                wall_reflection_rate=0.0, sensor_reflection_rate=0.0, qe=QE,
                qe_corrections=jnp.ones(num_sensors))
    return DetectorParams.from_flat(**{**base, **flat})


def barrel_target(det):
    """The mid-height barrel PMT: its index, its centre, and the outward unit normal there."""
    barrel = np.asarray(det.barr_points)
    target = int(np.argmin(np.abs(barrel[:, 2])))      # barrel PMTs lead all_points
    x, y, _ = barrel[target]
    return target, barrel[target], np.array([x, y, 0.0]) / np.hypot(x, y)


def photon_data(origins, directions):
    """The data-mode photon dict for explicit photons at t = 0, 400 nm."""
    n = len(origins)
    return dict(
        photon_origins=jnp.asarray(origins, jnp.float32),
        photon_directions=jnp.asarray(directions, jnp.float32),
        photon_times=jnp.zeros(n), wavelengths=jnp.full(n, 400.0), N=jnp.int32(n))
