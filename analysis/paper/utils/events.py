"""Shared event preparation for the paper figures.

Thin wrappers over the tracking pipeline's own machinery so the paper figures load
photons and build the data-mode photon dict through the SAME code path as production
(units and conventions match exactly — the reader returns origins in cm, and the photon
dict comes from TrackingPipeline._pad_event). Only a display-time rotation is layered on.
"""
import math
import types

import numpy as np


def dir_from_angles(theta, phi):
    """Unit direction from polar angle theta (from +z) and azimuth phi."""
    return np.array([math.sin(theta) * math.cos(phi),
                     math.sin(theta) * math.sin(phi), math.cos(theta)])


def rotation(d0, d1):
    """(axis, angle) of the rotation taking unit vector d0 onto unit vector d1."""
    d0 = np.asarray(d0, float); d0 = d0 / np.linalg.norm(d0)
    d1 = np.asarray(d1, float); d1 = d1 / np.linalg.norm(d1)
    axis = np.cross(d0, d1)
    s = np.linalg.norm(axis)
    if s < 1e-9:                                           # already aligned (or antiparallel)
        return np.array([1., 0., 0.]), 0.0
    return axis / s, float(np.arccos(np.clip(np.dot(d0, d1), -1.0, 1.0)))


def load_event(root, entry):
    """Load a PhotonSim event via the pipeline's reader (origins in cm — the gun-frame
    convention the simulator expects). Returns (raw, energy_MeV, true_direction), where
    true_direction is the photon-origin centroid (PhotonSim fires the primary from 0)."""
    from lucid.sources.event_io import read_photon_data_from_photonsim
    raw = read_photon_data_from_photonsim(str(root), entry)
    energy = float(raw['energy'])
    origins = np.asarray(raw['photon_origins'])
    d0 = origins.mean(0); d0 = d0 / np.linalg.norm(d0)
    return raw, energy, d0


def pad_event(raw, n_photons, rot_axis=None, rot_angle=0.0, translation=None):
    """Data-mode photon dict via TrackingPipeline._pad_event (single source of truth for
    tiling/units/keys). Optional display transforms are layered on top: a rotation (swinging
    the true direction onto the display direction) and a translation in metres (moving the
    vertex), applied by the simulator after the cm->m conversion, rotation before translation."""
    import jax.numpy as jnp
    from analysis.paper.utils.pipeline import TrackingPipeline
    pd = TrackingPipeline._pad_event(types.SimpleNamespace(cfg={'nbuf': n_photons}), raw)
    if rot_angle:
        pd['apply_rotation'] = True
        pd['rotation_axis'] = jnp.asarray(rot_axis, jnp.float32)
        pd['rotation_angle'] = jnp.asarray(rot_angle, jnp.float32)
    if translation is not None and np.any(np.asarray(translation)):
        pd['apply_translation'] = True
        pd['translation_vector'] = jnp.asarray(translation, jnp.float32)
    return pd


def build_track(vertex, direction, energy_MeV, t0=0.0):
    """(vec9, track) for the given vertex/direction/energy via the pipeline's truth9."""
    import jax.numpy as jnp
    from lucid.fitting import track_from_vec9
    from analysis.paper.utils.pipeline import truth9
    th9, _ = truth9(np.asarray(vertex, float), np.asarray(direction, float), energy_MeV, t0)
    return th9, track_from_vec9(jnp.asarray(th9))
