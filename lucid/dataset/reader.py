"""The dataset reader: events of the four files, in their written order.

Needs only ``h5py`` and numpy.
"""
from __future__ import annotations

import h5py
import numpy as np

__all__ = [
    "iter_events",
    "list_events",
    "read_sensor_event",
    "read_hits_event",
    "read_step_event",
    "read_labl_event",
]


def iter_events(filename):
    """``(seq_idx, group_name)`` of every ``event_NNN`` group in ``filename``, in numeric order.

    The groups are named ``event_{i:03d}``, so past 999 events a lexicographic sort puts
    ``event_1000`` before ``event_101``; this one does not.
    """
    with h5py.File(filename, 'r') as f:
        names = [k for k in f.keys() if k.startswith('event_')]
    for name in sorted(names, key=lambda k: int(k[len('event_'):])):
        yield int(name[len('event_'):]), name


def list_events(filename):
    """Return the ``config/source_event_idx`` array from a file."""
    with h5py.File(filename, 'r') as f:
        return np.asarray(f['config/source_event_idx'][:])


def _group_to_dict(grp):
    """Recursively copy attrs + datasets + subgroups into a plain dict."""
    out = {}
    for key, value in grp.attrs.items():
        out[key] = value
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:  # subgroup
            out[key] = _group_to_dict(item)
    return out


def _read_event(filename, event_idx):
    """Return the event_NNN/ group contents as a dict keyed by dataset/attr name."""
    with h5py.File(filename, 'r') as f:
        name = f'event_{int(event_idx):03d}'
        if name not in f:
            raise KeyError(
                f"Event group {name!r} not found in {filename}. "
                f"Available: {sorted(k for k in f.keys() if k.startswith('event_'))[:5]} ...")
        return _group_to_dict(f[name])


def read_sensor_event(filename, event_idx):
    """Read event ``event_idx`` from a sensor file."""
    return _read_event(filename, event_idx)


def read_hits_event(filename, event_idx):
    """Read event ``event_idx`` from a hits file."""
    return _read_event(filename, event_idx)


def read_step_event(filename, event_idx):
    """Read event ``event_idx`` from a step file."""
    return _read_event(filename, event_idx)


def read_labl_event(filename, event_idx):
    """Read event ``event_idx`` from a labl file.

    The returned dict contains top-level attrs plus four subdicts:
    ``per_event`` (contained, t0 = min per_interaction/t0),
    ``per_interaction`` (source_type, t0, vertex_{x,y,z}, n_primaries,
    n_particles, neutrino_pdg, neutrino_energy_MeV, contained, and
    CSR-encoded primary_{track_ids,pdgs,energies}_{offsets,data}),
    ``per_particle`` (category, contained, genealogy CSR,
    interaction_idx), and ``per_track`` (track_id, parent_id, pdg,
    initial_energy, n_cherenkov, particle_idx, ancestor, interaction).
    Triggered datasets additionally carry ``per_window`` (window_start,
    window_end, digit_offsets — the CSR into sensor.h5 digits).
    """
    return _read_event(filename, event_idx)
