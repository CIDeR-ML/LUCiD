"""V3 four-file HDF5 readers (sensor / inst / seg / labl).

Standalone module — no internal LUCiD dependencies beyond ``h5py``.
"""
from __future__ import annotations

import h5py
import numpy as np


def list_events_v3(filename):
    """Return the ``config/source_event_idx`` array from a v3 file."""
    with h5py.File(filename, 'r') as f:
        return np.asarray(f['config/source_event_idx'][:])


def _v3_group_to_dict(grp):
    """Recursively copy attrs + datasets + subgroups into a plain dict."""
    out = {}
    for key, value in grp.attrs.items():
        out[key] = value
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:  # subgroup
            out[key] = _v3_group_to_dict(item)
    return out


def _read_v3_event(filename, event_idx):
    """Return the event_NNN/ group contents as a dict keyed by dataset/attr name."""
    with h5py.File(filename, 'r') as f:
        name = f'event_{int(event_idx):03d}'
        if name not in f:
            raise KeyError(
                f"Event group {name!r} not found in {filename}. "
                f"Available: {sorted(k for k in f.keys() if k.startswith('event_'))[:5]} ...")
        return _v3_group_to_dict(f[name])


def read_sensor_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a sensor v3 file."""
    return _read_v3_event(filename, event_idx)


def read_inst_event_v3(filename, event_idx):
    """Read event ``event_idx`` from an inst v3 file."""
    return _read_v3_event(filename, event_idx)


def read_seg_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a seg v3 file."""
    return _read_v3_event(filename, event_idx)


def read_labl_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a labl v5 file.

    The returned dict contains top-level attrs plus four subdicts:
    ``per_event`` (contained, t0 = min per_interaction/t0),
    ``per_interaction`` (source_type, t0, vertex_{x,y,z}, n_primaries,
    n_particles, neutrino_pdg, neutrino_energy_MeV, contained, and
    CSR-encoded primary_{track_ids,pdgs,energies}_{offsets,data}),
    ``per_particle`` (category, contained, genealogy CSR,
    interaction_idx), and ``per_track`` (track_id, parent_id, pdg,
    initial_energy, n_cherenkov, particle_idx, ancestor, interaction).
    """
    return _read_v3_event(filename, event_idx)
