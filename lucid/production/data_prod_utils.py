"""Notebook/scripting helpers for loading v3 LUCiD dataset batches.

The notebooks under ``lucid/production/notebooks/`` import from this module.
Functions mirror the old API names (``read_multi_event_file``,
``get_track_hits``, ``get_particle_name``, ``print_event_info``) but read
the four-file v3 layout documented in ``docs/LUCID_DATASET.md``.
"""

from pathlib import Path
import numpy as np

from lucid.sources.event_io import (
    read_sensor_event_v3,
    read_inst_event_v3,
    read_seg_event_v3,
    read_labl_event_v3,
    list_events_v3,
)


PDG_NAMES = {
    11: 'e-', -11: 'e+',
    13: 'mu-', -13: 'mu+',
    22: 'gamma',
    111: 'pi0', 211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    2212: 'proton', -2212: 'antiproton',
    2112: 'neutron', -2112: 'antineutron',
}


CATEGORY_NAMES = {
    0: 'Primary', 1: 'DecayElectron', 2: 'SecondaryPion', 3: 'Gamma',
    255: 'Unknown', -1: 'Unknown',
}


def get_particle_name(pdg):
    return PDG_NAMES.get(int(pdg), f'PDG{int(pdg)}')


def _per_particle_pdg(labl):
    """Best-effort PDG per particle, from the last track_id in its genealogy."""
    pp = labl['per_particle']
    pt = labl['per_track']
    genealogy = np.asarray(pp['genealogy_data'])
    offsets = np.asarray(pp['genealogy_offsets'])
    n_p = int(labl['n_particles'])
    n_t = int(labl['n_tracks'])
    track_ids = np.asarray(pt['track_id']) if n_t > 0 else np.array([], dtype=np.int64)
    track_pdgs = np.asarray(pt['pdg']) if n_t > 0 else np.array([], dtype=np.int32)
    id_to_pdg = {int(tid): int(pdg) for tid, pdg in zip(track_ids, track_pdgs)}
    out = np.full(n_p, -1, dtype=np.int32)
    for i in range(n_p):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e > s:
            out[i] = id_to_pdg.get(int(genealogy[e - 1]), -1)
    return out


def _infer_n_sensors(sensor_file):
    import h5py
    with h5py.File(sensor_file, 'r') as f:
        return int(f['config'].attrs['n_sensors'])


def _resolve_dataset_paths(dataset_root_or_sensor_file, file_index):
    """Accept either a dataset root directory or a direct sensor file path."""
    p = Path(dataset_root_or_sensor_file)
    if p.is_file() and p.suffix == '.h5':
        stem = p.stem
        try:
            file_index = int(stem.split('_')[-1])
        except ValueError:
            pass
        root = p.parent.parent
    else:
        root = p
    sensor = root / 'sensor' / f'wc_sensor_{file_index:04d}.h5'
    inst = root / 'inst' / f'wc_inst_{file_index:04d}.h5'
    seg = root / 'seg' / f'wc_seg_{file_index:04d}.h5'
    labl = root / 'labl' / f'wc_labl_{file_index:04d}.h5'
    return sensor, inst, seg, labl, file_index


def load_event_v3(dataset_root, event_idx, file_index=0, n_sensors=None):
    """Load a single event from a v3 dataset batch and return a dict.

    The returned dict exposes dense ``(n_particles, n_sensors)`` PE/T
    matrices under keys ``Q`` and ``T`` for notebook compatibility. Raw v3
    reader outputs are also included under ``labl`` and ``seg`` for anyone
    who needs them.
    """
    sensor_p, inst_p, seg_p, labl_p, file_index = _resolve_dataset_paths(
        dataset_root, file_index)

    if n_sensors is None:
        n_sensors = _infer_n_sensors(sensor_p)

    sensor = read_sensor_event_v3(str(sensor_p), event_idx)
    inst = read_inst_event_v3(str(inst_p), event_idx)
    seg = read_seg_event_v3(str(seg_p), event_idx)
    labl = read_labl_event_v3(str(labl_p), event_idx)

    n_particles = int(labl['n_particles'])

    PE_per_particle = np.zeros((n_particles, n_sensors), dtype=np.float32)
    T_per_particle = np.full((n_particles, n_sensors), np.inf, dtype=np.float32)
    if int(inst.get('n_particle_hits', 0)) > 0:
        pi = np.asarray(inst['particle_idx'], dtype=np.int32)
        si = np.asarray(inst['sensor_idx'], dtype=np.int32)
        PE_per_particle[pi, si] = np.asarray(inst['PE'], dtype=np.float32)
        t_arr = np.asarray(inst['T'], dtype=np.float32)
        T_per_particle[pi, si] = np.where(t_arr > 0, t_arr, np.inf)

    PE = np.zeros(n_sensors, dtype=np.float32)
    T = np.zeros(n_sensors, dtype=np.float32)
    if int(sensor.get('n_hits', 0)) > 0:
        si_s = np.asarray(sensor['sensor_idx'], dtype=np.int32)
        PE[si_s] = np.asarray(sensor['PE'], dtype=np.float32)
        T[si_s] = np.asarray(sensor['T'], dtype=np.float32)

    return {
        'source_event_idx': int(sensor['source_event_idx']),
        'n_particles': n_particles,
        'Q': PE_per_particle,
        'T': T_per_particle,
        'Q_tot': PE,
        'T_tot': T,
        # t0 is now per-interaction. For single-interaction events every
        # row carries the same value; this tool picks the first.
        't0': float(labl['per_interaction']['t0'][0]),
        'edep_containment': float(labl['per_event']['edep_containment']),
        'PDG': _per_particle_pdg(labl),
        'Particle_Category': np.asarray(labl['per_particle']['category']),
        'edep_containment_per_particle': np.asarray(labl['per_particle']['edep_containment']),
        'labl': labl,
        'seg': seg,
    }


def get_track_hits(event, track_idx):
    """Return (indices, Q, T) for one particle row of an event dict.

    ``track_idx`` is the local particle index (0..n_particles-1). The name
    is kept for backwards compatibility with older notebooks; in v3 these
    rows correspond to Geant4-categorized particles, not individual tracks.
    """
    Q_row = event['Q'][track_idx]
    T_row = event['T'][track_idx]
    mask = Q_row > 0
    indices = np.where(mask)[0]
    return indices, Q_row[mask], T_row[mask]


def print_event_info(event):
    cat = event['Particle_Category']
    cat_names = [CATEGORY_NAMES.get(int(c), f'C{int(c)}') for c in cat]
    pdg = event['PDG']
    pdg_names = [get_particle_name(p) if int(p) != -1 else '?' for p in pdg]
    print(f"Event source_event_idx={event['source_event_idx']}, "
          f"n_particles={event['n_particles']}, t0={event['t0']:.2f} ns, "
          f"total PE={event['Q_tot'].sum():.1f}")
    for i in range(event['n_particles']):
        cont = event['edep_containment_per_particle'][i]
        cont_str = f"{cont*100:.1f}%" if np.isfinite(cont) else "n/a"
        print(f"  particle {i}: category={cat_names[i]}, pdg={pdg_names[i]}, "
              f"edep_containment={cont_str}")


def read_multi_event_file(dataset_root, file_index=0, verbose=False, n_sensors=None):
    """Return a list of event dicts for an entire v3 batch file.

    ``dataset_root`` is the directory containing ``sensor/``, ``inst/``,
    ``seg/``, ``labl/`` subdirectories. A direct sensor file path is also
    accepted; in that case ``file_index`` is parsed from the filename.
    """
    sensor_file, _, _, _, file_index = _resolve_dataset_paths(dataset_root, file_index)
    source_event_idx = list_events_v3(str(sensor_file))
    n_events = len(source_event_idx)
    if n_sensors is None:
        n_sensors = _infer_n_sensors(sensor_file)
    events = []
    for i in range(n_events):
        ev = load_event_v3(dataset_root, i, file_index=file_index, n_sensors=n_sensors)
        events.append(ev)
        if verbose:
            print_event_info(ev)
    return events
