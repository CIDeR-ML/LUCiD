"""HDF5 IO for photon-shotgun outputs.

Two output modes:

* ``waveform`` — dense ``(n_cases, num_sensors, n_time_bins)`` histograms are
  sparsified to COO ``(case_idx, sensor_id, time_bin, charge)`` and stored with
  chunked gzip. For very large runs (10k+ cases) use ``StreamingWaveformWriter``
  which sparsifies and appends each chunk incrementally.

* ``per_photon`` — dense ``(n_cases, n_photons)`` arrays of ``detected``,
  ``sensor_id``, ``hit_time`` stored with chunked gzip.
"""
from typing import Optional

import h5py
import numpy as np

from lucid.sources.shotgun_source import ShotgunSource


# ---------------------------------------------------------------------------
# COO sparsify / densify
# ---------------------------------------------------------------------------

def sparsify_waveform(waveform: np.ndarray, threshold: float = 0.0):
    """Convert dense ``(n_cases, num_sensors, n_time_bins)`` waveform to COO.

    Returns ``(case_idx, sensor_id, time_bin, charge, entries_per_case)``
    where ``entries_per_case`` is a CSR-style ``(n_cases + 1,)`` offset array.
    """
    wf = np.asarray(waveform)
    if wf.ndim == 2:
        wf = wf[None]
    n_cases = wf.shape[0]

    mask = wf > threshold
    case_idx, sensor_id, time_bin = np.nonzero(mask)
    charge = wf[case_idx, sensor_id, time_bin]

    counts = np.bincount(case_idx, minlength=n_cases)
    entries_per_case = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    return (case_idx.astype(np.int32), sensor_id.astype(np.int32),
            time_bin.astype(np.int32), charge.astype(np.float32),
            entries_per_case)


def densify_waveform(case_idx, sensor_id, time_bin, charge,
                     n_cases: int, num_sensors: int, n_time_bins: int) -> np.ndarray:
    wf = np.zeros((n_cases, num_sensors, n_time_bins), dtype=np.float32)
    wf[case_idx, sensor_id, time_bin] = charge
    return wf


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _write_meta(grp, meta: dict):
    for k, v in meta.items():
        if isinstance(v, str):
            grp.attrs[k] = v
        else:
            grp.attrs[k] = np.asarray(v)


def _write_source(grp, source: Optional[ShotgunSource]):
    if source is None:
        return
    grp.create_dataset('origins', data=np.asarray(source.origins),
                       compression='gzip', compression_opts=4)
    grp.create_dataset('directions', data=np.asarray(source.directions),
                       compression='gzip', compression_opts=4)
    grp.create_dataset('intensities', data=np.asarray(source.intensities),
                       compression='gzip', compression_opts=4)
    if source.wavelength is not None:
        grp.create_dataset('wavelength', data=np.asarray(source.wavelength),
                           compression='gzip', compression_opts=4)


def _read_source(grp) -> Optional[ShotgunSource]:
    if 'origins' not in grp:
        return None
    origins = grp['origins'][:]
    directions = grp['directions'][:]
    intensities = grp['intensities'][:]
    wavelength = grp['wavelength'][:] if 'wavelength' in grp else None
    return ShotgunSource(origins=origins, directions=directions,
                         intensities=intensities, wavelength=wavelength)


# ---------------------------------------------------------------------------
# One-shot save/load (small runs, fits in memory)
# ---------------------------------------------------------------------------

def save_shotgun_waveform(
    path: str,
    waveform: np.ndarray,
    n_dropped: np.ndarray,
    n_detected: np.ndarray,
    *,
    window_ns: float,
    bin_width_ns: float,
    tts_sigma_ns: float,
    source: Optional[ShotgunSource] = None,
    detector_config: str = '',
    physics_config: str = '',
    threshold: float = 0.0,
    save_source: bool = True,
):
    """Save waveform-mode outputs to HDF5 (sparsified)."""
    wf = np.asarray(waveform)
    if wf.ndim == 2:
        wf = wf[None]
    n_cases, num_sensors, n_time_bins = wf.shape

    case_idx, sensor_id, time_bin, charge, offsets = sparsify_waveform(wf, threshold=threshold)
    n_dropped = np.atleast_1d(np.asarray(n_dropped)).astype(np.int32)
    n_detected = np.atleast_1d(np.asarray(n_detected)).astype(np.int32)

    with h5py.File(path, 'w') as f:
        _write_meta(f.create_group('meta'), {
            'mode': 'waveform',
            'n_cases': n_cases,
            'num_sensors': num_sensors,
            'n_time_bins': n_time_bins,
            'window_ns': float(window_ns),
            'bin_width_ns': float(bin_width_ns),
            'tts_sigma_ns': float(tts_sigma_ns),
            'detector_config': detector_config,
            'physics_config': physics_config,
        })
        meas = f.create_group('measured')
        for name, data in (('case_idx', case_idx), ('sensor_id', sensor_id),
                           ('time_bin', time_bin), ('charge', charge)):
            meas.create_dataset(name, data=data, compression='gzip',
                                compression_opts=4, chunks=True)
        meas.create_dataset('entries_per_case', data=offsets)

        f.create_dataset('n_dropped', data=n_dropped)
        f.create_dataset('n_detected', data=n_detected)

        if save_source:
            _write_source(f.create_group('source'), source)


def load_shotgun_waveform(path: str, dense: bool = False):
    out = {}
    with h5py.File(path, 'r') as f:
        out['meta'] = dict(f['meta'].attrs)
        meas = f['measured']
        out['case_idx'] = meas['case_idx'][:]
        out['sensor_id'] = meas['sensor_id'][:]
        out['time_bin'] = meas['time_bin'][:]
        out['charge'] = meas['charge'][:]
        out['entries_per_case'] = meas['entries_per_case'][:]
        out['n_dropped'] = f['n_dropped'][:]
        out['n_detected'] = f['n_detected'][:]
        out['source'] = _read_source(f['source']) if 'source' in f else None

    if dense:
        meta = out['meta']
        out['waveform'] = densify_waveform(
            out['case_idx'], out['sensor_id'], out['time_bin'], out['charge'],
            n_cases=int(meta['n_cases']), num_sensors=int(meta['num_sensors']),
            n_time_bins=int(meta['n_time_bins']))
    return out


def save_shotgun_per_photon(
    path: str,
    detected: np.ndarray,
    sensor_id: np.ndarray,
    hit_time: np.ndarray,
    *,
    tts_sigma_ns: float,
    source: Optional[ShotgunSource] = None,
    detector_config: str = '',
    physics_config: str = '',
    save_source: bool = True,
):
    det = np.atleast_2d(np.asarray(detected))
    sid = np.atleast_2d(np.asarray(sensor_id))
    ht = np.atleast_2d(np.asarray(hit_time))
    n_cases, n_photons = det.shape

    with h5py.File(path, 'w') as f:
        _write_meta(f.create_group('meta'), {
            'mode': 'per_photon',
            'n_cases': n_cases,
            'n_photons': n_photons,
            'tts_sigma_ns': float(tts_sigma_ns),
            'detector_config': detector_config,
            'physics_config': physics_config,
        })
        pp = f.create_group('per_photon')
        pp.create_dataset('detected', data=det.astype(np.bool_),
                          compression='gzip', compression_opts=4, chunks=True)
        pp.create_dataset('sensor_id', data=sid.astype(np.int32),
                          compression='gzip', compression_opts=4, chunks=True)
        pp.create_dataset('hit_time', data=ht.astype(np.float32),
                          compression='gzip', compression_opts=4, chunks=True)

        if save_source:
            _write_source(f.create_group('source'), source)


def load_shotgun_per_photon(path: str) -> dict:
    out = {}
    with h5py.File(path, 'r') as f:
        out['meta'] = dict(f['meta'].attrs)
        pp = f['per_photon']
        out['detected'] = pp['detected'][:]
        out['sensor_id'] = pp['sensor_id'][:]
        out['hit_time'] = pp['hit_time'][:]
        out['source'] = _read_source(f['source']) if 'source' in f else None
    return out


# ---------------------------------------------------------------------------
# Streaming writers for large (10k+ case) runs
# ---------------------------------------------------------------------------

class StreamingWaveformWriter:
    """Append sparsified chunks to an HDF5 file without holding the dense
    ``(n_cases, num_sensors, n_time_bins)`` tensor in memory.

    Usage::

        with StreamingWaveformWriter(path, num_sensors, n_time_bins, wf_cfg) as w:
            for chunk in run_chunks():
                wf_chunk, nd_chunk, ndet_chunk, src_chunk = chunk
                w.append(wf_chunk, nd_chunk, ndet_chunk, src_chunk)
    """

    def __init__(self, path, *,
                 num_sensors: int, n_time_bins: int,
                 waveform_config: dict,
                 detector_config: str = '', physics_config: str = '',
                 n_photons: int = 0, K: int = 0,
                 save_source: bool = True,
                 threshold: float = 0.0):
        self.path = str(path)
        self.threshold = float(threshold)
        self.save_source = bool(save_source)
        self._case_offset = 0
        self._n_photons = int(n_photons)
        self._num_sensors = int(num_sensors)
        self._n_time_bins = int(n_time_bins)

        self.f = h5py.File(self.path, 'w')
        _write_meta(self.f.create_group('meta'), {
            'mode': 'waveform',
            'num_sensors': num_sensors,
            'n_time_bins': n_time_bins,
            'n_photons': n_photons,
            'K': K,
            'window_ns': float(waveform_config['window_ns']),
            'bin_width_ns': float(waveform_config['bin_width_ns']),
            'tts_sigma_ns': float(waveform_config['tts_sigma_ns']),
            'smear_time': int(bool(waveform_config.get('smear_time', True))),
            'smear_charge': int(bool(waveform_config.get('smear_charge', True))),
            'detector_config': detector_config,
            'physics_config': physics_config,
        })

        meas = self.f.create_group('measured')
        chunk_entries = 1 << 16
        self._case_idx = meas.create_dataset(
            'case_idx', shape=(0,), maxshape=(None,),
            chunks=(chunk_entries,), dtype=np.int32,
            compression='gzip', compression_opts=4)
        self._sensor_id = meas.create_dataset(
            'sensor_id', shape=(0,), maxshape=(None,),
            chunks=(chunk_entries,), dtype=np.int32,
            compression='gzip', compression_opts=4)
        self._time_bin = meas.create_dataset(
            'time_bin', shape=(0,), maxshape=(None,),
            chunks=(chunk_entries,), dtype=np.int32,
            compression='gzip', compression_opts=4)
        self._charge = meas.create_dataset(
            'charge', shape=(0,), maxshape=(None,),
            chunks=(chunk_entries,), dtype=np.float32,
            compression='gzip', compression_opts=4)

        self._n_dropped = self.f.create_dataset(
            'n_dropped', shape=(0,), maxshape=(None,), dtype=np.int32)
        self._n_detected = self.f.create_dataset(
            'n_detected', shape=(0,), maxshape=(None,), dtype=np.int32)

        self._source_grp = None
        self._src_origins = self._src_directions = None
        self._src_intensities = self._src_wavelength = None

    # per-case offsets are reconstructable from (case_idx, n_cases) so we skip
    # storing them incrementally; loader can rebuild via np.bincount.

    def _init_source_dsets(self, sample: ShotgunSource):
        grp = self.f.create_group('source')
        n_ph = int(sample.origins.shape[-2])
        self._src_origins = grp.create_dataset(
            'origins', shape=(0, n_ph, 3), maxshape=(None, n_ph, 3),
            chunks=(max(1, min(64, 4096 // max(n_ph, 1))), n_ph, 3),
            dtype=np.float32, compression='gzip', compression_opts=4)
        self._src_directions = grp.create_dataset(
            'directions', shape=(0, n_ph, 3), maxshape=(None, n_ph, 3),
            chunks=(max(1, min(64, 4096 // max(n_ph, 1))), n_ph, 3),
            dtype=np.float32, compression='gzip', compression_opts=4)
        self._src_intensities = grp.create_dataset(
            'intensities', shape=(0, n_ph), maxshape=(None, n_ph),
            chunks=(max(1, min(256, 65536 // max(n_ph, 1))), n_ph),
            dtype=np.float32, compression='gzip', compression_opts=4)
        if sample.wavelength is not None:
            wl = np.asarray(sample.wavelength)
            wl_shape = wl.shape[1:] if wl.ndim > 1 else ()
            self._src_wavelength = grp.create_dataset(
                'wavelength', shape=(0,) + wl_shape,
                maxshape=(None,) + wl_shape,
                dtype=np.float32, compression='gzip', compression_opts=4)
        self._source_grp = grp

    def append(self, waveform_chunk, n_dropped_chunk, n_detected_chunk,
               source_chunk: Optional[ShotgunSource] = None):
        wf = np.asarray(waveform_chunk)
        if wf.ndim == 2:
            wf = wf[None]
        n_cases = wf.shape[0]

        ci, sid, tb, ch, _ = sparsify_waveform(wf, threshold=self.threshold)
        ci_global = ci + np.int32(self._case_offset)
        cur = self._case_idx.shape[0]
        self._case_idx.resize(cur + ci.size, axis=0)
        self._sensor_id.resize(cur + ci.size, axis=0)
        self._time_bin.resize(cur + ci.size, axis=0)
        self._charge.resize(cur + ci.size, axis=0)
        self._case_idx[cur:cur + ci.size] = ci_global
        self._sensor_id[cur:cur + ci.size] = sid
        self._time_bin[cur:cur + ci.size] = tb
        self._charge[cur:cur + ci.size] = ch

        nd = np.atleast_1d(np.asarray(n_dropped_chunk)).astype(np.int32)
        ndet = np.atleast_1d(np.asarray(n_detected_chunk)).astype(np.int32)
        cur_c = self._n_dropped.shape[0]
        self._n_dropped.resize(cur_c + nd.size, axis=0)
        self._n_detected.resize(cur_c + ndet.size, axis=0)
        self._n_dropped[cur_c:cur_c + nd.size] = nd
        self._n_detected[cur_c:cur_c + ndet.size] = ndet

        if self.save_source and source_chunk is not None:
            if self._source_grp is None:
                self._init_source_dsets(source_chunk)
            self._append_source(source_chunk)

        self._case_offset += n_cases

    def _append_source(self, src: ShotgunSource):
        for dset, arr in ((self._src_origins, src.origins),
                          (self._src_directions, src.directions),
                          (self._src_intensities, src.intensities)):
            arr = np.asarray(arr)
            cur = dset.shape[0]
            dset.resize(cur + arr.shape[0], axis=0)
            dset[cur:cur + arr.shape[0]] = arr
        if self._src_wavelength is not None and src.wavelength is not None:
            arr = np.asarray(src.wavelength)
            cur = self._src_wavelength.shape[0]
            self._src_wavelength.resize(cur + arr.shape[0], axis=0)
            self._src_wavelength[cur:cur + arr.shape[0]] = arr

    def close(self):
        if self.f is None:
            return
        self.f['meta'].attrs['n_cases'] = np.int32(self._case_offset)
        # store per-case entry offsets for quick loading
        if self._case_offset > 0 and self._case_idx.shape[0] > 0:
            counts = np.bincount(self._case_idx[:], minlength=self._case_offset)
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            self.f['measured'].create_dataset('entries_per_case', data=offsets)
        self.f.close()
        self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class StreamingPerPhotonWriter:
    """Append per-photon chunks to HDF5 with fixed per-case width ``n_photons``."""

    def __init__(self, path, *,
                 n_photons: int, tts_sigma_ns: float,
                 detector_config: str = '', physics_config: str = '',
                 K: int = 0, save_source: bool = True):
        self.path = str(path)
        self.save_source = bool(save_source)
        self._case_offset = 0
        self._n_photons = int(n_photons)

        self.f = h5py.File(self.path, 'w')
        _write_meta(self.f.create_group('meta'), {
            'mode': 'per_photon',
            'n_photons': n_photons,
            'K': K,
            'tts_sigma_ns': float(tts_sigma_ns),
            'detector_config': detector_config,
            'physics_config': physics_config,
        })
        pp = self.f.create_group('per_photon')
        chunk_cases = max(1, min(256, 65536 // max(n_photons, 1)))
        self._detected = pp.create_dataset(
            'detected', shape=(0, n_photons), maxshape=(None, n_photons),
            chunks=(chunk_cases, n_photons), dtype=np.bool_,
            compression='gzip', compression_opts=4)
        self._sensor_id = pp.create_dataset(
            'sensor_id', shape=(0, n_photons), maxshape=(None, n_photons),
            chunks=(chunk_cases, n_photons), dtype=np.int32,
            compression='gzip', compression_opts=4)
        self._hit_time = pp.create_dataset(
            'hit_time', shape=(0, n_photons), maxshape=(None, n_photons),
            chunks=(chunk_cases, n_photons), dtype=np.float32,
            compression='gzip', compression_opts=4)

        self._source_grp = None
        self._src_origins = self._src_directions = None
        self._src_intensities = self._src_wavelength = None

    def _init_source_dsets(self, sample: ShotgunSource):
        grp = self.f.create_group('source')
        n_ph = self._n_photons
        self._src_origins = grp.create_dataset(
            'origins', shape=(0, n_ph, 3), maxshape=(None, n_ph, 3),
            chunks=(max(1, min(64, 4096 // max(n_ph, 1))), n_ph, 3),
            dtype=np.float32, compression='gzip', compression_opts=4)
        self._src_directions = grp.create_dataset(
            'directions', shape=(0, n_ph, 3), maxshape=(None, n_ph, 3),
            chunks=(max(1, min(64, 4096 // max(n_ph, 1))), n_ph, 3),
            dtype=np.float32, compression='gzip', compression_opts=4)
        self._src_intensities = grp.create_dataset(
            'intensities', shape=(0, n_ph), maxshape=(None, n_ph),
            chunks=(max(1, min(256, 65536 // max(n_ph, 1))), n_ph),
            dtype=np.float32, compression='gzip', compression_opts=4)
        if sample.wavelength is not None:
            wl = np.asarray(sample.wavelength)
            wl_shape = wl.shape[1:] if wl.ndim > 1 else ()
            self._src_wavelength = grp.create_dataset(
                'wavelength', shape=(0,) + wl_shape,
                maxshape=(None,) + wl_shape,
                dtype=np.float32, compression='gzip', compression_opts=4)
        self._source_grp = grp

    def append(self, detected, sensor_id, hit_time,
               source_chunk: Optional[ShotgunSource] = None):
        det = np.atleast_2d(np.asarray(detected))
        sid = np.atleast_2d(np.asarray(sensor_id))
        ht = np.atleast_2d(np.asarray(hit_time))
        n_cases = det.shape[0]
        cur = self._detected.shape[0]
        self._detected.resize(cur + n_cases, axis=0)
        self._sensor_id.resize(cur + n_cases, axis=0)
        self._hit_time.resize(cur + n_cases, axis=0)
        self._detected[cur:cur + n_cases] = det.astype(np.bool_)
        self._sensor_id[cur:cur + n_cases] = sid.astype(np.int32)
        self._hit_time[cur:cur + n_cases] = ht.astype(np.float32)

        if self.save_source and source_chunk is not None:
            if self._source_grp is None:
                self._init_source_dsets(source_chunk)
            self._append_source(source_chunk)

        self._case_offset += n_cases

    def _append_source(self, src: ShotgunSource):
        for dset, arr in ((self._src_origins, src.origins),
                          (self._src_directions, src.directions),
                          (self._src_intensities, src.intensities)):
            arr = np.asarray(arr)
            cur = dset.shape[0]
            dset.resize(cur + arr.shape[0], axis=0)
            dset[cur:cur + arr.shape[0]] = arr
        if self._src_wavelength is not None and src.wavelength is not None:
            arr = np.asarray(src.wavelength)
            cur = self._src_wavelength.shape[0]
            self._src_wavelength.resize(cur + arr.shape[0], axis=0)
            self._src_wavelength[cur:cur + arr.shape[0]] = arr

    def close(self):
        if self.f is None:
            return
        self.f['meta'].attrs['n_cases'] = np.int32(self._case_offset)
        self.f.close()
        self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
