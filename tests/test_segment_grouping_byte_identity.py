"""Byte-identity check: aggregate raw seg.h5 by group_id and compare
against the legacy C++-merged seg.h5 produced from the same Geant4
seed. Pass = every per-segment column matches at float32 precision.

This test runs only when both seg.h5 files are present at known
locations (not part of the default pytest sweep — it requires running
PhotonSim twice on the same seed, once with `/photon/emitRawSegments
true` and once with it `false`). Drive end-to-end via:

    LUCID_SEG_BYTE_IDENTITY_RAW=/path/to/raw/seg/wc_seg_0000.h5 \\
    LUCID_SEG_BYTE_IDENTITY_MERGED=/path/to/merged/seg/wc_seg_0000.h5 \\
    pytest tests/test_segment_grouping_byte_identity.py

Or invoke ``aggregate_by_group`` and ``compare_seg_files`` directly
from any other harness.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest


def aggregate_by_group(g: h5py.Group) -> dict:
    """Mirror ``DataManager.cc:514-526``. Aggregate raw rows by group_id.

    - ``start_*`` = first row of each group
    - ``end_*``   = last row of each group
    - ``dir_*``, ``beta_start``, ``time`` = first row's value (preserved
       across merges)
    - ``edep`` = sum (float64 accumulation cast to float32 to mirror C++)
    - ``n_cherenkov`` = sum
    - ``track_idx``, ``contained`` = first row's value (constant within a group)
    """
    n = int(g.attrs['n_segments'])
    if n == 0:
        return {k: np.array([], dtype=g[k].dtype) for k in g.keys()}

    gid = g['group_id'][:]
    n_groups = int(gid.max()) + 1

    is_first = np.r_[True, gid[1:] != gid[:-1]]
    is_last = np.r_[gid[:-1] != gid[1:], True]
    first_idx = np.where(is_first)[0]
    last_idx = np.where(is_last)[0]
    assert len(first_idx) == n_groups and len(last_idx) == n_groups

    out = {}
    for k in ('start_x', 'start_y', 'start_z',
             'dir_x', 'dir_y', 'dir_z',
             'time', 'beta_start',
             'track_idx', 'contained'):
        out[k] = g[k][:][first_idx]
    for k in ('end_x', 'end_y', 'end_z'):
        out[k] = g[k][:][last_idx]
    edep_f64 = g['edep'][:].astype(np.float64)
    out['edep'] = np.add.reduceat(edep_f64, first_idx).astype(np.float32)
    nche = g['n_cherenkov'][:]
    out['n_cherenkov'] = np.add.reduceat(nche, first_idx).astype(np.int32)
    return out


def compare_seg_files(raw_path: str, mgd_path: str,
                      edep_rtol: float = 5e-6) -> tuple[bool, list[str]]:
    """Returns (ok, mismatch_messages).

    Most columns must match bitwise. ``edep`` (and any other column
    that is a *sum* across raw sub-steps — currently only ``edep``)
    gets a small relative tolerance: the merged file holds a float32
    cast of a float64 sum, while aggregating the raw seg.h5 column
    casts each sub-step to float32 first, then sums — the two paths
    can differ by ~1 float32 ULP per merged group. This is a property
    of float32 round-off, not a port bug; ``edep_rtol=5e-6`` (~40
    float32 ULPs) is a comfortable bound.
    """
    msgs: list[str] = []
    with h5py.File(raw_path, 'r') as fr, h5py.File(mgd_path, 'r') as fm:
        ev_names = sorted({k for k in fr.keys() if k.startswith('event_')}
                          & {k for k in fm.keys() if k.startswith('event_')})
        for ev in ev_names:
            raw_g, mgd_g = fr[ev], fm[ev]
            n_raw = int(raw_g.attrs['n_segments'])
            n_mgd = int(mgd_g.attrs['n_segments'])
            agg = aggregate_by_group(raw_g)
            n_agg = len(agg['edep'])
            if n_agg != n_mgd:
                msgs.append(f'{ev}: aggregated count {n_agg} != merged {n_mgd} '
                            f'(raw={n_raw})')
                continue
            # Bitwise-required columns.
            bitwise_keys = ('start_x', 'start_y', 'start_z',
                            'end_x', 'end_y', 'end_z',
                            'dir_x', 'dir_y', 'dir_z',
                            'time', 'beta_start',
                            'n_cherenkov', 'track_idx')
            for k in bitwise_keys:
                a = agg[k]
                b = mgd_g[k][:]
                if a.shape != b.shape:
                    msgs.append(f'{ev}/{k}: shape {a.shape} vs {b.shape}')
                    continue
                if np.issubdtype(a.dtype, np.integer):
                    if not np.array_equal(a, b):
                        ndiff = int((a != b).sum())
                        msgs.append(f'{ev}/{k} (int): {ndiff} mismatches')
                else:
                    if a.tobytes() != b.tobytes():
                        neq = (a != b) | (np.isnan(a) ^ np.isnan(b))
                        ndiff = int(neq.sum())
                        if ndiff > 0:
                            max_abs = float(np.abs(b - a).max())
                            msgs.append(
                                f'{ev}/{k} (float): {ndiff} bitwise mismatches, '
                                f'max |Δ|={max_abs}')
            # edep — relative tolerance.
            a = agg['edep'].astype(np.float64)
            b = mgd_g['edep'][:].astype(np.float64)
            if not np.allclose(a, b, rtol=edep_rtol, atol=0.0):
                neq = ~np.isclose(a, b, rtol=edep_rtol, atol=0.0)
                ndiff = int(neq.sum())
                msgs.append(
                    f'{ev}/edep: {ndiff} rtol={edep_rtol} mismatches, '
                    f'max rel |Δ|={float((np.abs(b-a)/np.maximum(np.abs(b),1e-30)).max()):.2e}')
    return len(msgs) == 0, msgs


_RAW_ENV = 'LUCID_SEG_BYTE_IDENTITY_RAW'
_MGD_ENV = 'LUCID_SEG_BYTE_IDENTITY_MERGED'


@pytest.mark.skipif(
    _RAW_ENV not in os.environ or _MGD_ENV not in os.environ,
    reason=(f'set {_RAW_ENV} and {_MGD_ENV} to run; both must come from '
            'PhotonSim runs sharing the same /random/setSeeds (one with '
            '/photon/emitRawSegments true, one with false).'),
)
def test_aggregate_raw_matches_merged():
    raw = os.environ[_RAW_ENV]
    mgd = os.environ[_MGD_ENV]
    ok, msgs = compare_seg_files(raw, mgd)
    assert ok, '\n'.join(msgs)
