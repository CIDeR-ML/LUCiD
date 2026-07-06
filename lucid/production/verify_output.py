"""Verify that a four-file batch was written correctly.

A successful `lucid-run-job` leaves, for one batch `file_index = F`,
four files under `<output_dir>/{sensor,hits,step,labl}/wc_*_<F:04d>.h5`.
This module asserts:

    1. All four files exist and are non-zero.
    2. Each file opens with h5py.
    3. Each file's `config/` group carries the expected `dataset_name`,
       `file_index`, and non-zero `n_events`.

Returns `(ok, messages)`. Messages is a list of per-file status lines
suitable for logging.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


SUBDIRS = ("sensor", "hits", "step", "labl")


def batch_paths(output_dir: os.PathLike, file_index: int) -> dict[str, Path]:
    """Return the expected four-file paths for a given batch."""
    root = Path(output_dir)
    tag = f"{file_index:04d}"
    return {sub: root / sub / f"wc_{sub}_{tag}.h5" for sub in SUBDIRS}


def _check_digit_invariants(paths, n_sample: int = 3) -> tuple[bool, list[str]]:
    """Sample events and check the cross-file digit_idx FK + per_window CSR.

    - hits.h5 / step.sensor_hits ``digit_idx`` are in range and satisfy
      ``sensor_idx == sensor.h5.sensor_idx[digit_idx]``.
    - ``labl/per_window/digit_offsets`` is a valid CSR over the digit list
      (starts at 0, ends at n_digits, monotonic).
    """
    import h5py
    import numpy as np

    msgs: list[str] = []
    ok = True
    try:
        with h5py.File(paths["sensor"], "r") as sf, h5py.File(paths["hits"], "r") as hf, \
                h5py.File(paths["step"], "r") as gf, h5py.File(paths["labl"], "r") as lf:
            evs = sorted(k for k in sf if k.startswith("event_"))[:n_sample]
            for e in evs:
                si = sf[e]["sensor_idx"][:]
                nd = si.shape[0]
                hd = hf[e]["digit_idx"][:]
                if hd.size and not (hd.min() >= 0 and hd.max() < nd
                                    and (hf[e]["sensor_idx"][:] == si[hd]).all()):
                    msgs.append(f"BAD_DIGIT_FK(hits): {e}"); ok = False
                sh = gf[e].get("sensor_hits")
                if sh is not None and "digit_idx" in sh:
                    sd = sh["digit_idx"][:]
                    if sd.size and not (sd.min() >= 0 and sd.max() < nd
                                        and (sh["sensor_idx"][:] == si[sd]).all()):
                        msgs.append(f"BAD_DIGIT_FK(step): {e}"); ok = False
                pw = lf[e].get("per_window")
                if pw is not None:
                    off = pw["digit_offsets"][:]
                    if not (off[0] == 0 and off[-1] == nd and (np.diff(off) >= 0).all()):
                        msgs.append(f"BAD_PER_WINDOW_CSR: {e} (offsets end {int(off[-1])} != n_digits {nd})")
                        ok = False
            if ok:
                msgs.append(f"OK invariants (digit_idx FK, per_window CSR) on {len(evs)} sampled events")
    except Exception as exc:
        msgs.append(f"INVARIANT_CHECK_ERROR: {exc!r}"); ok = False
    return ok, msgs


def verify_batch(
    output_dir: os.PathLike,
    file_index: int,
    expected_dataset_name: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Check the four files for one batch. Return (ok, messages)."""
    import h5py

    paths = batch_paths(output_dir, file_index)
    messages: list[str] = []
    ok = True

    for sub, path in paths.items():
        if not path.exists():
            messages.append(f"MISSING: {path}")
            ok = False
            continue

        size = path.stat().st_size
        if size == 0:
            messages.append(f"EMPTY:   {path}")
            ok = False
            continue

        try:
            with h5py.File(path, "r") as h:
                cfg_attrs = dict(h["config"].attrs)
                name = cfg_attrs.get("dataset_name", "<missing>")
                fi = int(cfg_attrs.get("file_index", -1))
                n_events = int(cfg_attrs.get("n_events", -1))
        except Exception as e:
            messages.append(f"UNREADABLE: {path}: {e!r}")
            ok = False
            continue

        if fi != file_index:
            messages.append(
                f"BAD_FILE_INDEX: {path} has file_index={fi}, expected {file_index}"
            )
            ok = False

        if n_events <= 0:
            messages.append(f"BAD_N_EVENTS: {path} has n_events={n_events}")
            ok = False

        if expected_dataset_name is not None and name != expected_dataset_name:
            messages.append(
                f"BAD_DATASET_NAME: {path} has {name!r}, expected {expected_dataset_name!r}"
            )
            ok = False

        messages.append(
            f"OK {sub:<6} size={size:>10} dataset_name={name!r} file_index={fi} n_events={n_events}"
        )

    # Cross-file schema invariants (only when the four files are all readable).
    if ok:
        inv_ok, inv_msgs = _check_digit_invariants(paths)
        messages.extend(inv_msgs)
        ok = ok and inv_ok

    return ok, messages
