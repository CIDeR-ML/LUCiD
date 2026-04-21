"""Verify that a v3 four-file batch was written correctly.

A successful `lucid-run-job` leaves, for one batch `file_index = F`,
four files under `<output_dir>/{sensor,inst,seg,labl}/wc_*_<F:04d>.h5`.
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


V3_SUBDIRS = ("sensor", "inst", "seg", "labl")


def v3_batch_paths(output_dir: os.PathLike, file_index: int) -> dict[str, Path]:
    """Return the expected four-file paths for a given batch."""
    root = Path(output_dir)
    tag = f"{file_index:04d}"
    return {sub: root / sub / f"wc_{sub}_{tag}.h5" for sub in V3_SUBDIRS}


def verify_batch(
    output_dir: os.PathLike,
    file_index: int,
    expected_dataset_name: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Check the four v3 files for one batch. Return (ok, messages)."""
    import h5py

    paths = v3_batch_paths(output_dir, file_index)
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

    return ok, messages
