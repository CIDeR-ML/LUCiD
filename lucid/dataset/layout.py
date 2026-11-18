"""Where a dataset's batches are and whether one is whole.

Batch ``k`` of a dataset rooted at ``root`` is ``<root>/<sub>/wc_<sub>_<k:04d>.h5`` for each
``sub`` in :data:`lucid.dataset.schema.SUBDIRS`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from lucid.dataset.reader import iter_events
from lucid.dataset.schema import SUBDIRS

__all__ = ["batch_paths", "list_batches", "check_batch"]


def batch_paths(root: os.PathLike, file_index: int) -> dict[str, Path]:
    """The four files of batch ``file_index``, keyed by subdirectory."""
    root = Path(root)
    tag = f"{int(file_index):04d}"
    return {sub: root / sub / f"wc_{sub}_{tag}.h5" for sub in SUBDIRS}


def list_batches(root: os.PathLike) -> list[int]:
    """The batch indices with a ``sensor`` file under ``root``, in numeric order."""
    found = (
        re.fullmatch(r"wc_sensor_(\d+)\.h5", p.name) for p in (Path(root) / "sensor").iterdir()
    )
    return sorted(int(m.group(1)) for m in found if m)


def _check_digit_invariants(paths, modalities, n_sample: int = 3) -> tuple[bool, list[str]]:
    """Sample events and check the cross-file digit links among the modalities present.

    - hits.h5 / step.sensor_hits ``digit_idx`` are in range and satisfy
      ``sensor_idx == sensor.h5.sensor_idx[digit_idx]``.
    - ``labl/per_window/digit_offsets`` is a valid CSR over the digit list
      (starts at 0, ends at n_digits, monotonic).
    """
    import h5py
    import numpy as np

    if "sensor" not in modalities:
        return True, []
    msgs: list[str] = []
    ok = True
    try:
        files = {m: h5py.File(paths[m], "r") for m in modalities}
        try:
            sf = files["sensor"]
            evs = [name for _, name in iter_events(paths["sensor"])][:n_sample]
            for e in evs:
                si = sf[e]["sensor_idx"][:]
                nd = si.shape[0]
                if "hits" in files:
                    hd = files["hits"][e]["digit_idx"][:]
                    if hd.size and not (
                        hd.min() >= 0
                        and hd.max() < nd
                        and (files["hits"][e]["sensor_idx"][:] == si[hd]).all()
                    ):
                        msgs.append(f"BAD_DIGIT_FK(hits): {e}")
                        ok = False
                sh = files["step"][e].get("sensor_hits") if "step" in files else None
                if sh is not None and "digit_idx" in sh:
                    sd = sh["digit_idx"][:]
                    if sd.size and not (
                        sd.min() >= 0 and sd.max() < nd and (sh["sensor_idx"][:] == si[sd]).all()
                    ):
                        msgs.append(f"BAD_DIGIT_FK(step): {e}")
                        ok = False
                pw = files["labl"][e].get("per_window") if "labl" in files else None
                if pw is not None:
                    off = pw["digit_offsets"][:]
                    if not (off[0] == 0 and off[-1] == nd and (np.diff(off) >= 0).all()):
                        msgs.append(
                            f"BAD_PER_WINDOW_CSR: {e} (offsets end {int(off[-1])} != n_digits {nd})"
                        )
                        ok = False
        finally:
            for f in files.values():
                f.close()
        if ok:
            msgs.append(
                f"OK invariants (digit_idx FK, per_window CSR) on {len(evs)} sampled events"
            )
    except Exception as exc:
        msgs.append(f"INVARIANT_CHECK_ERROR: {exc!r}")
        ok = False
    return ok, msgs


def check_batch(
    output_dir: os.PathLike,
    file_index: int,
    expected_dataset_name: Optional[str] = None,
    *,
    max_events: Optional[int] = None,
    modalities=SUBDIRS,
) -> tuple[bool, list[str]]:
    """Check that batch ``file_index`` was written whole, by one job. Returns ``(ok, messages)``.

    Each file of ``modalities`` (default all four) exists, is non-empty and opens; its ``config``
    carries ``complete = True`` (the writer sets it last, so a file without it is incomplete), the
    batch's ``file_index`` and, when given, ``expected_dataset_name``; and its event groups are
    exactly ``event_000`` .. ``event_{n-1}`` for its ``n_events`` (a gap or a lost last event fails).
    Across the files, ``n_events`` and ``run_id`` agree (a file left by another attempt of the job
    has another ``run_id``), and ``n_events <= max_events`` when given. On a few sampled events the
    cross-file digit links are checked. ``messages`` is one status line per file, for a log; a
    failure line never starts with ``OK`` or ``EMPTY_KEPT``.
    """
    import h5py

    paths = {m: p for m, p in batch_paths(output_dir, file_index).items() if m in modalities}
    messages: list[str] = []
    ok = True
    n_by_sub, run_by_sub = {}, {}

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
                groups = {k for k in h.keys() if k.startswith("event_")}
        except Exception as e:
            messages.append(f"UNREADABLE: {path}: {e!r}")
            ok = False
            continue
        name = cfg_attrs.get("dataset_name", "<missing>")
        fi = int(cfg_attrs.get("file_index", -1))
        n_events = int(cfg_attrs.get("n_events", -1))
        complete = bool(cfg_attrs.get("complete", False))
        n_req = int(cfg_attrs.get("n_events_requested", n_events))
        n_by_sub[sub], run_by_sub[sub] = n_events, str(cfg_attrs.get("run_id", "<missing>"))

        if fi != file_index:
            messages.append(f"BAD_FILE_INDEX: {path} has file_index={fi}, expected {file_index}")
            ok = False
        if not complete:
            messages.append(f"INCOMPLETE: {path} config/complete != True (job killed mid-write?)")
            ok = False
        if n_events < 0:
            messages.append(f"BAD_N_EVENTS: {path} has n_events={n_events}")
            ok = False
        else:
            expected = {f"event_{i:03d}" for i in range(n_events)}
            if groups != expected:
                missing, extra = sorted(expected - groups), sorted(groups - expected)
                messages.append(
                    f"BAD_EVENTS: {path} has n_events={n_events} but event groups "
                    f"missing {missing[:5]} unexpected {extra[:5]}"
                )
                ok = False
            elif n_events == 0:
                # Not fatal: a selection may legitimately drop every event. Complete + empty.
                messages.append(f"EMPTY_KEPT: {path} kept 0/{n_req} events (selection dropped all)")
        if expected_dataset_name is not None and name != expected_dataset_name:
            messages.append(
                f"BAD_DATASET_NAME: {path} has {name!r}, expected {expected_dataset_name!r}"
            )
            ok = False
        drop = f" (kept {n_events}/{n_req})" if n_req != n_events else ""
        messages.append(
            f"OK {sub:<6} size={size:>10} dataset_name={name!r} file_index={fi} "
            f"n_events={n_events}{drop} complete={complete}"
        )

    if len(set(n_by_sub.values())) > 1:
        messages.append(f"N_EVENTS_MISMATCH: {n_by_sub}")
        ok = False
    if len(set(run_by_sub.values())) > 1:
        messages.append(f"RUN_ID_MISMATCH (files from different attempts): {run_by_sub}")
        ok = False
    if max_events is not None and any(n > max_events for n in n_by_sub.values()):
        messages.append(f"TOO_MANY_EVENTS: {n_by_sub} > expected {max_events}")
        ok = False

    # Cross-file schema invariants (only when every file opened and agreed).
    if ok:
        inv_ok, inv_msgs = _check_digit_invariants(paths, tuple(paths))
        messages.extend(inv_msgs)
        ok = ok and inv_ok
    return ok, messages
