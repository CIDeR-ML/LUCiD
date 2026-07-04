"""Per-job completion truth checks, shared across clusters.

Two flavours:

  - `is_complete_siren(root_path)` — a SIREN-input / smax cell sub-job has
    finished iff its ROOT contains the `OpticalPhotons` TTree key
    (preempted jobs leave basket bytes on disk but never reach
    `DataManager::Finalize()`, so the TTree directory entry is the
    unambiguous truth marker).

  - `is_complete_dataprod(cell_dir, file_index, expected_events)` — a
    dataprod sub-job has finished iff all four files exist, open with
    h5py, and report the expected `config.attrs['n_events']`.

Both used to live inline in `jobs/dataprod/verify_jobs.py` and
`jobs/siren_inputs/resubmit_failed.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def is_complete_siren(root_path: Path) -> bool:
    """True iff the ROOT file exists AND has the OpticalPhotons TTree key."""
    if not root_path.is_file():
        return False
    try:
        import uproot
        with uproot.open(root_path) as f:
            return "OpticalPhotons" in f
    except Exception:
        return False


def is_complete_dataprod(cell_dir: Path, file_index: int,
                          expected_events: Optional[int]) -> Tuple[bool, str]:
    """True iff the four files exist with expected n_events.

    Returns (ok, short_reason). `short_reason` is "" when ok. Imports
    h5py and the layout helpers lazily so this module is importable
    in environments without h5py (e.g. on the lxplus host outside the
    container).
    """
    import h5py
    from lucid.production.verify_output import SUBDIRS, batch_paths

    paths = batch_paths(cell_dir, file_index)
    for sub in SUBDIRS:
        p = paths[sub]
        if not p.is_file():
            return False, f"missing {sub}"
        if p.stat().st_size == 0:
            return False, f"empty {sub}"
        try:
            with h5py.File(p, "r") as h:
                n = int(h["config"].attrs.get("n_events", -1))
        except Exception as e:
            return False, f"unreadable {sub}: {e!r}"
        if n <= 0:
            return False, f"bad n_events={n} in {sub}"
        if expected_events is not None and n != expected_events:
            return False, f"{sub} has n_events={n}, expected {expected_events}"
    return True, ""
