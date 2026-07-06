"""Per-job completion truth checks, shared across clusters.

Two flavours:

  - `is_complete_siren(root_path)` — a SIREN-input / smax cell sub-job has
    finished iff its ROOT contains the `OpticalPhotons` TTree key
    (preempted jobs leave basket bytes on disk but never reach
    `DataManager::Finalize()`, so the TTree directory entry is the
    unambiguous truth marker).

  - `is_complete_dataprod(cell_dir, file_index, expected_events)` — a
    dataprod sub-job has finished iff all four files exist, open with
    h5py, agree on `config.attrs['n_events']`, and match the expected
    count: exactly for untriggered datasets, or `<= expected` for
    triggered ones (which drop non-triggering events).

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
    n_by_sub = {}
    triggered = False
    for sub in SUBDIRS:
        p = paths[sub]
        if not p.is_file():
            return False, f"missing {sub}"
        if p.stat().st_size == 0:
            return False, f"empty {sub}"
        try:
            with h5py.File(p, "r") as h:
                n_by_sub[sub] = int(h["config"].attrs.get("n_events", -1))
                triggered = triggered or str(h["config"].attrs.get("trigger", "none")) != "none"
        except Exception as e:
            return False, f"unreadable {sub}: {e!r}"
        if n_by_sub[sub] <= 0:
            return False, f"bad n_events={n_by_sub[sub]} in {sub}"

    # The four files are written together, so they must agree on n_events.
    if len(set(n_by_sub.values())) != 1:
        return False, f"n_events mismatch across files: {n_by_sub}"
    n = next(iter(n_by_sub.values()))

    if expected_events is not None:
        # A triggered dataset drops non-triggering events, so it writes n <=
        # expected; an untriggered job writes exactly the requested count. A
        # crashed/preempted job leaves *no* file (the batch is written
        # all-or-nothing after the event loop, caught above as "missing"), so a
        # written-but-short n can only be trigger drops -> accept it.
        if triggered:
            if n > expected_events:
                return False, f"n_events={n} > expected {expected_events}"
        elif n != expected_events:
            return False, f"n_events={n}, expected {expected_events}"
    return True, ""
