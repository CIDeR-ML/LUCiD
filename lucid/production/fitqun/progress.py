"""One progress line, emitted the same way by every long stage.

``tools/fitqun/eta.py`` turns these into a measured percentage and ETA. The line
carries its own total *and its own clock*, so a rate can be computed from two
lines alone -- no dependence on file mtime, which AFS and re-created logs make
unreliable -- and it is flushed so the number is current rather than whatever a
pipe happens to have released. The reference macros emit the identical line (see
``tools/fitqun/reference/PATCHES.md``), so one watcher covers C++ and Python.
"""
from __future__ import annotations

import sys
import time

_last = {}


def emit(index: int, total: int, *, every: float = 2.0, stream=None) -> None:
    """Print ``PROGRESS index/total t=<unix>``, at most once per ``every`` seconds.

    Throttled because a per-item line from a million-item loop is itself a cost;
    the final index is always emitted so a watcher sees completion.
    """
    stream = stream or sys.stdout
    key = id(stream)
    now = time.monotonic()
    if index >= total or now - _last.get(key, -1e9) >= every:
        _last[key] = now
        print(f"PROGRESS {int(index)}/{int(total)} t={time.time():.3f}",
              file=stream, flush=True)
