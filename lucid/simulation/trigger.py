"""Sliding-window readout trigger for the LUCiD data pipeline.

After hit-making (:mod:`lucid.simulation.digitizer`) the detector free-runs with
physics light *and* dark noise. A real DAQ doesn't record everything — it issues
a trigger when the detector-wide hit rate exceeds a threshold and reads out only
the coincident window(s). This module reproduces that: it slides a coincidence
window of width ``window_ns`` over the (detector-wide) digit times, opens a gate
when the multiplicity crosses **up** through ``n_thr``, closes it when the
multiplicity drops back **below**, expands each gate by ``±pad_ns``, and merges
overlaps. Only in-gate digits are kept; the gate list is stored as truth.

This is the SK-like hit-sum / N200 trigger, applied to the digit stream. The
threshold sits well above the dark floor (μ = rate·n_sensors·W): for 4.2 kHz on
~11k PMTs at W=200 ns, μ≈9.2, and because a trigger examines ~1/W ≈ 5·10⁶
windows/s the operating point is ~μ+7σ (n_thr≈30) to keep the accidental rate
below ~0.1 Hz — not the naive 3σ.

Pure NumPy; operates on the per-event digit time list, host-side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriggerConfig:
    """Sliding-window trigger parameters.

    ``window_ns``     — coincidence window W (SK-like ~200 ns; distinct from the
                        per-sensor integration window of the digitizer).
    ``n_thr``         — detector-wide hit multiplicity to fire (in-W hit count).
    ``pad_before_ns`` — gate expansion before the up-crossing (pre-trigger).
    ``pad_after_ns``  — gate expansion after the down-crossing (post-trigger).
    """
    window_ns: float = 200.0
    n_thr: int = 30
    pad_before_ns: float = 30.0
    pad_after_ns: float = 30.0

    @classmethod
    def from_block(cls, block):
        """Build from a physics-config ``"trigger"`` block (dict) or None.

        ``None`` disables the trigger (caller treats it as off). A dict may set
        any of ``window_ns``/``n_thr``/``pad_before_ns``/``pad_after_ns`` (also
        accepts a symmetric ``pad_ns`` shorthand).
        """
        if block is None:
            return None
        b = dict(block)
        if "pad_ns" in b:
            b.setdefault("pad_before_ns", b["pad_ns"])
            b.setdefault("pad_after_ns", b["pad_ns"])
            b.pop("pad_ns")
        allowed = {"window_ns", "n_thr", "pad_before_ns", "pad_after_ns"}
        return cls(**{k: v for k, v in b.items() if k in allowed})


def find_trigger_gates(times: np.ndarray, cfg: TriggerConfig) -> np.ndarray:
    """Return the trigger gates ``[[t_start, t_end], ...]`` for one event.

    The sliding count is the trailing-window multiplicity ``m(t) = #hits in
    (t-W, t]`` — it steps +1 at each hit time and -1 at ``hit+W``. A gate spans
    from the up-crossing of ``n_thr`` (minus pad) to the following down-crossing
    (plus pad); overlapping/adjacent gates are merged. Returns an empty ``(0, 2)``
    array when nothing triggers.

    ``pad_before_ns`` is the pre-trigger window: a burst rising slower than it
    can lose its leading edge (the count only clears ``n_thr`` partway up).
    Physical Cherenkov bursts rise in ~10 ns, well inside the pad, so in practice
    the gate brackets the whole burst (measured physics efficiency ~99.9%).
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    n = t.size
    if n == 0:
        return np.empty((0, 2))
    ev_t = np.concatenate([t, t + cfg.window_ns])                 # +1 at hit, -1 at hit+W
    ev_d = np.concatenate([np.ones(n, np.int64), -np.ones(n, np.int64)])
    order = np.lexsort((-ev_d, ev_t))                             # time asc; +1 before -1 at ties
    ev_t, ev_d = ev_t[order], ev_d[order]
    count = np.cumsum(ev_d)
    above = count >= cfg.n_thr
    prev = np.concatenate([[False], above[:-1]])
    starts = ev_t[above & ~prev] - cfg.pad_before_ns              # up-crossings
    ends = ev_t[~above & prev] + cfg.pad_after_ns                 # down-crossings
    gates = np.column_stack([starts, ends])
    if gates.shape[0] == 0:
        return np.empty((0, 2))
    gates = gates[np.argsort(gates[:, 0])]
    merged: list[list[float]] = []
    for s, e in gates:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([float(s), float(e)])
    return np.asarray(merged)


def assign_windows(times: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Per-hit trigger-window index (``0..n_gates-1``); ``-1`` if out of every gate.

    Gates are disjoint (post-merge), so each hit maps to at most one window.
    """
    times = np.asarray(times, dtype=np.float64)
    win = np.full(times.shape[0], -1, dtype=np.int64)
    if gates.shape[0] == 0 or times.size == 0:
        return win
    idx = np.searchsorted(gates[:, 0], times, side="right") - 1   # candidate gate
    ok = idx >= 0
    inside = np.zeros(times.shape[0], dtype=bool)
    inside[ok] = times[ok] <= gates[idx[ok], 1]
    win[inside] = idx[inside]
    return win


def hits_in_gates(times: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: True for hits inside any gate."""
    return assign_windows(times, gates) >= 0


def apply_trigger(sensor_digits: dict, hits_sparse: dict, seg_hits: dict,
                  cfg: TriggerConfig):
    """Apply the readout trigger to one event's digitizer output.

    Keeps only in-gate digits, **canonically re-sorts** them by
    ``(window, sensor_idx, T)`` (so each window is a contiguous digit slice and
    a PMT's hits within a window are time-ordered), remaps ``digit_idx`` in the
    ``hits_sparse`` / ``seg_hits`` decompositions to the new digit rows, and
    builds the ``per_window`` CSR (``window_start``, ``window_end``,
    ``digit_offsets``).

    Returns ``(sensor_digits, hits_sparse, seg_hits, per_window)`` or ``None``
    when the event produced no trigger (the caller drops the event).
    """
    T = np.asarray(sensor_digits["T"], dtype=np.float64)
    n_dig = T.shape[0]
    gates = find_trigger_gates(T, cfg)
    if gates.shape[0] == 0:
        return None
    win = assign_windows(T, gates)                    # per-digit window, -1 if out
    kept = np.flatnonzero(win >= 0)
    if kept.size == 0:
        return None

    # Canonical order of kept digits: window, then sensor, then time.
    si = np.asarray(sensor_digits["sensor_idx"])[kept]
    order = np.lexsort((T[kept], si, win[kept]))
    new_of_kept = kept[order]                          # old digit index in new order

    new_sd = {k: np.asarray(sensor_digits[k])[new_of_kept] for k in ("sensor_idx", "PE", "T")}

    # old digit_idx -> new row (or -1 if the digit was dropped)
    remap = np.full(n_dig, -1, dtype=np.int64)
    remap[new_of_kept] = np.arange(new_of_kept.size)

    def _remap(d):
        nd = remap[np.asarray(d["digit_idx"])]
        m = nd >= 0
        out = {k: np.asarray(v)[m] for k, v in d.items()}
        out["digit_idx"] = nd[m].astype(np.int32)
        return out

    new_hits = _remap(hits_sparse)
    new_seg = _remap(seg_hits) if seg_hits is not None else None

    # per_window CSR: window per new digit is non-decreasing after the sort.
    win_sorted = win[new_of_kept]
    counts = np.bincount(win_sorted, minlength=gates.shape[0])
    digit_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    per_window = {
        "window_start": gates[:, 0].astype(np.float32),
        "window_end": gates[:, 1].astype(np.float32),
        "digit_offsets": digit_offsets,
    }
    return new_sd, new_hits, new_seg, per_window
