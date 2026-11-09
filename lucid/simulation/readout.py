"""The readout of a data-mode event: digitizer model, trigger and selection, as one record.

A dataset's readout is three choices made together: the digitizer that turns detected
photo-electrons into digits, the sliding-window trigger that decides which digits are read out,
and, for low-energy datasets, the trigger-free truth selection (``min_physics_hits``) that
replaces it. :class:`Readout` holds the three, validates the config blocks they come from, and
answers the questions every event generator asks of them: how wide the dark-noise span must be,
which digits an event keeps, and what the dataset records about it.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np

from lucid.simulation.digitizer import resolve_model_config
from lucid.simulation.trigger import TriggerConfig, apply_trigger
from lucid.dataset.schema import EMISSION_PROCESS_DARK

__all__ = ["Readout"]

# Keys `TriggerConfig.from_block` understands. It silently drops anything else, so
# `"n_threshold": 45` instead of `"n_thr": 45` ran the default n_thr=30 and the only trace was
# the gate count; `Readout.from_blocks` refuses such a block instead.
_TRIGGER_BLOCK_KEYS = frozenset({
    "window_ns", "n_thr", "pad_before_ns", "pad_after_ns", "pad_ns",
})

# Dark-noise readout pad (ns) when no trigger is configured: brackets the physics envelope so
# single-window (min_physics_hits) events carry dark.
_DARK_PAD_NO_TRIGGER_NS = 100.0


def _strict_trigger_block(block):
    """Return ``block`` unchanged, raising on a key ``TriggerConfig`` ignores.

    Parameters
    ----------
    block : dict or None
        A physics- or dataset-config ``"trigger"`` block. ``None`` disables the trigger. Keys
        starting with ``_`` are comments and are allowed.

    Raises
    ------
    ValueError
        If ``block`` carries a key that is neither a trigger parameter nor a ``_``-prefixed
        comment, naming the key and the nearest valid one.
    """
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(
            f"trigger block must be a dict or None, got {type(block).__name__}")
    unknown = [k for k in block
               if k not in _TRIGGER_BLOCK_KEYS and not str(k).startswith("_")]
    if unknown:
        parts = []
        for k in sorted(unknown):
            near = difflib.get_close_matches(str(k), sorted(_TRIGGER_BLOCK_KEYS), n=1)
            parts.append(f"{k!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
        raise ValueError(
            "unknown key(s) in the trigger block: " + ", ".join(parts)
            + f"; valid keys are {sorted(_TRIGGER_BLOCK_KEYS)}")
    return block


def _keep_min_physics_hits(sensor_digits, hits_sparse, seg_hits, min_hits):
    """Truth-level selection: keep the event as ONE readout window iff it has >= ``min_hits``
    *physics* digits (a digit with any non-dark contribution).

    Coincident dark digits are kept and stay tagged ``emission_process = dark``. Digits are
    sorted into canonical (sensor, T) order with ``digit_idx`` remapped. Returns
    ``(sensor_digits, hits_sparse, seg_hits, per_window)`` or ``None`` to drop the event.
    """
    sd = sensor_digits
    nd = int(np.asarray(sd['sensor_idx']).shape[0])
    if nd == 0:
        return None
    hd = np.asarray(hits_sparse['digit_idx'])
    ep = np.asarray(hits_sparse['emission_process'])
    phys = np.zeros(nd, dtype=bool)
    np.logical_or.at(phys, hd[ep != EMISSION_PROCESS_DARK], True)
    if int(phys.sum()) < min_hits:
        return None

    T = np.asarray(sd['T'])
    order = np.lexsort((T, np.asarray(sd['sensor_idx'])))   # canonical (sensor, T)
    remap = np.empty(nd, dtype=np.int64)
    remap[order] = np.arange(nd)
    sd_out = {k: np.asarray(v)[order] for k, v in sd.items()}
    hits_out = dict(hits_sparse)
    hits_out['digit_idx'] = remap[hd].astype(hd.dtype)
    seg_out = None
    if seg_hits is not None:
        sdi = np.asarray(seg_hits['digit_idx'])
        seg_out = dict(seg_hits)
        seg_out['digit_idx'] = remap[sdi].astype(sdi.dtype)
    per_window = {
        'window_start':  np.array([float(T.min())], dtype=np.float64),
        'window_end':    np.array([float(T.max())], dtype=np.float64),
        'digit_offsets': np.array([0, nd], dtype=np.int32),
    }
    return sd_out, hits_out, seg_out, per_window


@dataclass(frozen=True)
class Readout:
    """Digitizer, trigger and selection of a dataset.

    Attributes
    ----------
    digitizer : dict
        The resolved digitizer model (:func:`lucid.simulation.digitizer.resolve_model_config`).
    trigger : TriggerConfig or None
        The sliding-window trigger; ``None`` reads out everything.
    min_physics_hits : int or None
        When set, the trigger-free truth selection replaces the trigger: an event is kept as one
        window iff it has at least this many physics digits.
    """

    digitizer: dict
    trigger: TriggerConfig | None = None
    min_physics_hits: int | None = None

    def __post_init__(self):
        # select() never runs the trigger under min_physics_hits, yet meta() would record it
        # and dark_pad_ns() would size the dark span from it.
        if self.trigger is not None and self.min_physics_hits is not None:
            raise ValueError(
                "the min_physics_hits selection replaces the trigger, so a trigger block with "
                "it is never applied; set the trigger to null")

    @classmethod
    def from_blocks(cls, digitizer=None, trigger=None, min_physics_hits=None) -> "Readout":
        """Build from config blocks: a ``digitizer`` block (``None`` = ``basic``), a ``trigger``
        block (``None`` = off; unknown keys raise) and the selection threshold."""
        return cls(resolve_model_config(digitizer),
                   TriggerConfig.from_block(_strict_trigger_block(trigger)),
                   min_physics_hits)

    def dark_pad_ns(self) -> float:
        """Dark-noise pad (ns) so the *whole* trigger gate is populated with dark.

        Dark is generated over ``[min(t_reco) - pad, max(t_reco) + pad]`` before the trigger
        runs. A gate can open ``pad_before`` ns before the first physics hit and close up to
        ``pad_after + window`` ns after the last, so the pad must be at least
        ``max(pad_before, pad_after) + window`` for every gate edge to sit inside the dark span;
        excess dark outside the gates is trigger-dropped.
        """
        if self.trigger is None:
            return _DARK_PAD_NO_TRIGGER_NS
        return max(self.trigger.pad_before_ns, self.trigger.pad_after_ns) + self.trigger.window_ns

    def select(self, sensor_digits, hits_sparse, seg_hits):
        """Apply the selection to one event's digits.

        Returns ``(sensor_digits, hits_sparse, seg_hits, per_window)``, or ``None`` when the event
        is dropped. Without a trigger or a threshold everything is kept and ``per_window`` is
        ``None``.
        """
        if self.min_physics_hits is not None:
            return _keep_min_physics_hits(sensor_digits, hits_sparse, seg_hits,
                                          self.min_physics_hits)
        if self.trigger is not None:
            return apply_trigger(sensor_digits, hits_sparse, seg_hits, self.trigger)
        return sensor_digits, hits_sparse, seg_hits, None

    def meta(self) -> dict:
        """The readout attributes a dataset records in its ``config/`` group."""
        out = {'digitizer_model': str(self.digitizer['model'])}
        if self.trigger is None:
            out['trigger'] = 'none'
        else:
            out.update(trigger='sliding_window',
                       trigger_window_ns=float(self.trigger.window_ns),
                       trigger_n_thr=int(self.trigger.n_thr),
                       trigger_pad_before_ns=float(self.trigger.pad_before_ns),
                       trigger_pad_after_ns=float(self.trigger.pad_after_ns))
        mph = self.min_physics_hits
        out['selection'] = 'min_physics_hits' if mph is not None else 'trigger'
        out['selection_min_physics_hits'] = int(mph) if mph is not None else 0
        return out
