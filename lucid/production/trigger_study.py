"""Trigger study / test: evaluate the sliding-window readout trigger on a dataset.

Loads a produced dataset (``sensor.h5`` digit times + ``hits.h5`` truth), runs
:mod:`lucid.simulation.trigger`, and reports how the trigger performs — physics
efficiency (physics digits kept), dark rejection (dark digits dropped), and the
per-event gate structure — plus a diagnostic plot decomposing the sliding
multiplicity into its physics-only and dark-only contributions.

A digit is classified from the ``hits.h5`` decomposition: it is a *physics*
digit if any of its rows is non-dark, else a pure *dark* digit.

Usage:
    python -m lucid.production.trigger_study DATASET_DIR
    python -m lucid.production.trigger_study DATASET_DIR -W 200 -N 30 --pad 30 \
        --plot out.png --plot-events event_009,event_001
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from lucid.simulation.trigger import TriggerConfig, find_trigger_gates, hits_in_gates
from lucid.simulation.digitizer import EMISSION_PROCESS_DARK as DARK


def _trigger_attrs_from_dataset(dataset_dir):
    """Read the trigger provenance the writer stamped on sensor.h5 config/.

    Lets the study default to the trigger the dataset was actually produced
    with, instead of hardcoded values that may not match it.
    """
    try:
        with h5py.File(_kind_file(dataset_dir, "sensor"), "r") as f:
            a = f["config"].attrs
            return {k[len("trigger_"):]: float(a[k]) if "ns" in k else int(a[k])
                    for k in a if k.startswith("trigger_")}
    except Exception:
        return {}


def _digit_is_physics(hits_grp, n_digits: int) -> np.ndarray:
    """Boolean per digit: True if it has any non-dark contribution in hits.h5."""
    di = hits_grp["digit_idx"][:]
    ep = hits_grp["emission_process"][:]
    phys = np.unique(di[ep != DARK])
    is_phys = np.zeros(n_digits, dtype=bool)
    is_phys[phys[(phys >= 0) & (phys < n_digits)]] = True
    return is_phys


def evaluate(dataset_dir: str, cfg: TriggerConfig):
    """Run the trigger on every event; return per-event rows + summary."""
    sf = h5py.File(_kind_file(dataset_dir, "sensor"), "r")
    hf = h5py.File(_kind_file(dataset_dir, "hits"), "r")
    evs = sorted(k for k in sf if k.startswith("event_"))
    rows = []
    for e in evs:
        T = sf[e]["T"][:].astype(np.float64)
        is_phys = _digit_is_physics(hf[e], len(T))
        is_dark = ~is_phys
        gates = find_trigger_gates(T, cfg)
        keep = hits_in_gates(T, gates)
        rows.append(dict(
            event=e, n=len(T), n_phys=int(is_phys.sum()), n_dark=int(is_dark.sum()),
            n_gates=gates.shape[0],
            gate_us=float((gates[:, 1] - gates[:, 0]).sum() / 1e3) if gates.size else 0.0,
            phys_eff=float(keep[is_phys].mean() * 100) if is_phys.any() else float("nan"),
            dark_kept=float(keep[is_dark].mean() * 100) if is_dark.any() else 0.0,
        ))
    return rows


def print_report(rows, cfg: TriggerConfig):
    print(f"trigger: W={cfg.window_ns:.0f}ns  N_thr={cfg.n_thr}  "
          f"pad={cfg.pad_before_ns:.0f}/{cfg.pad_after_ns:.0f}ns\n")
    hdr = f"{'event':12} {'nDig':>6} {'phys':>6} {'dark':>6} {'gates':>5} {'gate_us':>7} {'physEff%':>8} {'darkKept%':>9}"
    print(hdr)
    for r in rows:
        print(f"{r['event']:12} {r['n']:6d} {r['n_phys']:6d} {r['n_dark']:6d} {r['n_gates']:5d} "
              f"{r['gate_us']:7.2f} {r['phys_eff']:8.1f} {r['dark_kept']:9.2f}")
    eff = np.nanmean([r["phys_eff"] for r in rows])
    dk = np.mean([r["dark_kept"] for r in rows])
    print(f"\nOVERALL: physics efficiency = {eff:.1f}%   mean dark kept (coincident) = {dk:.1f}%")


def make_plot(dataset_dir: str, cfg: TriggerConfig, events, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sf = h5py.File(_kind_file(dataset_dir, "sensor"), "r")
    hf = h5py.File(_kind_file(dataset_dir, "hits"), "r")

    def mult_on_grid(t_sub, grid, W):
        ts = np.sort(np.asarray(t_sub, float))
        return np.searchsorted(ts, grid, "right") - np.searchsorted(ts, grid - W, "right")

    W = cfg.window_ns
    fig, axes = plt.subplots(len(events), 1, figsize=(11, 3.3 * len(events)), squeeze=False)
    for ax, e in zip(axes[:, 0], events):
        T = sf[e]["T"][:].astype(np.float64)
        is_phys = _digit_is_physics(hf[e], len(T))
        gates = find_trigger_gates(T, cfg)
        grid = np.unique(np.concatenate([T, T + W]))
        m_tot = mult_on_grid(T, grid, W)
        m_ph = mult_on_grid(T[is_phys], grid, W)
        m_dk = mult_on_grid(T[~is_phys], grid, W)
        ax.step(grid / 1e3, np.clip(m_dk, 0.1, None), where="post", color="C3", lw=0.9, alpha=0.8, label="dark only")
        ax.step(grid / 1e3, np.clip(m_ph, 0.1, None), where="post", color="C0", lw=0.9, alpha=0.8, label="physics only")
        ax.step(grid / 1e3, m_tot, where="post", color="k", lw=1.2, label="total (=phys+dark)")
        ax.axhline(cfg.n_thr, color="C1", ls="--", lw=1, label=f"N_thr={cfg.n_thr}")
        for s, en in gates:
            ax.axvspan(s / 1e3, en / 1e3, color="C2", alpha=0.15)
        ax.plot(T[is_phys] / 1e3, np.full(int(is_phys.sum()), 0.62), "|", color="C0", ms=5, alpha=0.4)
        ax.plot(T[~is_phys] / 1e3, np.full(int((~is_phys).sum()), 0.45), "|", color="C3", ms=5, alpha=0.4)
        ax.set_title(f"{e}: {len(T)} digits, {gates.shape[0]} gates", fontsize=10)
        ax.set_ylabel("hits in W"); ax.set_xlabel("time [μs]")
        ax.set_yscale("log"); ax.set_ylim(0.38, None)
        # legend "trigger gates" via a proxy patch — never drawn on the axes,
        # so only the real gate spans are shaded.
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="C2", alpha=0.15, label="trigger gates"))
        ax.legend(handles=handles, fontsize=7, ncol=3, frameon=False, loc="upper right")
    fig.suptitle("Sliding-window trigger: gates capture physics bursts, reject dark between", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved plot: {out_path}")


def _kind_file(dataset_dir: str, kind: str) -> str:
    """Locate the wc_<kind>_NNNN.h5 (subdir or flat layout)."""
    import glob
    for pat in (os.path.join(dataset_dir, kind, f"*_{kind}_*.h5"),
                os.path.join(dataset_dir, f"*_{kind}_*.h5")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no *_{kind}_*.h5 under {dataset_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", help="dataset root (with sensor/ hits/ subdirs)")
    # Defaults are taken from the dataset's own trigger provenance (config/
    # attrs trigger*), so the study validates the trigger that was actually
    # applied. Any flag given explicitly overrides it.
    p.add_argument("-W", "--window-ns", type=float, default=None)
    p.add_argument("-N", "--n-thr", type=int, default=None)
    p.add_argument("--pad", "--pad-ns", dest="pad_ns", type=float, default=None,
                   help="set both pads; --pad-before/--pad-after override individually")
    p.add_argument("--pad-before", dest="pad_before_ns", type=float, default=None)
    p.add_argument("--pad-after", dest="pad_after_ns", type=float, default=None)
    p.add_argument("--plot", metavar="OUT.png", default=None, help="write the diagnostic plot")
    p.add_argument("--plot-events", default=None,
                   help="comma list of event_NNN to plot (default: first 2)")
    p.add_argument("--no-report", action="store_true", help="skip the per-event table")
    args = p.parse_args(argv)

    stamped = _trigger_attrs_from_dataset(args.dataset_dir)
    pad = args.pad_ns
    cfg = TriggerConfig(
        window_ns=args.window_ns if args.window_ns is not None
                  else stamped.get("window_ns", 200.0),
        n_thr=args.n_thr if args.n_thr is not None else stamped.get("n_thr", 30),
        pad_before_ns=(args.pad_before_ns if args.pad_before_ns is not None
                       else pad if pad is not None
                       else stamped.get("pad_before_ns", 30.0)),
        pad_after_ns=(args.pad_after_ns if args.pad_after_ns is not None
                      else pad if pad is not None
                      else stamped.get("pad_after_ns", 30.0)),
    )
    if not args.no_report:
        print_report(evaluate(args.dataset_dir, cfg), cfg)
    if args.plot:
        if args.plot_events:
            events = args.plot_events.split(",")
        else:
            sf = h5py.File(_kind_file(args.dataset_dir, "sensor"), "r")
            events = sorted(k for k in sf if k.startswith("event_"))[:2]
        make_plot(args.dataset_dir, cfg, events, args.plot)


if __name__ == "__main__":
    main()
