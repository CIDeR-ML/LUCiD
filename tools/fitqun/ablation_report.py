"""Compare fiTQun light-model ablations by their closure against the data.

The test is exact and assumption-free: with the track pinned at MC truth, the
mean observed charge over ALL PMTs in a bin must equal the mean predicted mu in
that bin, because E[q] = mu. Unhit PMTs enter with q=0 -- that is what makes the
identity hold, and dropping them is what made earlier ratios meaningless.

Reported per ablation, and differentially in the variables the model depends on,
so a failure localises to a term rather than showing up as one contaminated
number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COLS = "event icab flgHit flgMask R cth mu muscat q tHit tau tres qeeff".split()


def load(path: Path) -> dict:
    # Tolerate a dump still being written: the last line can be truncated, and a
    # job that died mid-event leaves a short row. Drop malformed rows rather than
    # refusing the file, so a partial ablation is still usable.
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) != len(COLS):
                continue
            try:
                rows.append([float(x) for x in f])
            except ValueError:
                continue
    d = np.asarray(rows)
    c = {n: d[:, i] for i, n in enumerate(COLS)}
    keep = c["flgMask"] == 0          # fiTQun's active set; unhit PMTs kept
    return {k: v[keep] for k, v in c.items()}


def closure(c: dict, sel=None) -> tuple[float, float, int]:
    """(<q>/<mu>, its error, N) -- 1.0 if the model is right."""
    m = c["mu"] + c["muscat"]
    q = c["q"]
    if sel is not None:
        m, q = m[sel], q[sel]
    if len(m) == 0 or m.mean() <= 0:
        return float("nan"), float("nan"), len(m)
    return q.mean() / m.mean(), q.std() / np.sqrt(len(q)) / m.mean(), len(m)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", nargs="+", type=Path,
                   help="abl_<name>/pmt.txt files; the first is the baseline")
    p.add_argument("--plot", type=Path, help="write the comparison figure here")
    a = p.parse_args(argv)

    runs = []
    for f in a.dumps:
        name = f.parent.name.replace("abl_", "")
        try:
            runs.append((name, load(f)))
        except Exception as exc:                       # a failed ablation is data too
            print(f"  {name}: unreadable ({exc})")

    # An ablation is only an ablation if every variant is scored on the SAME
    # events. Runs finish at different rates, so intersect on event id before
    # comparing anything -- otherwise a difference in sample masquerades as a
    # difference in model.
    common = set.intersection(*[set(np.unique(c["event"])) for _, c in runs])
    print(f"  common events across all runs: {len(common)}\n")
    keep = np.array(sorted(common))
    runs = [(n, {k: v[np.isin(c["event"], keep)] for k, v in c.items()})
            for n, c in runs]

    print(f"{'ablation':<22}{'<q>/<mu>':>10}{'err':>8}{'pred_pe':>10}{'obs_pe':>10}"
          f"{'%indirect':>11}")
    for name, c in runs:
        r, e, _ = closure(c)
        tot = (c["mu"] + c["muscat"]).sum()
        ind = 100 * c["muscat"].sum() / tot if tot > 0 else 0
        print(f"{name:<22}{r:10.3f}{e:8.3f}{tot:10.0f}{c['q'].sum():10.0f}{ind:11.1f}")

    for var, edges, lab in [
        ("R", np.linspace(200, 4500, 7), "R [cm]"),
        ("cth", np.linspace(-1, 1, 7), "cos(theta0)"),
    ]:
        print(f"\n  <q>/<mu> vs {lab}")
        hdr = "".join(f"{n[:11]:>12}" for n, _ in runs)
        print(f"  {'bin':<18}{hdr}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            row = ""
            for _, c in runs:
                sel = (c[var] >= lo) & (c[var] < hi)
                r, _, n = closure(c, sel)
                row += f"{r:12.3f}" if n > 50 else f"{'-':>12}"
            print(f"  {lo:8.0f}-{hi:<9.0f}{row}")

    if a.plot:
        print(f"\n{plot(runs, a.plot)}")


def plot(runs, out: Path) -> Path:
    """Closure vs each model variable, one line per ablation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("mu", np.geomspace(1e-3, 60, 13), "predicted mu [pe]", True),
              ("R", np.linspace(200, 4500, 11), "R from vertex [cm]", False),
              ("cth", np.linspace(-1, 1, 13), "cos(theta0)", False)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for a, (var, edges, lab, logx) in zip(ax, panels):
        for i, (name, c) in enumerate(runs):
            x, y, e = [], [], []
            v = c["mu"] + c["muscat"] if var == "mu" else c[var]
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (v >= lo) & (v < hi)
                r, er, n = closure(c, sel)
                if n > 50 and np.isfinite(r):
                    x.append(np.sqrt(lo * hi) if logx else 0.5 * (lo + hi))
                    y.append(r); e.append(er)
            a.errorbar(x, y, yerr=e, fmt="o-", ms=4, lw=1.3, capsize=2,
                       color=colors[i], label=name)
        a.axhline(1.0, ls="--", color="k", lw=1.2)
        a.set_xlabel(lab); a.set_ylabel(r"$\langle q\rangle / \langle \mu\rangle$")
        a.set_yscale("log"); a.set_ylim(0.3, 40)
        if logx:
            a.set_xscale("log")
        a.grid(alpha=.3, lw=.5)
    ax[0].legend(fontsize=8, ncol=2)
    fig.suptitle("fiTQun light-model ablations, track pinned at MC truth: "
                 r"$\langle q\rangle/\langle\mu\rangle = 1$ if the model is right",
                 fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145, bbox_inches="tight")
    return out


if __name__ == "__main__":
    main()
