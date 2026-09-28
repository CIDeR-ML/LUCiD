"""Summarise a TuningMode=4 run into one QEEff value, and record it.

In TuningMode 4 ``fq1rnll`` carries the fitted QEEff rather than a likelihood,
because ``FitQEEff`` saves through
``SaveDefaultSnglTrkFit(X,0,0,iPID,totmu,QEEff_best,PCflg)`` (fiTQun.cc:4841).

The per-event distribution is skewed and has a hard ceiling at the fit bound
(0.2, fiTQun.cc:4815), so the median is used rather than the mean, and events
sitting on either bound are reported separately -- they mean the scale cannot be
reached with the tables in use, which is information, not an outlier to trim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: fiTQun.cc:4815 -- minimiser bounds on the QEEff parameter.
FIT_LO, FIT_HI = 0.0, 0.2


def summarise(fq_path: Path, pdg: int = 13) -> dict:
    import uproot

    t = uproot.open(fq_path)["fiTQun"]

    # These branches come back as an OBJECT array: one (npeak, nPID) block per
    # event, and npeak varies between events, so they cannot be cast to a plain
    # float array. Index the PID column explicitly. PIDarr is
    # {22,11,13,211,321,2212,48} (fiTQun_shared.cc:59), and TuningMode 4 fits
    # peak 0 only.
    ipid = {22: 0, 11: 1, 13: 2, 211: 3, 321: 4, 2212: 5, 48: 6}[abs(pdg)]

    def column(branch):
        arr = t[branch].array(library="np")
        out = np.full(len(arr), np.nan)
        for i, e in enumerate(arr):
            a = np.asarray(e)
            if a.size and a.ndim == 2 and a.shape[1] > ipid:
                out[i] = a[0][ipid]
        return out

    q = column("fq1rnll")          # TuningMode 4 stores QEEff here, not an NLL
    flg = column("fq1rpcflg")

    good = np.isfinite(q) & (flg >= 0)          # PCflg<0 marks non-convergence
    at_hi = good & (q >= FIT_HI * 0.999)
    at_lo = good & (q <= FIT_LO + 1e-12)
    use = good & ~at_hi & ~at_lo

    out = {
        "n_events": int(len(q)),
        "n_converged": int(good.sum()),
        "n_at_upper_bound": int(at_hi.sum()),
        "n_at_lower_bound": int(at_lo.sum()),
        "n_used": int(use.sum()),
    }
    if use.sum():
        v = q[use]
        lo, med, hi = np.percentile(v, [16, 50, 84])
        out.update(qeeff=float(med), q16=float(lo), q84=float(hi),
                   spread_pct=float(100 * 0.5 * (hi - lo) / med),
                   err=float(0.5 * (hi - lo) / np.sqrt(use.sum())))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fq", type=Path, required=True, help="TuningMode=4 output")
    p.add_argument("--pdg", type=int, default=13)
    p.add_argument("--manifest", type=Path, help="record the result here")
    p.add_argument("--held-out", default="", help="what sample this was fitted on")
    a = p.parse_args(argv)

    r = summarise(a.fq, a.pdg)
    print(f"QEEff from fiTQun TuningMode=4 (track fixed at MC truth), pdg={a.pdg}")
    print(f"  {r['n_converged']}/{r['n_events']} converged; "
          f"{r['n_at_upper_bound']} at the 0.2 bound, "
          f"{r['n_at_lower_bound']} at 0; {r['n_used']} used")
    if "qeeff" not in r:
        print("  no usable events -- nothing to record")
        return
    print(f"  QEEff = {r['qeeff']:.5f} +- {r['err']:.5f} "
          f"(68% band {r['q16']:.5f}-{r['q84']:.5f}, spread {r['spread_pct']:.1f}%)")
    if r["n_at_upper_bound"]:
        print(f"  WARNING: {r['n_at_upper_bound']} events pinned at the fit's upper "
              "bound -- the tables cannot reach the observed charge for those.")
    if a.held_out:
        print(f"  held-out sample: {a.held_out}")

    if a.manifest:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from lucid.production.fitqun import manifest as M

        m = M.load(a.manifest)
        M.record(a.manifest, detector=m.detector, config=m.config, pmt_type=m.pmt_type,
                 kind="scalar", install_as="fiTQun.QEEffWCSim", status="adopted",
                 value=round(r["qeeff"], 5),
                 note=(f"Measured with fiTQun's own procedure: TuningMode=4 calls "
                       f"FitQEEff, which pins all eight track parameters at MC truth and "
                       f"floats QEEff alone, so the fit cannot absorb a mismatch into "
                       f"momentum. Median over {r['n_used']} held-out events "
                       f"({a.held_out}); 68% band {r['q16']:.5f}-{r['q84']:.5f}. "
                       f"Re-measure whenever photon yield, QE curve, digitizer or "
                       f"geometry change, and always AFTER the time PDF is installed -- "
                       f"FitQEEffWrapper minimises the full likelihood including time."),
                 provenance=dict(method="fiTQun.TuningMode=4 / FitQEEff", **r))
        print(f"  recorded in {a.manifest}")


if __name__ == "__main__":
    main()
