"""Render a tune manifest as a table, and as the data a plot page consumes.

Deliberately dumb: it knows nothing about any particular input, it renders what
the manifest lists. A new input kind becomes visible here by being recorded, not
by editing this file -- except to add a plotter for it, which is the one thing
that has to know the file's internals.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import manifest as M

#: Kind -> the plot a reader needs to judge that input. A kind with no entry is
#: still reported in the table; it just has no figure yet.
PLOTS = {
    "cprofile": "gsthr and Nphot against momentum, against the reference tune",
    "cprofile_fit": "fitted vs raw profile at a few momenta, and the fit residual",
    "charge_pdf": "P(q|mu) slices at low/mid/high mu, and P_unhit(mu)",
    "angular_response": "response against cos(eta) with the two-piece fit overlaid",
    "scattable_6d": "indirect/direct ratio against source-PMT geometry, occupancy map",
    "scattable_3d": "same projections as the 6D table, for comparison",
    "time_pdf": "t_c distributions at low/mid/high predicted charge, per momentum",
    "scalar": "value against whatever it was measured from, when it was measured",
}

_ORDER = {s: i for i, s in enumerate(
    ("missing", "borrowed", "reference_default", "measured_not_adopted", "adopted"))}


def rows(m: M.Manifest) -> list[dict]:
    """Manifest entries worst-first, so what needs attention reads at the top."""
    out = []
    for a in sorted(m.artifacts, key=lambda a: (_ORDER.get(a.status, 9), a.kind,
                                                a.install_as)):
        exists = bool(a.path) and Path(a.path).exists()
        out.append({
            "kind": a.kind, "install_as": a.install_as, "status": a.status,
            "value": a.value, "path": a.path,
            "on_disk": exists,
            "size_mb": round(Path(a.path).stat().st_size / 1e6, 2) if exists else None,
            "note": a.note, "provenance": a.provenance,
            "plot": PLOTS.get(a.kind, ""),
        })
    return out


def text(m: M.Manifest) -> str:
    r = rows(m)
    counts: dict[str, int] = {}
    for x in r:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    w = max(len(x["install_as"]) for x in r)
    lines = [f"fiTQun tune inputs for {m.detector} "
             f"(config={m.config}, pmt={m.pmt_type})",
             "  " + "  ".join(f"{k}={v}" for k, v in
                              sorted(counts.items(), key=lambda kv: _ORDER.get(kv[0], 9))),
             ""]
    for x in r:
        val = f"= {x['value']:g}" if x["value"] is not None else (
            f"{x['size_mb']} MB" if x["size_mb"] else
            ("on disk" if x["on_disk"] else "-"))
        lines.append(f"  {x['status']:<18} {x['install_as']:<{w}}  {val}")
        if x["status"] != "adopted" and x["note"]:
            lines.append(f"  {'':<18} {'':<{w}}  -> {x['note'][:150]}")
    return "\n".join(lines)


def data(m: M.Manifest) -> dict:
    """What a plot page reads: the rows plus the verification summary."""
    return {"detector": m.detector, "config": m.config, "pmt_type": m.pmt_type,
            "rows": rows(m), "verify": M.verify(m)}


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--json", type=Path, help="also write the plot-page data here")
    p.add_argument("--plots", type=Path, help="render each input's figure here")
    a = p.parse_args(argv)
    m = M.load(a.manifest)
    print(text(m))
    if a.plots:
        from . import plots as P
        made = P.render(m, a.plots)
        n = sum(len(v) for v in made.values())
        print(f"\n{n} figures -> {a.plots}")
    if a.json:
        a.json.write_text(json.dumps(data(m), indent=2) + "\n")
        print(f"\nplot data -> {a.json}")


if __name__ == "__main__":
    main()
