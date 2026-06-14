#!/usr/bin/env python3
"""Rank the seeds of a SIREN seed-scan by loss.

Given a scan directory containing `seed*/` subdirs (as produced by
`generate_jobs.py`), print a table ordered by final validation loss and
name the best seed.

For each seed it reads:
  * `trained_model/<name>_metadata.json` -> training_info.final_{train,val}_loss
  * `training_history.json`              -> min over the val_loss history

Usage:
    python rank_seeds.py <scan_dir> [--sort final_val|best_val|final_train]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _load_seed(seed_dir: Path) -> dict:
    out = {"seed": seed_dir.name, "final_step": None,
           "final_train": None, "final_val": None, "best_val": None}

    metas = sorted(seed_dir.glob("trained_model/*_metadata.json"))
    if metas:
        ti = json.loads(metas[0].read_text()).get("training_info", {})
        out["final_step"] = ti.get("final_step")
        out["final_train"] = ti.get("final_train_loss")
        out["final_val"] = ti.get("final_val_loss")

    hist = seed_dir / "training_history.json"
    if hist.is_file():
        vl = json.loads(hist.read_text()).get("val_loss") or []
        if vl:
            out["best_val"] = min(vl)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan_dir", type=Path,
                   help="Scan directory containing seed*/ subdirs.")
    p.add_argument("--sort", choices=("final_val", "best_val", "final_train"),
                   default="final_val", help="Ranking key (default: final_val).")
    args = p.parse_args(argv)

    seed_dirs = sorted(args.scan_dir.glob("seed*"),
                       key=lambda d: int("".join(c for c in d.name if c.isdigit()) or 0))
    if not seed_dirs:
        print(f"error: no seed*/ subdirs under {args.scan_dir}", file=sys.stderr)
        return 2

    rows = [_load_seed(d) for d in seed_dirs if d.is_dir()]
    key = {"final_val": "final_val", "best_val": "best_val",
           "final_train": "final_train"}[args.sort]
    rows.sort(key=lambda r: r[key] if r[key] is not None else float("inf"))

    def f(x):
        return "   --" if x is None else f"{x:.6f}"

    print(f"{'seed':<8}{'step':>8}{'train_loss':>14}{'val_loss':>14}{'best_val':>14}")
    for r in rows:
        print(f"{r['seed']:<8}{(r['final_step'] or 0):>8}"
              f"{f(r['final_train']):>14}{f(r['final_val']):>14}{f(r['best_val']):>14}")

    ranked = [r for r in rows if r[key] is not None]
    if ranked:
        b = ranked[0]
        print(f"\n>>> best by {args.sort}: {b['seed']}  "
              f"(final_val={f(b['final_val'])}, best_val={f(b['best_val'])}, "
              f"final_train={f(b['final_train'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
