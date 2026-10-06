#!/usr/bin/env python
"""Compare paired same-key and independent-key synthetic closures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lucid.juno_laser.diagnostics import summarize_key_closure_pair


def main():
    """Load two closure JSON files and write their paired comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    same = json.loads(args.same.read_text())
    independent = json.loads(args.independent.read_text())
    result = summarize_key_closure_pair(same, independent)
    result["same_result"] = str(args.same.resolve())
    result["independent_result"] = str(args.independent.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
