"""Level 1 Baseline Comparison — Compare two captured baselines.

Usage:
  python baseline_scripts/L1_compare.py baseline_scripts/L1_baseline_tools.json baseline_scripts/L1_baseline_lucid.json
"""
import json
import sys
import numpy as np


def _flatten(x):
    """Recursively flatten nested lists/scalars into a 1D list."""
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [x]


def compare(old_file, new_file, rtol=1e-5, atol=1e-6):
    with open(old_file) as f:
        old = json.load(f)
    with open(new_file) as f:
        new = json.load(f)

    all_keys = sorted(set(old.keys()) | set(new.keys()))
    n_pass = 0
    n_fail = 0
    n_skip = 0

    for key in all_keys:
        if key not in old:
            print(f"  SKIP  {key} (only in new)")
            n_skip += 1
            continue
        if key not in new:
            print(f"  SKIP  {key} (only in old)")
            n_skip += 1
            continue

        # Flatten nested lists for comparison
        old_flat = np.array(_flatten(old[key]), dtype=np.float64)
        new_flat = np.array(_flatten(new[key]), dtype=np.float64)

        if old_flat.shape != new_flat.shape:
            print(f"  FAIL  {key}: shape mismatch {old_flat.shape} vs {new_flat.shape}")
            n_fail += 1
            continue

        if np.allclose(old_flat, new_flat, rtol=rtol, atol=atol):
            n_pass += 1
        else:
            max_abs = float(np.max(np.abs(old_flat - new_flat)))
            max_rel = float(np.max(np.abs(old_flat - new_flat) / (np.abs(old_flat) + 1e-30)))
            print(f"  FAIL  {key}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
            n_fail += 1

    print(f"\n{'='*50}")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    print(f"{'='*50}")
    return n_fail == 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <old_baseline.json> <new_baseline.json>")
        sys.exit(1)

    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
