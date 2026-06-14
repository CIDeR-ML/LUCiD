"""Cluster-agnostic planning for SIREN training hyperparameter scans (Stage 3).

A scan config defines a `baseline` dict + a `runs` list of explicit
per-run override dicts. For each run we:

  1. Validate override keys against `FLAG_MAP` (typos raise at config
     load — they don't silently drop hyperparameters).
  2. Resolve the run by merging baseline + overrides.
  3. Derive a compact folder name from the diff against baseline so the
     output tree is self-documenting (e.g. `patience=40` → `p40`).
  4. Build the `lucid-train-siren` CLI argument list.

The cluster-specific part (which submit-description to write per run,
which command to use to submit it) lives in `cluster.py`.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List


# Maps a baseline / run-override key to the matching `lucid-train-siren` flag.
# Unknown keys raise in resolve_run() so typos don't silently lose params.
FLAG_MAP: Dict[str, str] = {
    "material":            "--material",
    "particle":            "--particle",
    "data_type":           "--data-type",
    "h5_path":             "--h5-path",
    "num_steps":           "--num-steps",
    "batch_size":          "--batch-size",
    "learning_rate":       "--learning-rate",
    "min_lr":              "--min-lr",
    "patience":            "--patience",
    "lr_reduction_factor": "--lr-reduction-factor",
    "zero_threshold":      "--zero-threshold",
    "zero_keep_frac":      "--zero-keep-frac",
    "energy_balance":      "--energy-balance",
    "target_importance":   "--target-importance",
    "val_split":           "--val-split",
    "hidden_features":     "--hidden-features",
    "hidden_layers":       "--hidden-layers",
    "w0":                  "--w0",
    "weight_decay":        "--weight-decay",
    "grad_clip_norm":      "--grad-clip-norm",
    "seed":                "--seed",
    "log_every":           "--log-every",
    "val_every":           "--val-every",
    "checkpoint_every":    "--checkpoint-every",
    "prediction_plot_every": "--prediction-plot-every",
}


# Short labels for folder names. Unmapped keys fall back to the raw key.
NAME_MAP: Dict[str, str] = {
    "patience":       "p",
    "zero_threshold": "z",
    "zero_keep_frac": "zf",
    "energy_balance": "ebal",
    "target_importance": "ti",
    "batch_size":     "b",
    "learning_rate":  "lr",
    "min_lr":         "minlr",
    "num_steps":      "steps",
    "hidden_features": "hf",
    "hidden_layers":   "hl",
    "w0":              "w0",
    "val_split":       "vs",
    "weight_decay":    "wd",
    "grad_clip_norm":  "gc",
}


def _fmt_value(v: Any) -> str:
    """Compact formatter for folder names.

    Floats < 1 → scientific (``1e-02``); other floats → ``%g``; ints stay
    as-is; strings stay verbatim.
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1:
            return f"{v:.0e}".replace("e+", "e").replace("e-0", "e-")
        return f"{v:g}"
    return str(v)


def derive_run_name(diff: Dict[str, Any]) -> str:
    """Compact folder name from a run's diff against baseline."""
    if not diff:
        return "baseline"
    parts = [f"{NAME_MAP.get(k, k)}{_fmt_value(v)}"
             for k, v in sorted(diff.items())]
    return "_".join(parts)


def resolve_run(baseline: Dict[str, Any],
                overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge baseline + overrides, validate keys, return the full config."""
    unknown = [k for k in overrides if k not in FLAG_MAP]
    if unknown:
        raise ValueError(
            f"Unknown override key(s): {unknown}. Add them to FLAG_MAP in "
            f"cluster_common/train_planning.py if they're real new CLI flags."
        )
    merged = dict(baseline)
    merged.update(overrides)
    missing = [k for k in ("material", "particle", "data_type", "h5_path")
               if k not in merged]
    if missing:
        raise ValueError(f"baseline missing required keys: {missing}")
    return merged


def _fmt_cli(v: Any) -> str:
    """Format a Python value the way the CLI accepts it."""
    if isinstance(v, bool):
        # Booleans are not currently represented as values in lucid-train-siren;
        # all on/off flags are dest pairs (e.g. --prediction-plots /
        # --no-prediction-plots). If we ever need them here, add the special
        # case at that point.
        raise ValueError("boolean baseline values aren't supported")
    if isinstance(v, float):
        # Avoid scientific-notation surprises for the CLI parser: %.10g is
        # round-trip-safe for the floats we use here.
        return f"{v:.10g}"
    return str(v)


def build_cli_args(resolved: Dict[str, Any], run_dir) -> str:
    """Build the `lucid-train-siren` argument string from the resolved config."""
    pieces: List[str] = ["--output-dir", shlex.quote(str(run_dir)),
                         "--no-monitoring"]
    for key, value in resolved.items():
        if key not in FLAG_MAP:
            continue  # already validated; just defensive
        pieces.extend([FLAG_MAP[key], shlex.quote(_fmt_cli(value))])
    return " ".join(pieces)
