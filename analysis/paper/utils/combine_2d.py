"""Combine the per-shard partial landscapes into the full npz. Each shard filled its disjoint round-robin
points and left zeros elsewhere, so summing across shards reconstructs the full grid exactly."""
import glob, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.utils import paths                            # noqa: E402

FIGURE = "calib_loss_geometry"


def main():
    base = str(paths.data_dir(FIGURE) / "landscape2d_neyman")
    parts = sorted(glob.glob(f"{base}_part*.npz"))
    assert parts, "no shard partials found"
    d0 = np.load(parts[0]); X, Y = d0["X"], d0["Y"]
    keys = ["L", "Gx", "Gy", "Fxx", "Fxy", "Fyy"]
    acc = {k: np.zeros_like(d0[k]) for k in keys}
    for p in parts:
        d = np.load(p)
        for k in keys:
            acc[k] += d[k]                                         # disjoint points -> sum = union
    np.savez(f"{base}.npz", X=X, Y=Y, truth_x=d0["truth_x"], truth_y=d0["truth_y"],
             wl=d0["wl"], **acc)
    print(f"[combine] {len(parts)} shards -> {base}.npz   "
          f"L range [{acc['L'].min():.3g}, {acc['L'].max():.3g}]")


# Guarded so that importing this module (a test collector, a dependency scan) does not run the
# combine as a side effect; `fig_calib_loss_geometry` runs it as a script via subprocess.
if __name__ == "__main__":
    main()
