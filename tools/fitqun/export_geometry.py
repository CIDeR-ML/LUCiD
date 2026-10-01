#!/usr/bin/env python3
"""Write LUCiD's detector geometry in the plain form SadQun and the WCSim
converter read.

SadQun deliberately does not link numpy or HDF5: the only thing WCSim ever did
for fiTQun was fill PMTpos/PMTdir/PMTQE, so a text file is enough and keeps the
container to ROOT alone.

**Pass ``--sensor`` whenever the geometry will be used to interpret hits.**
A LUCiD sensor file's ``sensor_idx`` indexes that file's own
``config/sensor_positions``, which is a *different ordering* from
``sk_geometry.npz``'s ``positions_mm`` -- for SK_WAND not one of the 11096 rows
agrees, and rows differ by up to 48 m. Writing the npz order while hits carry
sensor-file indices attributes every hit to an unrelated PMT: the total charge
stays right, the spatial pattern is destroyed, and a reconstruction returns a
random vertex and no direction at all. With ``--sensor`` the sensor file's order
is authoritative and orientations are matched to it by position.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="LUCiD sk_geometry.npz")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--sensor", type=Path, default=None,
                   help="a LUCiD wc_sensor_*.h5 whose sensor_idx ordering the "
                        "output must follow (required if hits will be mapped)")
    p.add_argument("--match-tol-cm", type=float, default=5.0)
    a = p.parse_args()

    g = np.load(a.npz)
    pos_cm = g["positions_mm"] / 10.0          # mm -> cm, fiTQun works in cm
    dirs = np.asarray(g["directions"], dtype=np.float64)

    if a.sensor is not None:
        import h5py
        with h5py.File(a.sensor, "r") as f:
            sensor_cm = np.asarray(f["config/sensor_positions"], dtype=np.float64) * 100.0
        if len(sensor_cm) != len(pos_cm):
            raise SystemExit(f"sensor file has {len(sensor_cm)} sensors, npz has {len(pos_cm)}")
        # Reorder the npz to the sensor file's order, matching by position. The two
        # differ by ~3 cm (float32 storage), so match nearest rather than exactly.
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(pos_cm).query(sensor_cm)
        if dist.max() > a.match_tol_cm:
            raise SystemExit(f"cannot match sensor to npz geometry: max {dist.max():.1f} cm")
        if len(set(idx.tolist())) != len(idx):
            raise SystemExit("sensor-to-npz match is not one-to-one")
        print(f"{a.sensor.name}: reordered to the sensor file's sensor_idx "
              f"(max position difference {dist.max():.2f} cm)")
        pos_cm, dirs = sensor_cm, dirs[idx]
    qe = g["PMTQE"] if "PMTQE" in g.files else np.ones(len(pos_cm))

    lines = [
        f"n_pmt {len(pos_cm)}",
        f"pmt_radius_cm {float(g['sensor_radius']) * 100.0:.6f}",
        f"det_radius_cm {float(g['radius']) * 100.0:.6f}",
        f"det_halfz_cm {float(g['height']) * 100.0 / 2.0:.6f}",
    ]
    for i, (p_, d_, q_) in enumerate(zip(pos_cm, dirs, qe)):
        lines.append(f"pmt {i} {p_[0]:.6f} {p_[1]:.6f} {p_[2]:.6f} "
                     f"{d_[0]:.6f} {d_[1]:.6f} {d_[2]:.6f} {float(q_):.6f}")

    a.output.write_text("\n".join(lines) + "\n")
    print(f"{a.output}: {len(pos_cm)} PMTs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
