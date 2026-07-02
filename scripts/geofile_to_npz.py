"""geofile_to_npz — convert a WCSim-style detector geofile (.txt) to a PMT-array .npz.

`lucid.geometry.Cylinder.from_pmt_file()` reads a `.npz` conforming to
`lucid/geometry/PMT_NPZ_SCHEMA.md`. The example detectors (SK/HK/WCTE) ship those `.npz` files,
but the *converter* that produces them from the source `config/geofile_*.txt` was previously kept
only locally — so users could not reproduce or add a measured detector. This is that converter.

Geofile format (cm): a header block ("Detector radius & height ...", "Type 1/2 ...", "OD ...",
"Centre offset ...") followed by one row per PMT:
    pmt_id  pmt_id2  pmt_type  x  y  z  dx  dy  dz  [extra]

Run:  python scripts/geofile_to_npz.py config/geofile_SuperK.txt config/sk_geometry.npz

Scope: validated to reproduce `config/sk_geometry.npz` byte-for-byte (single-type cylindrical
geofile). Detectors with mPMT + outer-detector PMTs (HyperK, NuPRISM/WCTE) additionally split
into active vs `inactive_*` sub-PMT/OD arrays (see the schema's "inactive" section) which this
converter does not yet emit — their `.npz` ship pre-built in `config/`.
"""
import argparse
import numpy as np


def parse_geofile(path):
    radius_cm = height_cm = size_cm = None
    rows = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if line.startswith('Detector radius'):
                radius_cm, height_cm = float(s[-2]), float(s[-1])
            elif line.startswith('Type 1'):
                size_cm = float(s[-1])                       # active-PMT size (radius, cm)
            elif s[0].lstrip('-').replace('.', '', 1).isdigit() and len(s) >= 9:
                rows.append([float(x) for x in s])           # a PMT data row
    if radius_cm is None or size_cm is None:
        raise ValueError(f'{path}: could not parse the header (radius/height/size)')
    return radius_cm, height_cm, size_cm, np.array(rows)


def to_npz(path):
    radius_cm, height_cm, size_cm, r = parse_geofile(path)
    pmt_id = r[:, 0].astype(np.int32)
    pmt_type = r[:, 2].astype(np.int32)
    positions_mm = r[:, 3:6] * 10.0                          # cm -> mm
    directions = r[:, 6:9].astype(np.float64)
    radius_m, height_m = radius_cm / 100.0, height_cm / 100.0
    # surface: nearest face — cap if closer to a z-cap plane than to the barrel wall.
    x, y, z = positions_mm[:, 0], positions_mm[:, 1], positions_mm[:, 2]
    r_xy = np.hypot(x, y)
    d_wall = radius_m * 1000.0 - r_xy
    d_cap = height_m * 1000.0 / 2.0 - np.abs(z)
    surfaces = np.where(d_cap < d_wall, np.where(z > 0, 'top', 'bottom'), 'barrel').astype('<U6')
    return dict(positions_mm=positions_mm, directions=directions, surfaces=surfaces,
                pmt_id=pmt_id, pmt_type=pmt_type,
                radius=np.float64(radius_m), height=np.float64(height_m),
                sensor_radius=np.float64(size_cm / 100.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geofile', help='source geofile .txt (cm)')
    ap.add_argument('out', help='output .npz (schema: lucid/geometry/PMT_NPZ_SCHEMA.md)')
    args = ap.parse_args()
    d = to_npz(args.geofile)
    np.savez(args.out, **d)
    surf, cnt = np.unique(d['surfaces'], return_counts=True)
    print(f'{len(d["pmt_id"])} PMTs -> {args.out}')
    print(f'  radius {d["radius"]:.3f} m, height {d["height"]:.3f} m, sensor_radius {d["sensor_radius"]:.3f} m')
    print(f'  surfaces: {dict(zip(surf.tolist(), cnt.tolist()))}')


if __name__ == '__main__':
    main()
