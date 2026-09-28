#!/usr/bin/env python3
"""Convert SKDetSim ``reflect.dec12.dat`` into a portable PMT response.

The converter reproduces the wavelength and angle-bin selection in RFPMSG.
For unpolarized photons it averages the s- and p-polarized absorptances, then
normalizes the selected curve to normal incidence.  The source table remains
external; its SHA-256 is stored in the output metadata for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


N_WAVELENGTHS = 108
N_ANGLES = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wavelength", type=float, default=405.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    raw = np.loadtxt(args.source)
    if raw.shape != (N_WAVELENGTHS * N_ANGLES, 8):
        raise ValueError(
            f"expected {(N_WAVELENGTHS * N_ANGLES, 8)}, got {raw.shape}")
    table = raw.reshape(N_WAVELENGTHS, N_ANGLES, 8)

    # RFPMSG uses iwav=int(lambda_nm - 336), then clamps its Fortran index to
    # [1, 108]. Convert that one-based index to NumPy's zero-based convention.
    fortran_iwav = int(args.wavelength - 336.0)
    fortran_iwav = min(N_WAVELENGTHS, max(1, fortran_iwav))
    block = table[fortran_iwav - 1]
    wavelength_nm = float(block[0, 0])
    if not np.allclose(block[:, 0], wavelength_nm):
        raise ValueError("selected SK wavelength block is inconsistent")

    angle_deg = block[:, 1]
    absorptance_s = block[:, 6]
    absorptance_p = block[:, 7]
    absorptance = 0.5 * (absorptance_s + absorptance_p)
    relative = absorptance / absorptance[0]
    incidence_cosine = np.cos(np.deg2rad(angle_deg))
    order = np.argsort(incidence_cosine)

    metadata = {
        "model": "SKDetSim RFPMSG unpolarized absorptance ratio",
        "source_path": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "requested_wavelength_nm": args.wavelength,
        "selected_fortran_iwav": fortran_iwav,
        "selected_table_wavelength_nm": wavelength_nm,
        "normal_incidence_absorptance": float(absorptance[0]),
        "normalization": "unpolarized absorptance / first angle-bin absorptance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        incidence_cosine=incidence_cosine[order],
        relative_efficiency=relative[order],
        angle_deg=angle_deg[order],
        absorptance=absorptance[order],
        absorptance_s=absorptance_s[order],
        absorptance_p=absorptance_p[order],
        wavelength_nm=np.asarray(wavelength_nm),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
