#!/usr/bin/env python3
"""
Validate per-energy PhotonSim ROOT files before building the SIREN lookup table.

Scans `<data-dir>/<E>MeV/output.root` and reports missing, empty, corrupted,
and good files (with the photon count from `PhotonHist_AngleDistance`).

Used as a sanity check before invoking `lucid-build-photon-table`. See
`docs/SIREN_TRAINING_INPUTS.md` for the full pipeline.
"""

import uproot
from pathlib import Path
import os

def check_root_files(path="data/mu-"):
    """Check all ROOT files and report their status."""
    data_dir = Path(path)

    # Track file status
    good_files = []
    empty_files = []
    missing_files = []
    corrupted_files = []

    print("Checking ROOT files...\n")

    # Check each energy directory
    for energy in range(100, 1010, 10):
        energy_dir = data_dir / f"{energy}MeV"
        root_file = energy_dir / "output.root"

        if not root_file.exists():
            missing_files.append(energy)
            print(f"{energy:4d} MeV: MISSING")
            continue

        # Check file size
        file_size = os.path.getsize(root_file)

        if file_size == 0:
            empty_files.append(energy)
            print(f"{energy:4d} MeV: EMPTY FILE (0 bytes)")
            continue

        # Try to open and check contents
        try:
            with uproot.open(root_file) as f:
                keys = list(f.keys())

                if len(keys) == 0:
                    empty_files.append(energy)
                    print(f"{energy:4d} MeV: NO OBJECTS (size: {file_size/1024/1024:.1f} MB)")
                elif any("PhotonHist_AngleDistance" in key for key in keys):
                    # Access with the cycle number
                    hist_key = next(key for key in keys if "PhotonHist_AngleDistance" in key)
                    hist = f[hist_key]
                    photons = hist.values().sum()
                    good_files.append(energy)
                    print(f"{energy:4d} MeV: OK - {photons:,.0f} photons (size: {file_size/1024/1024:.1f} MB)")
                else:
                    print(f"{energy:4d} MeV: MISSING HISTOGRAM (has: {keys})")
                    corrupted_files.append(energy)

        except Exception as e:
            corrupted_files.append(energy)
            print(f"{energy:4d} MeV: CORRUPTED - {str(e)}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total files checked: 91")
    if good_files:
        print(f"Good files: {len(good_files)} ({good_files[0]}-{good_files[-1]} MeV)")
    else:
        print(f"Good files: 0")
    print(f"Empty files: {len(empty_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")

    if empty_files:
        print(f"\nEmpty files at energies: {empty_files}")
    if missing_files:
        print(f"Missing files at energies: {missing_files}")
    if corrupted_files:
        print(f"Corrupted files at energies: {corrupted_files}")

    # Check for pattern
    print("\n=== ANALYSIS ===")
    if empty_files:
        # Check if it's a continuous range
        if len(empty_files) > 1:
            continuous = all(empty_files[i+1] - empty_files[i] == 10 for i in range(len(empty_files)-1))
            if continuous:
                print(f"Continuous range of empty files: {empty_files[0]}-{empty_files[-1]} MeV")

if __name__ == '__main__':
    check_root_files()
