#!/usr/bin/env python
"""Extract JUNO Acrylic-addition placements from J26.3.2 GDML."""

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET


VOLUME_NAME = "lAddition0x93fdf80"
SOLID_NAME = "solidAddition_down0x93fc180"


def extract(gdml_path):
    """Return Acrylic-addition placements and effective dimensions."""
    root = ET.parse(gdml_path).getroot()
    solid = root.find(f".//polycone[@name='{SOLID_NAME}']")
    if solid is None or solid.attrib.get("lunit", "mm") != "mm":
        raise ValueError("Could not find the expected millimetre polycone")
    planes = solid.findall("zplane")
    z_values = [float(item.attrib["z"]) for item in planes]
    radii = [float(item.attrib["rmax"]) for item in planes]
    base_radius_m = max(radii) * 0.001
    tip_radius_m = radii[-1] * 0.001
    length_m = (0.0 - min(z_values)) * 0.001

    rows = []
    for physical in root.findall(".//physvol"):
        volume_ref = physical.find("volumeref")
        position = physical.find("position")
        if volume_ref is None or position is None:
            continue
        if volume_ref.attrib.get("ref") != VOLUME_NAME:
            continue
        copy_no = int(physical.attrib.get("copynumber", len(rows)))
        xyz_m = [
            float(position.attrib[name]) * 0.001
            for name in ("x", "y", "z")
        ]
        radius_m = math.sqrt(sum(value**2 for value in xyz_m))
        if abs(radius_m - 17.824) > 2e-6:
            raise ValueError(
                f"Acrylic addition {copy_no} is not on r=17.824 m"
            )
        rows.append({
            "copy_no": copy_no,
            "x_m": xyz_m[0],
            "y_m": xyz_m[1],
            "z_m": xyz_m[2],
            "base_radius_m": base_radius_m,
            "tip_radius_m": tip_radius_m,
            "length_m": length_m,
        })
    if len(rows) != 590:
        raise ValueError(f"Expected 590 Acrylic additions, found {len(rows)}")
    return rows


def main():
    """Write a portable CSV in deterministic GDML placement order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gdml", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = extract(args.gdml)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
