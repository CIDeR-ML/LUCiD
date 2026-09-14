#!/usr/bin/env python
"""Extract the two radial StrutSteel tube families from JUNOSW GDML."""

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET


FAMILIES = {
    "lSteel0x92ad230": ("main_85mm", "sStrut0x92ad140"),
    "lSteel20x9300cd0": ("secondary_60mm", "sStrut0x9300be0"),
}


def extract(gdml_path):
    """Return support placements and dimensions in metres."""
    root = ET.parse(gdml_path).getroot()
    tubes = {
        item.attrib["name"]: item for item in root.findall(".//tube")
    }
    rows = []
    family_counts = {name: 0 for name in FAMILIES}
    for physical in root.findall(".//physvol"):
        volume_ref = physical.find("volumeref")
        position = physical.find("position")
        rotation = physical.find("rotation")
        if volume_ref is None or position is None:
            continue
        volume_name = volume_ref.attrib.get("ref")
        if volume_name not in FAMILIES:
            continue
        family_name, solid_name = FAMILIES[volume_name]
        tube = tubes[solid_name]
        scale = (
            0.001 if tube.attrib.get("lunit", "mm") == "mm" else 1.0
        )
        copy_no = int(physical.attrib.get(
            "copynumber", family_counts[volume_name]
        ))
        family_counts[volume_name] += 1
        if rotation is None or rotation.attrib.get("unit", "deg") != "deg":
            raise ValueError("Support placement lacks a degree rotation")
        angle_x = math.radians(float(rotation.attrib["x"]))
        angle_y = math.radians(float(rotation.attrib["y"]))
        local_z = (
            -math.sin(angle_y),
            math.sin(angle_x) * math.cos(angle_y),
            math.cos(angle_x) * math.cos(angle_y),
        )
        centre_mm = tuple(
            float(position.attrib[name]) for name in ("x", "y", "z")
        )
        centre_norm = math.sqrt(sum(value**2 for value in centre_mm))
        radial_inward = tuple(-value / centre_norm for value in centre_mm)
        alignment = sum(
            value * reference
            for value, reference in zip(local_z, radial_inward)
        )
        if alignment < 1.0 - 1e-10:
            raise ValueError(
                f"Support {physical.attrib.get('name')} is not radial: "
                f"axis dot radial={alignment}"
            )
        rows.append({
            "copy_no": copy_no,
            "family": family_name,
            "center_x_m": centre_mm[0] * scale,
            "center_y_m": centre_mm[1] * scale,
            "center_z_m": centre_mm[2] * scale,
            "radius_m": float(tube.attrib["rmax"]) * scale,
            "half_length_m": 0.5 * float(tube.attrib["z"]) * scale,
        })
    expected = {
        "lSteel0x92ad230": 370,
        "lSteel20x9300cd0": 220,
    }
    if family_counts != expected:
        raise ValueError(
            f"Unexpected support counts {family_counts}; expected {expected}"
        )
    return rows


def main():
    """Write a portable CSV in deterministic GDML placement order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gdml", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = extract(args.gdml)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
