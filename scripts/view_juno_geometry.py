#!/usr/bin/env python3
"""Serve an instanced Three.js view of the effective JUNO geometry in LUCiD."""

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = PROJECT_ROOT / "viewer" / "geometry"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lucid.geometry import generate_detector  # noqa: E402
from lucid.geometry.juno_pmt import (  # noqa: E402
    HAMAMATSU,
    HIGH_QE_NNVT,
    NNVT,
)


PMT_STYLES = {
    HAMAMATSU: ("Hamamatsu", "#ef8354"),
    NNVT: ("NNVT", "#4f8fc0"),
    HIGH_QE_NNVT: ("High-QE NNVT", "#8ac926"),
}
SUPPORT_STYLES = {
    0: ("Main 85 mm struts", "#d1495b"),
    1: ("Secondary 60 mm struts", "#9d4edd"),
}


def _packed_rows(*columns):
    """Pack columns as little-endian float32 rows for zero-copy browser use."""
    return np.column_stack(columns).astype("<f4", copy=False).tobytes()


def build_geometry_payload(detector):
    """Return a JSON manifest and binary instance buffers for a detector."""
    positions = np.asarray(detector.all_points, dtype=np.float32)
    count = len(positions)
    copy_numbers = getattr(detector, "pmt_copy_numbers", None)
    if copy_numbers is None:
        copy_numbers = np.arange(count, dtype=np.int32)
    type_codes = getattr(detector, "pmt_type_codes", None)
    if type_codes is None:
        type_codes = np.full(count, -1, dtype=np.int8)
    axes = getattr(detector, "pmt_inward_axes", None)
    if axes is None:
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        axes = -positions / np.maximum(norms, 1e-12)

    dimensions = getattr(detector, "pmt_dimensions", None)
    explicit_front_heads = dimensions is not None
    if dimensions is None:
        radial = np.full(count, detector.S_radius, dtype=np.float32)
        axial = np.full(count, 0.02, dtype=np.float32)
    else:
        radial = np.asarray(dimensions.outer_radial_m, dtype=np.float32)
        axial = np.asarray(dimensions.outer_axial_m, dtype=np.float32)

    payloads = {}
    pmt_groups = []
    styles = dict(PMT_STYLES)
    styles[-1] = ("PMTs", "#f2c14e")
    for code, (name, color) in styles.items():
        selected = np.asarray(type_codes) == code
        group_count = int(np.count_nonzero(selected))
        if not group_count:
            continue
        url = f"/data/pmt_{code}.f32"
        payloads[url] = _packed_rows(
            positions[selected, 0], positions[selected, 1], positions[selected, 2],
            np.asarray(axes)[selected, 0], np.asarray(axes)[selected, 1],
            np.asarray(axes)[selected, 2], radial[selected], axial[selected],
            np.asarray(copy_numbers)[selected],
        )
        pmt_groups.append({
            "code": int(code),
            "name": name,
            "color": color,
            "count": group_count,
            "url": url,
            "stride": 9,
        })

    support_groups = []
    rods = getattr(detector, "support_rods", None)
    if rods is not None:
        for code, (name, color) in SUPPORT_STYLES.items():
            selected = np.asarray(rods.family_codes) == code
            group_count = int(np.count_nonzero(selected))
            if not group_count:
                continue
            url = f"/data/support_{code}.f32"
            payloads[url] = _packed_rows(
                rods.centers_m[selected, 0], rods.centers_m[selected, 1],
                rods.centers_m[selected, 2], rods.inward_axes[selected, 0],
                rods.inward_axes[selected, 1], rods.inward_axes[selected, 2],
                rods.radii_m[selected], rods.half_lengths_m[selected],
                rods.copy_numbers[selected],
            )
            support_groups.append({
                "code": int(code),
                "name": name,
                "color": color,
                "count": group_count,
                "url": url,
                "stride": 9,
            })

    boundaries = []
    boundary_specs = (
        ("r_inner", "LS-Acrylic boundary", "#f4d35e", True),
        ("r_acrylic", "Acrylic-Water boundary", "#48cae4", True),
        ("r_outer", "PMT equator surface", "#adb5bd", False),
    )
    for attribute, name, color, visible in boundary_specs:
        radius = getattr(detector, attribute, None)
        if radius is not None:
            boundaries.append({
                "name": name,
                "radius": float(radius),
                "color": color,
                "visible": visible,
            })
    if not boundaries and hasattr(detector, "r"):
        boundaries.append({
            "name": "Detector boundary",
            "radius": float(detector.r),
            "color": "#adb5bd",
            "visible": True,
        })

    manifest = {
        "format": "lucid-instanced-geometry-v1",
        "detector_type": type(detector).__name__,
        "pmt_shape": "front_head" if explicit_front_heads else "disc",
        "pmt_count": count,
        "support_count": 0 if rods is None else len(rods.centers_m),
        "boundaries": boundaries,
        "pmt_groups": pmt_groups,
        "support_groups": support_groups,
    }
    return manifest, payloads


def make_handler(manifest, payloads):
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    static_files = {
        "/": (VIEWER_ROOT / "index.html", "text/html; charset=utf-8"),
        "/index.html": (VIEWER_ROOT / "index.html", "text/html; charset=utf-8"),
        "/geometry.js": (VIEWER_ROOT / "geometry.js", "text/javascript; charset=utf-8"),
        "/geometry.css": (VIEWER_ROOT / "geometry.css", "text/css; charset=utf-8"),
    }

    class GeometryHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/geometry.json":
                self._send(manifest_bytes, "application/json")
                return
            if path in payloads:
                self._send(payloads[path], "application/octet-stream")
                return
            if path in static_files:
                file_path, content_type = static_files[path]
                try:
                    body = file_path.read_bytes()
                except FileNotFoundError:
                    self.send_error(404)
                    return
                self._send(body, content_type)
                return
            self.send_error(404)

        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            if args and str(args[0]).startswith("GET /data/"):
                return
            super().log_message(format, *args)

    return GeometryHandler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/JUNO_laser_geom_config.json"),
        help="LUCiD geometry config (default: %(default)s)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the viewer in a browser (default: true)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detector = generate_detector(args.config)
    manifest, payloads = build_geometry_payload(detector)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(manifest, payloads)
    )
    url = f"http://{args.host}:{args.port}/"
    binary_size = sum(map(len, payloads.values()))
    print("=== LUCiD JUNO Geometry Viewer ===")
    print(f"Config:   {args.config}")
    print(f"PMTs:     {manifest['pmt_count']:,}")
    print(f"Supports: {manifest['support_count']:,}")
    print(f"Payload:  {binary_size / 1_000_000:.2f} MB")
    print(f"URL:      {url}")
    print("Ctrl+C to stop")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
