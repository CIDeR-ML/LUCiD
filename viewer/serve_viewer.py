#!/usr/bin/env python3
"""LUCiD Viewer — local file server with HTTP Range support.

Serves LUCiD production HDF5 files (sensor/hits/step/labl) and the viewer
frontend. Files are matched by filename pattern ``wc_{kind}_NNNN.h5``
(or ``{dataset}_{kind}_NNNN.h5``).

Supports flat directories and subdirectory layouts
(``dataset_root/{sensor,hits,step,labl}/wc_*_NNNN.h5``), per
``docs/LUCID_DATASET.md``.

Usage:
    python3 viewer/serve_viewer.py production_run/
    python3 viewer/serve_viewer.py production_run/ --dataset wc
    python3 viewer/serve_viewer.py production_run/ --port 9000 --open
"""

import os
import re
import sys
import json
import argparse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from glob import glob

KINDS = ('sensor', 'hits', 'step', 'labl')


def find_h5_files(prod_dir):
    """Find all *_{kind}_*.h5 files, searching both flat and subdirectory layouts."""
    found = {}
    for kind in KINDS:
        files = []
        sub = os.path.join(prod_dir, kind)
        if os.path.isdir(sub):
            files += glob(os.path.join(sub, f'*_{kind}_*.h5'))
        files += glob(os.path.join(prod_dir, f'*_{kind}_*.h5'))
        files = sorted(set(os.path.abspath(f) for f in files))
        if files:
            found[kind] = files
    return found


def extract_dataset(filename):
    """Extract dataset name from '{dataset}_{kind}_{batch}.h5'."""
    base = os.path.basename(filename)
    for kind in KINDS:
        m = re.match(rf'^(.+)_{kind}_\d+\.h5$', base)
        if m:
            return m.group(1)
    return None


def extract_batch(filename):
    """Extract batch number NNNN."""
    base = os.path.basename(filename)
    m = re.match(r'^.+_[a-z]+_(\d+)\.h5$', base)
    return int(m.group(1)) if m else 0


def discover_datasets(prod_dir):
    """Return {dataset_name: {kind: [filepaths sorted by batch], ...}}."""
    all_files = find_h5_files(prod_dir)
    datasets = {}
    for kind, files in all_files.items():
        for f in files:
            ds = extract_dataset(f)
            if ds:
                datasets.setdefault(ds, {}).setdefault(kind, []).append(f)
    for ds in datasets:
        for kind in datasets[ds]:
            datasets[ds][kind].sort(key=extract_batch)
    return datasets


def select_dataset(prod_dir, requested=None):
    """Select a dataset and return the first batch's manifest dict."""
    datasets = discover_datasets(prod_dir)

    if not datasets:
        sys.exit(f"Error: no HDF5 files matching "
                 f"*_{{{','.join(KINDS)}}}_*.h5 found in {prod_dir}")

    if requested:
        if requested not in datasets:
            available = ', '.join(sorted(datasets.keys()))
            sys.exit(f"Error: dataset '{requested}' not found. Available: {available}")
        ds = requested
    elif len(datasets) == 1:
        ds = next(iter(datasets))
    else:
        complete = {k: v for k, v in datasets.items() if all(k2 in v for k2 in KINDS)}
        if len(complete) == 1:
            ds = next(iter(complete))
        else:
            candidates = complete or datasets
            print(f"Multiple datasets found in {prod_dir}:")
            for name, kinds in sorted(candidates.items()):
                labels = ', '.join(sorted(kinds.keys()))
                print(f"  {name}  ({labels})")
            sys.exit("Use --dataset <name> to select one.")

    info = datasets[ds]
    missing = [k for k in KINDS if k not in info]
    if missing:
        sys.exit(f"Error: dataset '{ds}' missing {', '.join(missing)} files")

    # For now, serve the first batch of each kind. (Multi-batch navigation
    # would add the other batches to the manifest; this MVP picks NNNN=0.)
    first = {k: info[k][0] for k in KINDS}
    return ds, first, info


def build_manifest(prod_dir, file_map):
    """Convert absolute paths to relative paths for the manifest."""
    return {kind: os.path.relpath(path, prod_dir) for kind, path in file_map.items()}


class RangeHandler(SimpleHTTPRequestHandler):
    base_dir = None
    project_dir = None
    manifest = None

    STATIC_FILES = {
        '/viewer.js':         ('viewer.js', 'application/javascript'),
        '/viewer.css':        ('viewer.css', 'text/css'),
        '/shaders.js':        ('shaders.js', 'application/javascript'),
        '/colormaps.js':      ('colormaps.js', 'application/javascript'),
        '/h5_worker.js':      ('h5_worker.js', 'application/javascript'),
        '/geometry_layout.js': ('geometry_layout.js', 'application/javascript'),
    }

    def do_HEAD(self):
        # Static viewer files live under project_dir; resolve them first.
        if self.path in self.STATIC_FILES:
            rel, _ct = self.STATIC_FILES[self.path]
            path = os.path.join(self.project_dir, rel)
        elif self.path == '/' or self.path.split('?')[0] == '/':
            path = os.path.join(self.project_dir, 'index.html')
        else:
            path = self._resolve_path()
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', os.path.getsize(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Expose-Headers',
                         'Content-Length, Content-Range, Accept-Ranges')
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path.split('?')[0] == '/':
            self._send_file(os.path.join(self.project_dir, 'index.html'), 'text/html')
            return
        if self.path in self.STATIC_FILES:
            rel, ct = self.STATIC_FILES[self.path]
            self._send_file(os.path.join(self.project_dir, rel), ct)
            return
        if self.path == '/manifest.json':
            body = json.dumps(self.manifest).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
            return
        path = self._resolve_path()
        if not path:
            self.send_error(404)
            return
        file_size = os.path.getsize(path)
        rng = self.headers.get('Range')
        if rng:
            try:
                spec = rng.replace('bytes=', '')
                parts = spec.split('-')
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
            except (ValueError, IndexError):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            self.send_header('Content-Length', length)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Expose-Headers',
                             'Content-Length, Content-Range, Accept-Ranges')
            self.end_headers()
            with open(path, 'rb') as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self._send_file(path, 'application/octet-stream')

    def _resolve_path(self):
        clean = self.path.split('?')[0].lstrip('/')
        if not clean:
            return None
        real = os.path.normpath(os.path.join(self.base_dir, clean))
        if not real.startswith(self.base_dir) or not os.path.isfile(real):
            return None
        return real

    def _send_file(self, path, content_type):
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if 'Range' not in (self.headers.get('Range') or ''):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(
        description='LUCiD Viewer — serve production HDF5 for browser visualization')
    parser.add_argument('data_dir', help='Directory containing production HDF5 files')
    parser.add_argument('--dataset', '-d', help='Dataset name (auto-detected if only one)')
    parser.add_argument('--port', '-p', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1',
                        help='Bind address. Default 127.0.0.1 (localhost only). '
                             'Use 0.0.0.0 to serve on all interfaces — e.g. when '
                             'viewing over an SSH tunnel pinned to a specific '
                             'remote node (ssh -L PORT:<node>:PORT ...).')
    parser.add_argument('--open', '-o', action='store_true',
                        help='Open browser automatically')
    args = parser.parse_args()

    prod_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(prod_dir):
        sys.exit(f"Error: {prod_dir} is not a directory")
    project_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_name, file_map, all_batches = select_dataset(prod_dir, args.dataset)
    manifest = build_manifest(prod_dir, file_map)

    print("=== LUCiD Viewer ===")
    print(f"Dataset: {dataset_name}")
    total_batches = max(len(all_batches[k]) for k in KINDS)
    if total_batches > 1:
        print(f"(found {total_batches} batches per kind — serving batch 0 only)")
    for k in KINDS:
        path = os.path.join(prod_dir, manifest[k])
        sz = os.path.getsize(path)
        label = f'{sz/1e9:.2f} GB' if sz > 1e8 else f'{sz/1e6:.1f} MB'
        print(f"  {k:8s}  {manifest[k]}  ({label})")

    RangeHandler.base_dir = prod_dir
    RangeHandler.project_dir = project_dir
    RangeHandler.manifest = manifest

    url = f'http://127.0.0.1:{args.port}/'

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedServer((args.host, args.port), RangeHandler)
    print(f"\n{url}")
    print("Ctrl+C to stop\n")

    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
