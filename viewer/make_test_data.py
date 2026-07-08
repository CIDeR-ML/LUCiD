#!/usr/bin/env python3
"""Synthesize a minimal v5 LUCiD dataset for viewer smoke tests.

Produces four HDF5 files matching `docs/LUCID_DATASET.md` — sensor, hits,
step, labl — with a small number of events and synthetic but reasonable
content. Intended to exercise the browser viewer without running the
full production pipeline. Emits the v5 schema: per_interaction/ with
one row per source interaction (not per primary), per_interaction fields
include CSR primary_{track_ids,pdgs,energies} lists plus neutrino probe
metadata, and per_event/t0 derived as min(per_interaction/t0).

Usage:
    python3 make_test_data.py --out ./test_data
    python3 make_test_data.py --out ./test_data --geom box
    python3 make_test_data.py --out ./test_data --geom sphere --events 5
"""

import argparse
import os
from pathlib import Path
import numpy as np
import h5py


def place_cylinder(n_sensors, r, halfH):
    barrel_area = 2 * np.pi * r * 2 * halfH
    cap_area = np.pi * r * r
    n_barrel = int(n_sensors * barrel_area / (barrel_area + 2 * cap_area))
    n_cap = (n_sensors - n_barrel) // 2

    # Barrel grid.
    n_rows = max(2, int(np.sqrt(n_barrel * (2 * halfH) / (2 * np.pi * r))))
    n_cols = max(2, n_barrel // n_rows)
    zs = np.linspace(-halfH + 0.5, halfH - 0.5, n_rows)
    thetas = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)
    pts = []
    for z in zs:
        for t in thetas:
            pts.append([r * np.cos(t), r * np.sin(t), z])
    pts = pts[:n_barrel]
    # Top and bottom cap — Fibonacci disk.
    for sign in (+1, -1):
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n_cap):
            rad = r * np.sqrt((i + 0.5) / n_cap) * 0.9
            ang = i * 2 * np.pi / phi
            pts.append([rad * np.cos(ang), rad * np.sin(ang), sign * halfH])
    pts = np.array(pts[:n_sensors], dtype=np.float32)
    return pts


def place_sphere(n_sensors, r):
    phi = (1 + np.sqrt(5)) / 2
    out = np.empty((n_sensors, 3), dtype=np.float32)
    for i in range(n_sensors):
        y = 1 - (i / max(1, n_sensors - 1)) * 2
        rad = np.sqrt(1 - y * y)
        theta = 2 * np.pi * i / phi
        out[i] = [rad * np.cos(theta) * r, y * r, rad * np.sin(theta) * r]
    return out


def place_box(n_sensors, L, W, H):
    # Proportional by face area.
    A_fb = 2 * L * H  # front + back
    A_lr = 2 * W * H  # left + right
    A_tb = 2 * L * W  # top + bottom
    total = A_fb + A_lr + A_tb
    n_fb = int(n_sensors * A_fb / total) // 2 * 2
    n_lr = int(n_sensors * A_lr / total) // 2 * 2
    n_tb = n_sensors - n_fb - n_lr
    n_fb //= 2; n_lr //= 2; n_tb //= 2

    pts = []
    def face_grid(n, d1, d2, place_fn):
        if n <= 0: return []
        nr = max(2, int(np.sqrt(n * d2 / d1)))
        nc = max(2, n // nr)
        g1 = np.linspace(-d1/2 + 0.3, d1/2 - 0.3, nc)
        g2 = np.linspace(-d2/2 + 0.3, d2/2 - 0.3, nr)
        out = [place_fn(a, b) for b in g2 for a in g1][:n]
        return out
    # Front/back  y = ±W/2
    pts += face_grid(n_fb, L, H, lambda x, z: [x,  W/2, z])
    pts += face_grid(n_fb, L, H, lambda x, z: [x, -W/2, z])
    # Left/right  x = ±L/2
    pts += face_grid(n_lr, W, H, lambda y, z: [ L/2, y, z])
    pts += face_grid(n_lr, W, H, lambda y, z: [-L/2, y, z])
    # Top/bottom z = ±H/2
    pts += face_grid(n_tb, L, W, lambda x, y: [x, y,  H/2])
    pts += face_grid(n_tb, L, W, lambda x, y: [x, y, -H/2])
    return np.array(pts[:n_sensors], dtype=np.float32)


def write_common_config(cfg_grp, n_events, provenance_extras=None):
    cfg_grp.attrs['format_version'] = 5
    cfg_grp.attrs['n_events'] = n_events
    cfg_grp.attrs['git_commit'] = 'test-stub'
    cfg_grp.attrs['run_id'] = 'test-stub-000'
    cfg_grp.attrs['dataset_name'] = 'viewer-smoke-test'
    cfg_grp.attrs['file_index'] = 0
    cfg_grp.attrs['source_file'] = 'synthetic'
    cfg_grp.attrs['lucid_master_seed'] = 42
    cfg_grp.attrs['photonsim_seed'] = -1
    if provenance_extras:
        for k, v in provenance_extras.items():
            cfg_grp.attrs[k] = v


def gen_event(rng, n_sensors, sensor_positions, n_particles=6):
    """Generate one synthetic event. Returns a dict of arrays per file."""
    # Sensor hits: each particle lights up a random radial cluster.
    sensor_hits = {}   # sensor_idx -> (PE, T)
    hits_rows = []     # list of (particle, sensor, PE, T)
    for p in range(n_particles):
        # Random "direction" — pick 3-5 clusters of 10-60 sensors.
        n_clusters = rng.integers(3, 6)
        for _ in range(n_clusters):
            # Pick a random sensor as cluster center, hit its neighbors.
            center_idx = rng.integers(0, n_sensors)
            center = sensor_positions[center_idx]
            dists = np.linalg.norm(sensor_positions - center, axis=1)
            n_in_cluster = rng.integers(8, 40)
            idx_sorted = np.argsort(dists)[:n_in_cluster]
            base_t = rng.uniform(0, 50)
            for s in idx_sorted:
                pe = rng.lognormal(0.0, 0.8)  # typical small PE
                t = base_t + rng.normal(0, 1.5)
                hits_rows.append((p, int(s), float(pe), float(t)))
                if s in sensor_hits:
                    prev_pe, prev_t = sensor_hits[s]
                    sensor_hits[s] = (prev_pe + pe, min(prev_t, t))
                else:
                    sensor_hits[s] = (pe, t)

    sensor_idx = np.array(sorted(sensor_hits.keys()), dtype=np.uint16)
    sensor_PE = np.array([sensor_hits[s][0] for s in sensor_idx], dtype=np.float32)
    sensor_T = np.array([sensor_hits[s][1] for s in sensor_idx], dtype=np.float32)

    hits_arr = np.array(hits_rows, dtype=[
        ('p', 'i4'), ('s', 'u2'), ('pe', 'f4'), ('t', 'f4')])
    hits_arr.sort(order=['p', 's'])

    # Segments: one straight line per particle, 30-80 segments each.
    edep_rows = []
    track_rows = []
    track_idx = 0
    for p in range(n_particles):
        # Each particle gets 1-3 tracks.
        n_tracks = rng.integers(1, 4)
        for k in range(n_tracks):
            n_seg = rng.integers(30, 80)
            # Straight line in a random direction, starting near origin.
            start = rng.uniform(-2, 2, size=3).astype(np.float32)
            direction = rng.normal(0, 1, size=3)
            direction = (direction / np.linalg.norm(direction)).astype(np.float32)
            step = 0.4
            t0_track = rng.uniform(0, 5)
            for i in range(n_seg):
                s = start + direction * (i * step)
                e = start + direction * ((i + 1) * step)
                time = t0_track + i * 0.015   # ~ns/step at c
                edep = max(0.01, rng.normal(2.0, 0.5))
                beta = min(0.99, rng.normal(0.85, 0.1))
                ncher = int(max(0, rng.normal(5, 2)))
                edep_rows.append((track_idx, s[0], s[1], s[2], e[0], e[1], e[2],
                                 direction[0], direction[1], direction[2],
                                 time, edep, beta, ncher))
            pdg = int(rng.choice([13, -13, 11, -11, 22, 2212, 211]))
            track_rows.append((track_idx, 0, pdg, rng.uniform(100, 1500),
                               sum(1 for r in edep_rows[-n_seg:] for _ in [r]) * 5, p))
            track_idx += 1

    edep_rows = np.array(edep_rows, dtype=[
        ('track_idx', 'i4'),
        ('start_x', 'f4'), ('start_y', 'f4'), ('start_z', 'f4'),
        ('end_x', 'f4'), ('end_y', 'f4'), ('end_z', 'f4'),
        ('dir_x', 'f4'), ('dir_y', 'f4'), ('dir_z', 'f4'),
        ('time', 'f4'), ('edep', 'f4'), ('beta', 'f4'), ('ncher', 'i4')])
    # Synthetic per-segment contained flag (random; this stub doesn't
    # know the detector geometry). Real datasets compute it against
    # detector_bounds in `_compute_contained`.
    edep_contained = rng.integers(0, 2, size=len(edep_rows)).astype(bool)
    track_rows = np.array(track_rows, dtype=[
        ('track_id', 'i4'), ('parent_id', 'i4'), ('pdg', 'i2'),
        ('init_e', 'f4'), ('n_cher', 'i4'), ('particle_idx', 'i4')])

    # labl: categories and genealogy stubs.
    categories = rng.integers(0, 4, size=n_particles, endpoint=False).astype(np.uint8)
    contained_per_particle = rng.integers(0, 2, size=n_particles).astype(bool)

    # Genealogy stub: empty chains (vlen not critical for viewer MVP).
    gen_data = np.array([], dtype=np.int32)
    gen_off = np.zeros(n_particles + 1, dtype=np.uint32)

    # v5 per_interaction: one row per source interaction. This stub models
    # a 2-interaction event (akin to pile-up) — half the particles belong
    # to interaction 0 and the other half to interaction 1 — so the viewer
    # exercises the multi-interaction code path. Each interaction row
    # records its full primary list via CSR offsets+data, plus synthetic
    # source/neutrino metadata.
    n_interactions = 2 if n_particles >= 2 else 1
    # Split particles across interactions: first half → interaction 0, rest → 1.
    particle_to_interaction = np.concatenate([
        np.zeros(n_particles // n_interactions, dtype=np.int32),
        np.ones(n_particles - n_particles // n_interactions, dtype=np.int32),
    ])[:n_particles]
    if n_interactions == 1:
        particle_to_interaction = np.zeros(n_particles, dtype=np.int32)

    per_interaction_t0 = rng.uniform(-250.0, 250.0, size=n_interactions).astype(np.float32)
    source_type_arr = np.array(
        [0] + [1] * (n_interactions - 1), dtype=np.uint8)   # first gun, rest "genie"
    neutrino_pdg = np.where(source_type_arr == 1, np.int16(14), np.int16(0))
    neutrino_energy_MeV = np.where(
        source_type_arr == 1,
        rng.uniform(200.0, 2000.0, size=n_interactions).astype(np.float32),
        np.float32(0.0),
    ).astype(np.float32)
    per_interaction_contained = rng.integers(
        0, 2, size=n_interactions).astype(bool)

    # Per-interaction primary lists — pick the first track of each particle
    # in that interaction. In this stub each particle has one track cluster,
    # so n_primaries per interaction equals n_particles in that interaction.
    primary_tid_chunks = [[] for _ in range(n_interactions)]
    primary_pdg_chunks = [[] for _ in range(n_interactions)]
    primary_e_chunks   = [[] for _ in range(n_interactions)]
    seen_particle = set()
    for t_row in track_rows:
        p = int(t_row['particle_idx'])
        if p in seen_particle:
            continue
        seen_particle.add(p)
        i = int(particle_to_interaction[p])
        primary_tid_chunks[i].append(int(t_row['track_id']))
        primary_pdg_chunks[i].append(int(t_row['pdg']))
        primary_e_chunks[i].append(float(t_row['init_e']))

    def _csr(chunks, dtype):
        offsets = np.zeros(len(chunks) + 1, dtype=np.uint32)
        for i, c in enumerate(chunks):
            offsets[i + 1] = offsets[i] + len(c)
        data = np.array([x for c in chunks for x in c], dtype=dtype)
        return offsets, data

    ptid_off, ptid_data = _csr(primary_tid_chunks, np.int32)
    ppdg_off, ppdg_data = _csr(primary_pdg_chunks, np.int16)
    pen_off,  pen_data  = _csr(primary_e_chunks,   np.float32)

    # per_particle/interaction_idx FK.
    interaction_idx = particle_to_interaction.copy()

    # per_track derived: ancestor = its particle's first primary; interaction = its particle's interaction.
    ancestor_per_particle = np.zeros(n_particles, dtype=np.int32)
    for t_row in track_rows:
        p = int(t_row['particle_idx'])
        if ancestor_per_particle[p] == 0:
            ancestor_per_particle[p] = int(t_row['track_id'])
    track_ancestor = np.array(
        [ancestor_per_particle[int(r['particle_idx'])] for r in track_rows],
        dtype=np.int32)
    track_interaction = np.array(
        [particle_to_interaction[int(r['particle_idx'])] for r in track_rows],
        dtype=np.int32)

    n_particles_per_interaction = np.bincount(
        particle_to_interaction, minlength=n_interactions).astype(np.int32)
    n_primaries_per_interaction = np.array(
        [len(c) for c in primary_tid_chunks], dtype=np.int32)

    return {
        'sensor': {'sensor_idx': sensor_idx, 'PE': sensor_PE, 'T': sensor_T},
        'hits':   {'particle_idx': hits_arr['p'].astype(np.int32),
                   'sensor_idx': hits_arr['s'],
                   'PE': hits_arr['pe'], 'T': hits_arr['t']},
        'step':    {**{k: edep_rows[k] for k in edep_rows.dtype.names},
                   'contained': edep_contained},
        'labl':   {
            # per_event/t0 = min(per_interaction/t0) — the earliest
            # interaction time in the event, used by downstream tools
            # (viewer) as a single-scalar reference without walking
            # per_interaction.
            'per_event':    {'t0': np.float32(float(per_interaction_t0.min())),
                             'contained': bool(per_interaction_contained.all() and n_interactions > 0)},
            'per_interaction': {
                'source_type':                 source_type_arr,
                't0':                          per_interaction_t0,
                'vertex_x':                    rng.uniform(-1, 1, size=n_interactions).astype(np.float32),
                'vertex_y':                    rng.uniform(-1, 1, size=n_interactions).astype(np.float32),
                'vertex_z':                    rng.uniform(-1, 1, size=n_interactions).astype(np.float32),
                'n_primaries':                 n_primaries_per_interaction,
                'n_particles':                 n_particles_per_interaction,
                'neutrino_pdg':                neutrino_pdg,
                'neutrino_energy_MeV':         neutrino_energy_MeV,
                'contained':                   per_interaction_contained,
                'primary_track_ids_offsets':   ptid_off,
                'primary_track_ids_data':      ptid_data,
                'primary_pdgs_offsets':        ppdg_off,
                'primary_pdgs_data':           ppdg_data,
                'primary_energies_offsets':    pen_off,
                'primary_energies_data':       pen_data,
            },
            'per_particle': {'category': categories, 'contained': contained_per_particle,
                             'genealogy_data': gen_data, 'genealogy_offsets': gen_off,
                             'ext_genealogy_data': gen_data.copy(),
                             'ext_genealogy_offsets': gen_off.copy(),
                             'interaction_idx': interaction_idx},
            'per_track':    {'track_id': track_rows['track_id'],
                             'parent_id': track_rows['parent_id'],
                             'pdg': track_rows['pdg'],
                             'initial_energy': track_rows['init_e'],
                             'n_cherenkov': track_rows['n_cher'],
                             'particle_idx': track_rows['particle_idx'],
                             'ancestor': track_ancestor,
                             'interaction': track_interaction},
            'n_particles': n_particles,
            'n_tracks':    len(track_rows),
        },
        'n_segments': len(edep_rows),
    }


def write_dataset(out_dir, geom, n_events, n_sensors, seed):
    rng = np.random.default_rng(seed)

    # Geometry.
    if geom == 'cylinder':
        r, halfH = 16.9, 18.1
        sensor_positions = place_cylinder(n_sensors, r, halfH)
        detector_bbox = np.array([-r, -r, -halfH, r, r, halfH], dtype=np.float32)
        shape_attrs = {'detector_radius': np.float32(r), 'detector_half_height': np.float32(halfH)}
    elif geom == 'sphere':
        r = 17.5
        sensor_positions = place_sphere(n_sensors, r)
        detector_bbox = np.array([-r, -r, -r, r, r, r], dtype=np.float32)
        shape_attrs = {'detector_radius': np.float32(r), 'detector_half_height': np.float32(r)}
    elif geom == 'box':
        L, W, H = 12.0, 6.0, 5.0
        sensor_positions = place_box(n_sensors, L, W, H)
        detector_bbox = np.array([-L/2, -W/2, -H/2, L/2, W/2, H/2], dtype=np.float32)
        shape_attrs = {'detector_radius': np.float32(max(L, W, H) / 2),
                       'detector_half_height': np.float32(H / 2)}
    else:
        raise SystemExit(f"unknown geom: {geom}")

    n_sensors = len(sensor_positions)

    # Build subdirectories.
    out = Path(out_dir)
    for k in ('sensor', 'hits', 'step', 'labl'):
        (out / k).mkdir(parents=True, exist_ok=True)

    paths = {
        'sensor': out / 'sensor' / 'wc_sensor_0000.h5',
        'hits':   out / 'hits'   / 'wc_hits_0000.h5',
        'step':    out / 'step'    / 'wc_step_0000.h5',
        'labl':   out / 'labl'   / 'wc_labl_0000.h5',
    }
    for p in paths.values():
        if p.exists(): p.unlink()

    source_event_idx = np.arange(n_events, dtype=np.uint32)

    # Create all four files and populate configs.
    fs = {k: h5py.File(p, 'w') for k, p in paths.items()}

    # sensor config
    sc = fs['sensor'].create_group('config')
    write_common_config(sc, n_events)
    sc.attrs['n_sensors'] = n_sensors
    sc.attrs['detector_type'] = geom
    sc.attrs['material'] = 'water'
    sc.attrs['smearing_applied'] = True
    sc.attrs['smearing_charge_function'] = 'SK_like'
    sc.attrs['smearing_time_function'] = 'SK_like'
    sc.create_dataset('source_event_idx', data=source_event_idx)
    sc.create_dataset('sensor_positions', data=sensor_positions)

    # hits config
    ic = fs['hits'].create_group('config')
    write_common_config(ic, n_events)
    ic.attrs['n_sensors'] = n_sensors
    ic.attrs['detector_type'] = geom
    ic.attrs['material'] = 'water'
    ic.create_dataset('source_event_idx', data=source_event_idx)
    ic.create_dataset('sensor_positions', data=sensor_positions)

    # step config
    gc = fs['step'].create_group('config')
    write_common_config(gc, n_events)
    gc.attrs['detector_type'] = geom
    gc.attrs['detector_shape'] = geom
    gc.attrs['material'] = 'water'
    for k, v in shape_attrs.items(): gc.attrs[k] = v
    gc.create_dataset('detector_bbox', data=detector_bbox)
    gc.create_dataset('detector_axis', data=np.array([0, 0, 1], dtype=np.float32))
    gc.create_dataset('source_event_idx', data=source_event_idx)

    # labl config
    lc = fs['labl'].create_group('config')
    write_common_config(lc, n_events)
    lc.attrs['label_names'] = ['category']
    lc.create_dataset('source_event_idx', data=source_event_idx)

    # Per-event.
    for e in range(n_events):
        k = f'event_{e:03d}'
        ev = gen_event(rng, n_sensors, sensor_positions)

        # sensor
        g = fs['sensor'].create_group(k)
        g.attrs['source_event_idx'] = np.uint32(e)
        g.attrs['n_hits'] = len(ev['sensor']['sensor_idx'])
        for name, arr in ev['sensor'].items():
            g.create_dataset(name, data=arr)

        # hits
        g = fs['hits'].create_group(k)
        g.attrs['source_event_idx'] = np.uint32(e)
        g.attrs['n_particles'] = ev['labl']['n_particles']
        g.attrs['n_particle_hits'] = len(ev['hits']['particle_idx'])
        for name, arr in ev['hits'].items():
            g.create_dataset(name, data=arr)

        # step
        g = fs['step'].create_group(k)
        g.attrs['source_event_idx'] = np.uint32(e)
        g.attrs['n_tracks'] = ev['labl']['n_tracks']
        g.attrs['n_segments'] = ev['n_segments']
        name_map = {'beta': 'beta_start', 'ncher': 'n_cherenkov'}
        for name, arr in ev['step'].items():
            g.create_dataset(name_map.get(name, name), data=arr)

        # labl
        g = fs['labl'].create_group(k)
        g.attrs['source_event_idx'] = np.uint32(e)
        g.attrs['n_particles'] = ev['labl']['n_particles']
        g.attrs['n_tracks'] = ev['labl']['n_tracks']
        pe_g = g.create_group('per_event')
        pe_g.create_dataset('t0', data=ev['labl']['per_event']['t0'])
        pe_g.create_dataset('contained',
                            data=np.bool_(ev['labl']['per_event']['contained']))
        pi_g = g.create_group('per_interaction')
        for name, arr in ev['labl']['per_interaction'].items():
            pi_g.create_dataset(name, data=arr)
        pp_g = g.create_group('per_particle')
        for name, arr in ev['labl']['per_particle'].items():
            pp_g.create_dataset(name, data=arr)
        pt_g = g.create_group('per_track')
        for name, arr in ev['labl']['per_track'].items():
            pt_g.create_dataset(name, data=arr)

    for f in fs.values(): f.close()
    print(f'Wrote {n_events} events × 4 files to {out}/')
    print(f'  geom={geom}  n_sensors={n_sensors}')
    print(f'  serve with: python3 serve_viewer.py {out} --open')


def main():
    ap = argparse.ArgumentParser(description='Synthesize a stub LUCiD dataset for viewer testing.')
    ap.add_argument('--out', '-o', default='./test_data', help='Output directory')
    ap.add_argument('--geom', '-g', choices=('cylinder', 'box', 'sphere'), default='cylinder')
    ap.add_argument('--events', '-e', type=int, default=5)
    ap.add_argument('--sensors', '-n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    write_dataset(args.out, args.geom, args.events, args.sensors, args.seed)


if __name__ == '__main__':
    main()
