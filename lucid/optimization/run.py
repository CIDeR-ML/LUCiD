#!/usr/bin/env python3
"""``lucid-optimize`` — config-driven single-track reconstruction on PhotonSim ROOT events.

The validated recon path: the retired 5-stage Adam ``pipeline.py`` is
replaced by the Fisher-Gauss-Newton two-start fit (``lucid.fitting.fit_track_multistart``):
energy scan → charge-grid vertex ‖ time-multilateration vertex (each + a cone direction) →
GN fit from both seeds, keep the lower-loss basin (1% margin). Reconstructs to ~12 cm / ~1.0°
on 1 GeV muons. For the ≤20-line library form see
``examples/hello_reconstruct.py``.

Usage:  lucid-optimize <config.json>     (or  python -m lucid.optimization.run <config.json>)

Config (keys + defaults):
  {"geom": "...geom_config.json",            # REQUIRED — detector geometry JSON
   "physics_config": "...physics_config.json",
   "data": "/path/event.root | /path/dir",   # REQUIRED — a .root file (or dir → first file)
   "events": [0,1,2] | {"start":0,"count":10},
   "k": 8, "nphot": 250000, "nbuf": 400000, "tts": 2.5,
   "grid": {"n_cap":80,"n_angular":120,"n_height":80},
   "out": "recon_out"}
"""
import sys
import os
import glob
import json
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim, pad_photon_data
from lucid.fitting import (ReconModel, fit_track_multistart, track_from_vec9, vec9_from_track,
                           vec9_dir, seed_vertex_time)
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds
from lucid.optimization.utils.functions import (
    hierarchical_direction_search_cone, energy_scan_optimization)


def load_config(path):
    """Load a JSON config file (kept as a small public helper — notebooks import it)."""
    with open(path) as f:
        return json.load(f)


def _events(cfg):
    ev = cfg.get('events', {'start': 0, 'count': 1})
    if isinstance(ev, dict):
        return list(range(ev['start'], ev['start'] + ev['count']))
    return list(ev)


def main():
    ap = argparse.ArgumentParser(description='Single-track reconstruction (Fisher-GN two-start).')
    ap.add_argument('config_file')
    cfg = load_config(ap.parse_args().config_file)

    GEOM = cfg['geom']; PHYS = cfg.get('physics_config')
    data = cfg['data']; root = sorted(glob.glob(os.path.join(data, '*.root')))[0] if os.path.isdir(data) else data
    K = cfg.get('k', 8); NPH = cfg.get('nphot', 250_000); NBUF = cfg.get('nbuf', 400_000)
    TTS = cfg.get('tts', 2.5); GRID = cfg.get('grid', dict(n_cap=80, n_angular=120, n_height=80))
    OUT = cfg.get('out', 'recon_out'); os.makedirs(OUT, exist_ok=True)
    events = _events(cfg)
    print(f'lucid-optimize: {len(events)} event(s) from {root} | K={K} nphot={NPH} tts={TTS}', flush=True)

    det = generate_detector(GEOM); ND = len(det.all_points); POS = np.asarray(det.all_points)
    bounds = get_detector_bounds(det)
    dp_data = load_detector_params(PHYS, num_sensors=ND)
    dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(float(TTS))))
    data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                     physics_config=PHYS, default_detector_params=dp_data, particle='muon',
                                     wavelength_mode=True, apply_smearing=False, **GRID)
    pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon', physics_config=PHYS,
                                 default_detector_params=True, particle='muon', wavelength_mode=True,
                                 pos_grad_threshold=K, n_grad_iters=K, **GRID)
    model = ReconModel(pred, ND, sigma=float(TTS), delta=1.0)
    dummy = track_from_vec9(jnp.array([1050., 0, 0, 0, 0., 1., 0., 1., 0.]))   # is_data ignores the track

    for ev in events:
        t0 = time.time()
        raw = read_photon_data_from_photonsim(root, ev)
        pd, n = pad_photon_data(raw, NBUF)
        c, t = jax.lax.stop_gradient(data_sim(dummy, jax.random.PRNGKey(7000 + ev), pd))
        oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
        ocf, otf, POSf = jnp.asarray(oc), jnp.asarray(ot), jnp.asarray(POS)
        e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1 / jnp.sqrt(3)), jnp.pi / 4, 0.,
                                      POSf, otf, ocf, (ocf, otf), 1000., 700., 12, 0)['best_energy']

        def make_seed(vtx, t0g):
            c2 = hierarchical_direction_search_cone(pred, jnp.asarray(vtx), t0g, POSf, otf, ocf,
                                                    (ocf, otf), e0, 3, 8, 90., 0.5, 0)
            dg = np.array([np.sin(c2['best_theta']) * np.cos(c2['best_phi']),
                           np.sin(c2['best_theta']) * np.sin(c2['best_phi']), np.cos(c2['best_theta'])])
            return vec9_from_track(e0, np.asarray(vtx), dg, t0=t0g)
        p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.zeros(3), 0.0, 0.0,
                                               bounds, n_div=5, t0_n_div=5, levels=6, verbosity=0)
        seedA = make_seed(np.asarray(p1['best_position']), float(p1['best_t0']))
        seedB = make_seed(*seed_vertex_time(POS, oc, ot))
        res, MS = fit_track_multistart(model, oc, ot, [seedA, seedB], nkeys=4, niters=250)
        np.savez(os.path.join(OUT, f'ev{ev:04d}.npz'), fit=res, direction=vec9_dir(res),
                 which=MS['which'], losses=np.array(MS['losses']),
                 n_hit=int((oc > 0).sum()), q_tot=float(oc.sum()), event=ev)
        print(f'  ev{ev:04d}: E={res[0]:.0f}MeV vtx=({res[1]:.2f},{res[2]:.2f},{res[3]:.2f})m '
              f'seed={"AB"[MS["which"]]} [{time.time()-t0:.0f}s]', flush=True)
    print(f'done -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
