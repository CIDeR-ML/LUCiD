#!/usr/bin/env python3
"""Photon-shotgun production runner.

Generates N cases × n_photons photons through the LUCiD simulator, streaming
the output to HDF5 chunk-by-chunk so large runs stay within GPU/host memory.

Example
-------
10,000 isotropic Cherenkov shots of 100k photons each from random positions in
SK, waveform mode::

    python -m lucid.production.photon_shotgun.run \\
        --detector config/SK_geom_config.json \\
        --physics-config config/SK_physics_config.json \\
        --n-cases 10000 --n-photons 100000 \\
        --position-mode uniform --direction-mode isotropic \\
        --wavelength cherenkov \\
        --output-mode waveform \\
        --chunk 20 \\
        -o runs/shotgun_SK_10k.h5
"""
import argparse
import os
import time

import jax
import numpy as np

from lucid.simulation.shotgun import setup_shotgun_simulator
from lucid.sources.shotgun_source import shotgun_source, stack_shotgun_sources
from lucid.production.photon_shotgun.utils import (
    read_detector_bounds,
    sample_positions_uniform, sample_directions_isotropic,
)
from lucid.production.photon_shotgun.io import (
    StreamingWaveformWriter, StreamingPerPhotonWriter,
)


def _parse_wavelength(s: str):
    if s.lower() == 'cherenkov':
        return 'cherenkov'
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--wavelength must be a float (nm) or 'cherenkov'; got {s!r}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--detector', required=True, help='Detector geometry JSON')
    p.add_argument('--physics-config', required=True, help='Physics config JSON')
    p.add_argument('--detector-type', default='Cylinder',
                   choices=['Cylinder', 'Sphere', 'Box'])
    p.add_argument('--n-cases', type=int, required=True)
    p.add_argument('--n-photons', type=int, required=True,
                   help='Photons per case')
    p.add_argument('--position-mode', default='uniform',
                   choices=['uniform', 'center'],
                   help='uniform = random inside detector volume; '
                        'center = all cases at origin (requires --direction)')
    p.add_argument('--position-fraction', type=float, default=0.9,
                   help='Shrink factor for uniform-position sampling (default 0.9)')
    p.add_argument('--direction-mode', default='isotropic',
                   choices=['isotropic', 'fixed'])
    p.add_argument('--direction', nargs=3, metavar=('DX', 'DY', 'DZ'),
                   type=float, default=None,
                   help='Fixed direction (required when --direction-mode fixed)')
    p.add_argument('--origin', nargs=3, metavar=('X', 'Y', 'Z'),
                   type=float, default=None,
                   help='Fixed origin (required when --position-mode center)')
    p.add_argument('--wavelength', type=_parse_wavelength, default='cherenkov')
    p.add_argument('--intensity', type=float, default=None,
                   help="Per-photon weight. Default: 1.0 in Bernoulli mode; "
                        "1/n_photons in --expected-value mode so the output "
                        "is an expected-detection-rate density (per emitted "
                        "photon). Pass explicitly to override.")
    p.add_argument('--wavelength-sampling',
                   choices=['cherenkov', 'cherenkov_qe'],
                   default='cherenkov',
                   help="How LUCiD samples λ when source.wavelength is None. "
                        "'cherenkov' (default): λ~1/λ², per-photon QE weight. "
                        "'cherenkov_qe': λ~QE(λ)/λ², scalar <QE>_C weight — "
                        "lower variance, but density-estimate semantics.")
    p.add_argument('--expected-value', action='store_true',
                   help="Use expected-value propagator + continuous QE weight "
                        "(hit_mode='waveform_expected'). Same output shape as "
                        "Bernoulli waveform but with substantially lower MC "
                        "noise — the output is the expected waveform, not a "
                        "per-shot realization. Requires --output-mode waveform.")
    p.add_argument('--output-mode', choices=['waveform', 'per_photon'],
                   default='waveform')
    p.add_argument('--K', type=int, default=12)
    p.add_argument('--window-ns', type=float, default=500.0)
    p.add_argument('--bin-ns', type=float, default=1.0)
    p.add_argument('--tts-sigma-ns', type=float, default=1.0)
    p.add_argument('--no-smear-time', action='store_true')
    p.add_argument('--no-smear-charge', action='store_true')
    p.add_argument('--chunk', type=int, default=20,
                   help='Cases per vmap batch (balances GPU memory vs launch overhead)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--save-source', action='store_true',
                   help='Persist per-case photon origin/direction arrays (large!)')
    p.add_argument('--charge-threshold', type=float, default=0.0,
                   help="Sparsification threshold on per-bin charge. Default 0 "
                        "(keep everything) works for Bernoulli. For expected-"
                        "value mode the waveform is non-sparse — set e.g. "
                        "0.01 to drop sub-percent-photon bins and keep storage "
                        "tractable.")
    return p


def _build_positions_directions(args, n_cases: int):
    key = jax.random.PRNGKey(args.seed)
    pos_key, dir_key = jax.random.split(key)

    if args.position_mode == 'uniform':
        bounds = read_detector_bounds(args.detector)
        positions = sample_positions_uniform(
            n_cases, bounds, fraction=args.position_fraction, key=pos_key)
    elif args.position_mode == 'center':
        if args.origin is None:
            raise SystemExit('--position-mode center requires --origin')
        positions = np.tile(np.asarray(args.origin, dtype=np.float32), (n_cases, 1))
    else:
        raise ValueError(args.position_mode)

    if args.direction_mode == 'isotropic':
        directions = sample_directions_isotropic(n_cases, key=dir_key)
    elif args.direction_mode == 'fixed':
        if args.direction is None:
            raise SystemExit('--direction-mode fixed requires --direction')
        directions = np.tile(np.asarray(args.direction, dtype=np.float32), (n_cases, 1))
    else:
        raise ValueError(args.direction_mode)
    return positions, directions


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Per-photon intensity. Default: 1.0 in Bernoulli (integer counts); 1/n in
    # expected mode so the saved waveform is a density (expected detection
    # rate per emitted photon). Explicit --intensity always wins.
    if args.intensity is None:
        args.intensity = (1.0 / args.n_photons) if args.expected_value else 1.0
    print(f"[shotgun] per-photon intensity = {args.intensity:.3e}")

    print(f"[shotgun] building simulator — detector={args.detector_type}, "
          f"n_photons={args.n_photons}, mode={args.output_mode}, K={args.K}")
    t0 = time.time()
    sim = setup_shotgun_simulator(
        args.detector,
        physics_config=args.physics_config,
        n_photons=args.n_photons,
        output_mode=args.output_mode,
        K=args.K,
        detector_type=args.detector_type,
        window_ns=args.window_ns,
        bin_width_ns=args.bin_ns,
        tts_sigma_ns=args.tts_sigma_ns,
        smear_time=not args.no_smear_time,
        smear_charge=not args.no_smear_charge,
        wavelength_sampling=args.wavelength_sampling,
        use_expected_value=args.expected_value,
    )
    print(f"[shotgun] setup: {time.time()-t0:.1f}s  num_sensors={sim.num_sensors}")

    positions, directions = _build_positions_directions(args, args.n_cases)
    print(f"[shotgun] sampled {args.n_cases} positions "
          f"(mode={args.position_mode}, dir={args.direction_mode})")
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.n_cases)

    def _chunk_source(lo, hi):
        """Build sources for [lo, hi) on the fly — avoids holding all n_cases
        photon arrays in memory."""
        subs = [
            shotgun_source(positions[j], directions[j],
                           n_photons=args.n_photons,
                           wavelength=args.wavelength,
                           intensity=args.intensity)
            for j in range(lo, hi)
        ]
        return stack_shotgun_sources(subs)

    total_phot = args.n_cases * args.n_photons
    print(f"[shotgun] running {args.n_cases} cases × {args.n_photons:,} "
          f"photons = {total_phot:,} total | chunk={args.chunk}")

    # Warm JIT so timing reflects steady-state.
    _warm = shotgun_source(positions[0], directions[0],
                           n_photons=args.n_photons,
                           wavelength=args.wavelength,
                           intensity=args.intensity)
    _ = sim(_warm, keys[0])[0].block_until_ready()
    del _warm

    n_time_bins = int(round(sim.waveform_config['window_ns']
                             / sim.waveform_config['bin_width_ns']))
    t0 = time.time()

    if args.output_mode == 'waveform':
        writer = StreamingWaveformWriter(
            args.output,
            num_sensors=sim.num_sensors, n_time_bins=n_time_bins,
            waveform_config=sim.waveform_config,
            detector_config=args.detector, physics_config=args.physics_config,
            n_photons=args.n_photons, K=args.K, save_source=args.save_source,
            threshold=args.charge_threshold)
        total_det = total_drop = 0
    else:
        writer = StreamingPerPhotonWriter(
            args.output,
            n_photons=args.n_photons, tts_sigma_ns=args.tts_sigma_ns,
            detector_config=args.detector, physics_config=args.physics_config,
            K=args.K, save_source=args.save_source)
        total_det = 0

    try:
        for i in range(0, args.n_cases, args.chunk):
            hi = min(i + args.chunk, args.n_cases)
            batched = _chunk_source(i, hi)
            sub_keys = keys[i:hi]
            out = sim.batch(batched, sub_keys)
            out[0].block_until_ready()
            src_chunk = batched if args.save_source else None
            if args.output_mode == 'waveform':
                wf, nd, ndet = out
                writer.append(np.asarray(wf), np.asarray(nd), np.asarray(ndet),
                              source_chunk=src_chunk)
                total_det += int(np.asarray(ndet).sum())
                total_drop += int(np.asarray(nd).sum())
            else:
                det, sid, ht = out
                writer.append(np.asarray(det), np.asarray(sid), np.asarray(ht),
                              source_chunk=src_chunk)
                total_det += int(np.asarray(det).sum())
            dt = time.time() - t0
            print(f"[shotgun] {hi}/{args.n_cases} cases  "
                  f"{dt:.1f}s  ({hi / dt:.1f} cases/s)", flush=True)
    finally:
        writer.close()

    if args.output_mode == 'waveform':
        print(f"[shotgun] total detected={total_det:,}  dropped={total_drop}")
    else:
        print(f"[shotgun] total detected={total_det:,}")

    dt = time.time() - t0
    print(f"[shotgun] done in {dt:.1f}s  ({total_phot / dt / 1e6:.2f} M phot/s)")
    print(f"[shotgun] wrote {args.output} ({os.path.getsize(args.output)/1e6:.2f} MB)")


if __name__ == '__main__':
    main()
