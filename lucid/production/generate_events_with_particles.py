#!/usr/bin/env python3
"""Generate v3 four-file datasets from a PhotonSim particle-based ROOT file.

For each batch of ``--batch-size`` events this script writes four parallel
HDF5 files under the output dataset root:

    ``{output}/sensor/wc_sensor_NNNN.h5``
    ``{output}/hits/wc_hits_NNNN.h5``
    ``{output}/step/wc_step_NNNN.h5``
    ``{output}/labl/wc_labl_NNNN.h5``

See ``docs/LUCID_DATASET.md`` for the full v3 schema.

Usage:
    python generate_events_with_particles.py \\
        --root-file test.root \\
        --config SK_like_geom_config.json \\
        --output datasets/my_run/ \\
        --dataset-name my_run_2026_04_19

``--dataset-name`` is required and identifies the logical dataset. ``--run-id``
is an optional UUID (auto-generated if omitted). One dataset per output
directory — concurrent writes to the same directory are not supported.
"""

import argparse
import sys
import os

import numpy as np

from lucid.sources.event_generation import generate_events_from_photonsim_particles
from lucid.simulation import setup_event_simulator
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.utils import base_dir_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate v3 four-file datasets from a PhotonSim particle-based ROOT file.',
        epilog='Output goes to {output}/{sensor,hits,step,labl}/wc_*_NNNN.h5.',
    )
    parser.add_argument('--root-file', type=str, required=True,
                        help='Path to PhotonSim ROOT file with particle-based data.')
    parser.add_argument('--output', type=str, required=True,
                        help='Dataset root directory. Four subdirectories '
                             '(sensor/, hits/, step/, labl/) are created under it.')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Logical dataset identifier written to every config/ group.')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Unique batch identifier (auto-UUID4 if omitted). '
                             'One logical dataset per output directory; do not mix runs.')
    parser.add_argument('--config', type=str,
                        default=base_dir_path() + 'config/SK_like_geom_config.json',
                        help='Detector geometry JSON (default: SK_like_geom_config.json).')
    parser.add_argument('--physics-config', type=str,
                        default=base_dir_path() + 'config/SK_like_physics_config.json',
                        help='Physics config JSON (medium, QE curve, detector params).')
    parser.add_argument('--n-events', type=int, default=None,
                        help='Number of events to generate (default: all entries in ROOT file).')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Events per output batch (one batch = one set of four files).')
    parser.add_argument('--file-index-start', type=int, default=0,
                        help='Index of the first batch file (default 0).')
    parser.add_argument('--master-seed', type=int, default=None,
                        help='Random seed for reproducibility (random if omitted).')
    parser.add_argument('--apply-smearing', action='store_true',
                        help='Apply SK-like smearing to per-sensor PE/T.')
    parser.add_argument('--apply-rotation', action='store_true',
                        help='(Ignored — PhotonSim already randomizes directions.)')
    parser.add_argument('--apply-translation', action='store_true',
                        help='Random per-event translation of photons/tracks.')
    args = parser.parse_args()

    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file not found: {args.root_file}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Error: Geometry config not found: {args.config}")
        sys.exit(1)

    # Verify ROOT file uses the chunked OpticalPhotonsRaw layout
    # (per-photon scalars including PhotonWavelength live there).
    import uproot
    _root = uproot.open(args.root_file)
    if 'OpticalPhotonsRaw' not in _root:
        print("Error: ROOT file does not contain 'OpticalPhotonsRaw' tree.")
        print("Regenerate with the current PhotonSim build (per-photon scalars "
              "moved off OpticalPhotons into a chunked sister tree).")
        sys.exit(1)
    _root.close()

    # Derive sensor positions / geometry metadata from the detector config
    det_geom = DetectorGeometry.from_config(args.config)
    sensor_positions = np.asarray(det_geom.sensor_points, dtype=np.float32)
    detector_type = str(det_geom.detector_type)
    material = str(det_geom.medium.material)
    print(f"\nDetector geometry: {detector_type} / {material}, "
          f"{sensor_positions.shape[0]} sensors")

    # Setup the simulator with baked-in detector params (loaded from physics config).
    # This also normalizes scalar qe_corrections via the guard at
    # simulator.py:174-183 before the closure is built.
    print(f"\nSetting up event simulator with baked-in detector params")
    print(f"  Physics config: {args.physics_config}")
    simulate_event = setup_event_simulator(
        args.config,
        0,  # n_photons irrelevant in data mode — driven by ROOT
        K=12,
        is_data=True,
        temperature=0.0,
        apply_smearing=False,  # per-particle smearing off; PE-sum smearing applied later
        physics_config=args.physics_config,
        default_detector_params=True,
        hit_mode='per_segment',  # mandatory for data mode (seg/sensor_hits/ ground truth)
    )
    dp = simulate_event.default_detector_params
    print(f"  Wall reflection rate: {float(dp.reflection.wall_reflection_rate):.3f}")
    print(f"  Sensor reflection rate: {float(dp.reflection.sensor_reflection_rate):.3f}")
    print(f"  QE (scalar): {float(dp.response.qe):.3f}")

    print(f"\nGenerating v3 dataset:")
    print(f"  Source ROOT: {args.root_file}")
    print(f"  Output root: {args.output}")
    print(f"  Dataset name: {args.dataset_name}")
    if args.run_id:
        print(f"  Run id: {args.run_id}")
    if args.n_events:
        print(f"  Number of events: {args.n_events}")
    if args.master_seed is not None:
        print(f"  Master seed: {args.master_seed}")
    print(f"  Apply smearing: {args.apply_smearing}")
    print(f"  Apply translation: {args.apply_translation}")

    saved_files = generate_events_from_photonsim_particles(
        event_simulator=simulate_event,
        root_file_path=args.root_file,
        sensor_positions=sensor_positions,
        output_dir=args.output,
        n_events=args.n_events,
        batch_size=args.batch_size,
        master_seed=args.master_seed,
        apply_smearing=args.apply_smearing,
        apply_rotation=args.apply_rotation,
        apply_translation=args.apply_translation,
        dataset_name=args.dataset_name,
        run_id=args.run_id,
        file_index_start=args.file_index_start,
    )

    print(f"\nWrote {len(saved_files)} files under {args.output}/"
          f"{{sensor,hits,step,labl}}/")


if __name__ == '__main__':
    main()
