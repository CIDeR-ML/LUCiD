#!/usr/bin/env python3
"""LUCiD forward simulation for the controlled SK-IV laser comparisons."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from functools import partial
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LUCID = Path(os.environ.get('LUCID_CHECKOUT', HERE.parents[1])).resolve()
sys.path.insert(0, str(LUCID))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lucid_ballistic_mpl')

if os.environ.get('BALLISTIC_DEBUG'):
    import faulthandler
    faulthandler.dump_traceback_later(45, repeat=True)

import jax
import jax.numpy as jnp
import numpy as np
import uproot
from lucid.detector_params import DetectorParams
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import gaussian_laser_source
from lucid.wavelength.medium import make_medium

POSITION = np.array([-0.707, -7.777, 18.027])
DIRECTION = np.array([0.01123, 0.02418, -0.9768])
DIRECTION /= np.linalg.norm(DIRECTION)
DEFAULT_QE_TABLE = os.environ.get('SK_QE_TABLE')


def sk_qe_corrections(pmt_ids, qe_table):
    """Return SK's relative PMT efficiencies with detector-wide mean one."""
    table = np.loadtxt(qe_table)
    ids = table[:, 0].astype(int)
    if len(ids) != len(pmt_ids) or not np.array_equal(np.sort(ids), np.arange(1, len(ids) + 1)):
        raise ValueError('Unexpected SK QE-table cable IDs')
    by_cable = np.empty(len(ids), dtype=float)
    by_cable[ids - 1] = table[:, 1]
    corrections = by_cable[np.asarray(pmt_ids, dtype=int) - 1]
    return corrections / by_cable.mean()


# Uproot 5's default asynchronous local-file backend hangs on this host.
# Scope the synchronous local reader override to this standalone runner.
@patch('uproot.open', partial(uproot.open, handler=uproot.source.file.MemmapSource))
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=HERE / 'lucid.npz')
    ap.add_argument('--geometry', type=Path, default=LUCID / 'config/SK_geom_config.json',
                    help='SK geometry config (defaults to the selected LUCiD checkout)')
    ap.add_argument('--qe-table', type=Path, default=DEFAULT_QE_TABLE,
                    help='SK qetable3_0.dat (or set SK_QE_TABLE)')
    ap.add_argument('--rays', type=int, default=25000)
    ap.add_argument('--batches', type=int, default=8)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--uniform-qe', action='store_true',
                    help='Use the old uniform-QE baseline instead of SK per-PMT factors')
    ap.add_argument('--pmt-siren', type=Path,
                    help='Portable .npz (or legacy .npy + .json) PMT response SIREN')
    ap.add_argument('--pmt-detection-response', type=Path,
                    help='Tabulated local-incidence PMT response .npz')
    ap.add_argument('--water-absorption', action='store_true',
                    help='Enable LUCiD native water absorption at the 405 nm laser wavelength')
    ap.add_argument('--water-rayleigh', action='store_true',
                    help='Enable LUCiD native symmetric/Rayleigh scattering at 405 nm')
    ap.add_argument('--water-mie', action='store_true',
                    help='Enable LUCiD native asymmetric/Mie scattering at 405 nm')
    ap.add_argument('--reflection-model', choices=('off', 'angular'), default='off',
                    help='Surface reflection model (angular: SK-oriented blacksheet + PMT Fresnel model)')
    ap.add_argument('--wall-r0', type=float, default=0.05,
                    help='Normal-incidence blacksheet reflectance for --reflection-model angular')
    ap.add_argument('--wall-reflection-power', type=float, default=1.0,
                    help='Schlick angular exponent for blacksheet reflection')
    ap.add_argument('--wall-specular-fraction', type=float, default=0.55)
    ap.add_argument('--cathode-n-real', type=float, default=2.8)
    ap.add_argument('--cathode-n-imag', type=float, default=1.5)
    ap.add_argument('--sensor-specular-fraction', type=float, default=0.90)
    ap.add_argument('--mie-phase-model', choices=('hg', 'sk4'), default='hg',
                    help='Mie angular phase function (use sk4 for SKDetSim comparison)')
    ap.add_argument('--pmt-timing', choices=('none', 'sk4'), default='none',
                    help='Accepted-PE PMT timing response (default: none)')
    ap.add_argument('--sensor-acceptance', choices=('sphere', 'cosine'), default='sphere',
                    help='Sensor interception model; cosine is a flat-disc diagnostic')
    ap.add_argument('--sensor-acceptance-power', type=float, default=1.0,
                    help='Power of incidence cosine in the diagnostic model')
    ap.add_argument('--sensor-shape', choices=('sphere', 'sk20inch'), default='sphere',
                    help='PMT interception geometry (default: legacy sphere)')
    ap.add_argument('--sensor-candidate-selection', choices=('all', 'wall'), default='all',
                    help='PMT routing rule (wall requires --sensor-shape sk20inch)')
    ap.add_argument('--sampled-transport', action='store_true',
                    help='Sample transport decisions instead of using the average-response estimator')
    ap.add_argument('--propagation-speed', type=float,
                    help='Fixed photon group speed in m/ns (default: configured material speed)')
    ap.add_argument('--incidence-diagnostics', action='store_true',
                    help='Save PMT-axis incidence cosine and propagation step per deposition')
    ap.add_argument('--transport-diagnostics', action='store_true',
                    help='Save the first water-scatter state for transport comparisons')
    args = ap.parse_args()
    print('Imports complete; loading detector', flush=True)
    if args.output.exists():
        ap.error('Output exists; choose a new --output')
    if min(args.rays, args.batches, args.steps) < 1:
        ap.error('Counts must be positive')
    if args.pmt_detection_response is not None and args.sensor_shape != 'sk20inch':
        ap.error('--pmt-detection-response requires --sensor-shape sk20inch')
    if not args.uniform_qe and args.qe_table is None:
        ap.error('--qe-table or SK_QE_TABLE is required unless --uniform-qe is used')
    if args.sensor_candidate_selection == 'wall' and args.sensor_shape != 'sk20inch':
        ap.error('--sensor-candidate-selection wall requires --sensor-shape sk20inch')
    if not (0 <= args.wall_r0 <= 1 and 0 <= args.wall_specular_fraction <= 1
            and 0 <= args.sensor_specular_fraction <= 1):
        ap.error('reflection probabilities/fractions must lie in [0, 1]')
    geom = args.geometry.resolve()
    detector = generate_detector(str(geom))
    n = len(detector.all_points)
    qe_corrections = (np.ones(n) if args.uniform_qe
                      else sk_qe_corrections(detector.pmt_id, args.qe_table))
    # Infinite lengths generate intermediate inf/inf in the two-channel
    # sampler. Finite 1e20 m makes water losses <1e-18 per crossing, with
    # finite branch probabilities. K=1 retains precisely the direct deposit.
    absorption_length = 1e20
    scatter_length = 1e20
    mie_scatter_length = 1e20
    mie_asymmetry = 0.0
    if args.water_absorption or args.water_rayleigh or args.water_mie:
        water_405 = make_medium('water', wavelength_grid=jnp.asarray([405.0]))
    if args.water_absorption:
        absorption_length = float(1.0 / water_405.absorption_coeff[0])
    if args.water_rayleigh:
        scatter_length = float(1.0 / water_405.scatter_coeff[0])
    if args.water_mie:
        mie_scatter_length = float(1.0 / water_405.mie_scatter_coeff[0])
        mie_asymmetry = float(water_405.mie_asymmetry)
    dp = DetectorParams.from_flat(num_sensors=n, scatter_length=scatter_length,
        mie_scatter_length=mie_scatter_length, g=mie_asymmetry,
        absorption_length=absorption_length,
        wall_reflection_rate=0., sensor_reflection_rate=0.,
        wall_R0=args.wall_r0, wall_p=args.wall_reflection_power,
        wall_fspec=args.wall_specular_fraction,
        cathode_nr=args.cathode_n_real, cathode_nk=args.cathode_n_imag,
        sensor_fspec=args.sensor_specular_fraction, qe=0.2,
        qe_corrections=qe_corrections)
    source = gaussian_laser_source(position=POSITION, direction=DIRECTION,
        intensity=8090., beam_width_deg=3300., wavelength=405.)
    print('Building propagation grid and simulator', flush=True)
    simulator_options = dict(
        temperature=None, K=args.steps, is_calibration=True,
        hit_mode='per_photon', wavelength_mode=False,
        use_expected_value=not args.sampled_transport,
        reflection_model=('scalar' if args.reflection_model == 'off' else 'angular'),
        reflection_wavelength=405.0, pmt_timing_model=args.pmt_timing,
        propagation_speed_m_per_ns=args.propagation_speed,
        return_incidence_diagnostics=args.incidence_diagnostics,
        return_transport_diagnostics=args.transport_diagnostics,
        sensor_shape=args.sensor_shape,
        sensor_candidate_selection=args.sensor_candidate_selection,
        sensor_acceptance_model=args.sensor_acceptance,
        sensor_acceptance_power=args.sensor_acceptance_power,
        max_candidates_per_ray=8, n_cap=140, n_angular=240, n_height=140)
    if 'mie_phase_model' in inspect.signature(setup_event_simulator).parameters:
        simulator_options['mie_phase_model'] = args.mie_phase_model
    elif args.mie_phase_model != 'hg':
        ap.error('selected LUCiD checkout predates selectable Mie phase models')
    if args.pmt_siren is not None:
        simulator_options['pmt_response_model'] = args.pmt_siren
    if args.pmt_detection_response is not None:
        simulator_options['pmt_detection_model'] = args.pmt_detection_response
    simulate = setup_event_simulator(str(geom), args.rays, **simulator_options)
    print('Simulator ready; compiling first batch', flush=True)
    charges, times, weights, sensors, batch_indices = [], [], [], [], []
    batch_runtime_seconds = []
    incidence_cosines, local_incidence_cosines = [], []
    propagation_steps, photon_indices = [], []
    transport_photon_indices = []
    first_scatter_positions, first_scatter_in_directions = [], []
    first_scatter_out_directions = []
    first_scatter_distances, first_surface_distances = [], []
    for i in range(args.batches):
        start = perf_counter()
        out = simulate(source, dp, jax.random.PRNGKey(20260908 + i))
        logw, t, idx, q = [np.asarray(x) for x in out[:4]]
        if not np.all(np.isfinite(q)) or q.sum() <= 0:
            raise RuntimeError('Nonfinite or empty direct-light prediction')
        elapsed = perf_counter() - start
        batch_runtime_seconds.append(elapsed)
        valid = (logw > -1e9) & np.isfinite(t) & (t > 0)
        charges.append(q)
        times.append(t[valid])
        weights.append(np.exp(logw[valid]) / args.batches)
        sensors.append(idx[valid])
        batch_indices.append(np.full(
            np.count_nonzero(valid), i, dtype=np.int16))
        if args.incidence_diagnostics:
            incidence_cosines.append(np.asarray(out[4])[valid])
            propagation_steps.append(np.asarray(out[5])[valid])
            local_incidence_cosines.append(np.asarray(out[6])[valid])
            photon_indices.append(
                (i * args.rays + np.arange(logw.size) % args.rays)[valid])
        if args.transport_diagnostics:
            offset = 7 if args.incidence_diagnostics else 4
            (pre_pos, pre_dir, post_pos, post_dir,
             continuation, surface_distance) = [
                np.asarray(x) for x in out[offset:offset + 6]]
            # With reflection disabled, a positive first-step continuation is
            # exactly a scatter before the detector surface.
            scattered = continuation[0] > 0.0
            transport_photon_indices.append(
                i * args.rays + np.flatnonzero(scattered))
            first_scatter_positions.append(post_pos[0, scattered])
            first_scatter_in_directions.append(pre_dir[0, scattered])
            first_scatter_out_directions.append(post_dir[0, scattered])
            first_scatter_distances.append(np.linalg.norm(
                post_pos[0, scattered] - pre_pos[0, scattered], axis=1))
            first_surface_distances.append(surface_distance[0, scattered])
        print('Batch {}/{}: {:.1f}s, {:.3f} PE/pulse'.format(
            i+1, args.batches, elapsed, q.sum()), flush=True)
    charges = np.asarray(charges, dtype=float)
    xyz = np.asarray(detector.all_points)
    delta = xyz - POSITION
    angles = np.degrees(np.arccos(np.clip(delta @ DIRECTION / np.linalg.norm(delta, axis=1), -1, 1)))
    deps = [geom, *sorted((LUCID / 'lucid').rglob('*.py'))]
    connection_table = geom.parent / 'ConnectionTable_SK5.root'
    if connection_table.exists():
        deps.append(connection_table)
    provenance = {}
    for path in deps:
        try:
            label = str(path.relative_to(LUCID))
        except ValueError:
            label = str(path)
        provenance[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not args.uniform_qe:
        provenance[str(args.qe_table)] = hashlib.sha256(
            args.qe_table.read_bytes()).hexdigest()
    if args.pmt_siren is not None:
        model_path = args.pmt_siren.resolve()
        provenance[str(model_path)] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        sidecar = model_path.with_suffix('.json')
        if model_path.suffix == '.npy' and sidecar.exists():
            provenance[str(sidecar)] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if args.pmt_detection_response is not None:
        response_path = args.pmt_detection_response.resolve()
        provenance[str(response_path)] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_arrays = {}
    if args.incidence_diagnostics:
        diagnostic_arrays = {
            'incidence_cosine': np.concatenate(incidence_cosines).astype(np.float32),
            'local_incidence_cosine': np.concatenate(
                local_incidence_cosines).astype(np.float32),
            'propagation_step': np.concatenate(propagation_steps).astype(np.int8),
            'photon_index': np.concatenate(photon_indices).astype(np.int64),
        }
    if args.transport_diagnostics:
        diagnostic_arrays.update({
            'transport_photon_index': np.concatenate(
                transport_photon_indices).astype(np.int64),
            'first_scatter_position': np.concatenate(
                first_scatter_positions).astype(np.float32),
            'first_scatter_in_direction': np.concatenate(
                first_scatter_in_directions).astype(np.float32),
            'first_scatter_out_direction': np.concatenate(
                first_scatter_out_directions).astype(np.float32),
            'first_scatter_distance': np.concatenate(
                first_scatter_distances).astype(np.float32),
            'first_surface_distance': np.concatenate(
                first_surface_distances).astype(np.float32),
        })
    np.savez_compressed(args.output, charge=charges.mean(axis=0), batch_charge=charges,
        charge_sem=charges.std(axis=0, ddof=1)/np.sqrt(args.batches) if args.batches > 1 else np.full(n, np.nan),
        xyz=xyz, cable_id=detector.pmt_id, angle_deg=angles,
        qe_corrections=qe_corrections,
        time=np.concatenate(times), weight=np.concatenate(weights), sensor=np.concatenate(sensors),
        batch_index=np.concatenate(batch_indices),
        position=POSITION, direction=DIRECTION, photons_per_pulse=8090,
        rays_per_batch=args.rays, batches=args.batches, steps=args.steps,
        batch_runtime_seconds=np.asarray(batch_runtime_seconds),
        speed_m_per_ns=make_medium('water').speed_of_light,
        provenance=json.dumps(provenance), settings=json.dumps({
            'scatter_length_m': scatter_length,
            'mie_scatter_length_m': mie_scatter_length,
            'water_rayleigh_model': ('LUCiD water reference at 405 nm'
                                      if args.water_rayleigh else 'off'),
            'water_mie_model': ('LUCiD water reference at 405 nm'
                                if args.water_mie else 'off'),
            'mie_phase_model': args.mie_phase_model,
            'mie_asymmetry_g': (
                mie_asymmetry if args.mie_phase_model == 'hg' else None),
            'mie_phase_note': (
                'g is unused; SK4 SGMIES has fixed p(mu)=2*mu on 0<=mu<=1'
                if args.mie_phase_model == 'sk4'
                else 'Henyey-Greenstein phase controlled by mie_asymmetry_g'),
            'absorption_length_m': absorption_length,
            'water_absorption_model': ('LUCiD water reference at 405 nm'
                                       if args.water_absorption else 'off'),
            'reflection_model': args.reflection_model,
            'wall_reflection_rate': 0, 'sensor_reflection_rate': 0,
            'wall_R0': args.wall_r0,
            'wall_reflection_power': args.wall_reflection_power,
            'wall_specular_fraction': args.wall_specular_fraction,
            'cathode_n_real': args.cathode_n_real,
            'cathode_n_imag': args.cathode_n_imag,
            'sensor_specular_fraction': args.sensor_specular_fraction,
            'qe': 0.2,
            'qe_model': 'uniform' if args.uniform_qe else 'SK qetable3_0, normalized to detector mean 1',
            'pmt_response_model': None if args.pmt_siren is None else str(args.pmt_siren.resolve()),
            'pmt_detection_model': (None if args.pmt_detection_response is None
                                    else str(args.pmt_detection_response.resolve())),
            'pmt_timing_model': args.pmt_timing,
            'transport_estimator': ('sampled' if args.sampled_transport else 'average response'),
            'propagation_speed_m_per_ns': (make_medium('water').speed_of_light
                                           if args.propagation_speed is None
                                           else args.propagation_speed),
            'sensor_shape': args.sensor_shape,
            'sensor_candidate_selection': args.sensor_candidate_selection,
            'sensor_acceptance_model': args.sensor_acceptance,
            'sensor_acceptance_power': args.sensor_acceptance_power,
            'beam_width_deg': 3300, 'wavelength_nm': 405,
            'incidence_diagnostics': args.incidence_diagnostics,
            'transport_diagnostics': args.transport_diagnostics,
            'first_seed': 20260908, 'jax_version': jax.__version__}),
        **diagnostic_arrays)
    print(args.output)


if __name__ == '__main__':
    main()
