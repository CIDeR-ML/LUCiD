"""generate_example_events — write a few lightweight per-event HDF5 files (Q/T + truth).

Produces the simple single-event files (per-PMT charge + first-arrival time + truth track) that
tutorials, tests, and quick studies want — WITHOUT the heavyweight 4-file v3 production dataset
(that's `lucid-run-job`). Reads a PhotonSim ROOT file and runs the data-mode forward.

Run:  python scripts/generate_example_events.py --root data/water/muon/1000MeV_100events.root \
          --n-events 5 --out example_events
"""
import argparse, os
import jax, jax.numpy as jnp, numpy as np, h5py
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import track_from_vec9, vec9_from_track
from lucid.sources.event_io import read_photon_data_from_photonsim, pad_photon_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='PhotonSim ROOT file')
    ap.add_argument('--geom', default='config/SK_like_geom_config.json')
    ap.add_argument('--physics', default='config/SK_like_physics_config.json')
    ap.add_argument('--n-events', type=int, default=5)
    ap.add_argument('--out', default='example_events', help='output directory')
    ap.add_argument('--nbuf', type=int, default=400_000, help='photon buffer size')
    ap.add_argument('--tts', type=float, default=2.5, help='per-photon transit-time spread (ns)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    det = generate_detector(args.geom); ND = len(det.all_points)
    dp = load_detector_params(args.physics, num_sensors=ND)
    dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(args.tts)))
    sim = setup_event_simulator(args.geom, args.nbuf, temperature=None, K=8, is_data=True,
                                hit_mode='realistic', apply_smearing=False, physics_config=args.physics,
                                default_detector_params=dp, particle='muon', wavelength_mode=True,
                                n_cap=100, n_angular=150, n_height=100)
    dummy = track_from_vec9(jnp.asarray(vec9_from_track(1000., [0, 0, 0], [0, 0, 1], t0=0.)))

    print(f'{ND} sensors | writing {args.n_events} events to {args.out}/')
    for ev in range(args.n_events):
        raw = read_photon_data_from_photonsim(args.root, ev)
        pd, _ = pad_photon_data(raw, args.nbuf)
        c, t = jax.lax.stop_gradient(sim(dummy, jax.random.PRNGKey(1000 + ev), pd))
        charge = np.asarray(c); time_ = np.where(charge > 0, np.asarray(t), 0.)
        origins = np.asarray(raw['photon_origins']).astype(float)
        vtx = origins.mean(0) / 100.0                                   # ROOT is mm -> m (approx event centroid)
        path = os.path.join(args.out, f'event_{ev}.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('charge', data=charge)
            f.create_dataset('time', data=time_)
            f.create_dataset('sensor_positions', data=np.asarray(det.all_points))
            f.attrs['truth_vertex_m'] = vtx
            f.attrs['n_hit'] = int((charge > 0).sum())
        print(f'  event_{ev}.h5: {int((charge>0).sum())} hit PMTs, {charge.sum():.0f} pe')
    print('done')


if __name__ == '__main__':
    main()
