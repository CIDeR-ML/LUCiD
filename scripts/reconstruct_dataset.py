"""reconstruct_dataset — reconstruct a whole PhotonSim dataset and report resolution.

Portable version of the reconstruction campaign (scripts/campaign_recon): for each event in a
PhotonSim ROOT it randomizes the event into the fiducial volume (which gives an exact known
truth vertex/direction), runs the two-start data-driven seeder + Fisher-Gauss-Newton fit, and
aggregates vertex / direction / energy / t0 resolution. Writes one npz per event + a summary.
The campaign-scale companion to `lucid-optimize` (one config) and
`examples/seed_reconstruct.py` (one event).

Run:  python scripts/reconstruct_dataset.py --root data/water/muon/1000MeV_100events.root \
          --events 20 --out recon_out
"""
import argparse, os
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import (ReconModel, fit_track_multistart, seed_vertex_time,
                           vec9_from_track, vec9_dir, track_from_vec9)
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds
from lucid.optimization.utils.functions import (hierarchical_direction_search_cone,
                                                energy_scan_optimization)
from lucid.sources.event_io import read_photon_data_from_photonsim, pad_photon_data

GRID = dict(n_cap=80, n_angular=120, n_height=80)


def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def rand_tf(raw, ev, fidr, fidz):
    """Randomize the event into the fiducial volume; return (raw', truth vertex[m], truth dir).

    The muon gun fires from the ROOT origin (0,0,0) along +z; applying the same rigid transform
    to the photons and to that initial track gives the exact physical truth.
    """
    rng = np.random.default_rng(100003 + ev); beta = np.degrees(np.arccos(rng.uniform(-1, 1)))
    al = rng.uniform(0, 2 * np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
    rr = fidr * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
    sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-fidz, fidz)]) * 100.0
    raw = dict(raw); O = np.asarray(raw['photon_origins']).astype(float)
    D = np.asarray(raw['photon_directions']).astype(float); R = _rotax(axis, beta); c = O.mean(0)
    raw['photon_origins'] = (O - c) @ R.T + c + sh
    raw['photon_directions'] = D @ R.T
    vtx_true = ((np.zeros(3) - c) @ R.T + c + sh) / 100.0     # meters
    dir_true = np.array([0., 0., 1.]) @ R.T
    return raw, vtx_true, dir_true


def truth9(vtx, d, energy):
    pol = float(np.arccos(np.clip(d[2], -1, 1))); az = float(np.arctan2(d[1], d[0]))
    return np.array([float(energy), vtx[0], vtx[1], vtx[2],
                     np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.]), np.asarray(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='PhotonSim ROOT file')
    ap.add_argument('--geom', default='config/SK_like_geom_config.json')
    ap.add_argument('--physics', default='config/SK_like_physics_config.json')
    ap.add_argument('--events', type=int, default=20)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--out', default='recon_out')
    ap.add_argument('--energy', type=float, default=1000., help='fallback truth energy (MeV) if ROOT lacks it')
    ap.add_argument('--nbuf', type=int, default=400_000)
    ap.add_argument('--nph', type=int, default=250_000)
    ap.add_argument('--niters', type=int, default=130)
    ap.add_argument('--nkeys', type=int, default=4)
    ap.add_argument('--fidr', type=float, default=12.0, help='fiducial radius (m)')
    ap.add_argument('--fidz', type=float, default=12.0, help='fiducial half-height (m)')
    ap.add_argument('--cherenkov-band', type=float, nargs=2, default=[274.91, 673.83],
                    metavar=('LO', 'HI'), help='model Cherenkov emission band [LO HI] in nm '
                    '(GEANT4-consistent); use "--cherenkov-band 0 0" for an unbanded model')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    det = generate_detector(args.geom); ND = len(det.all_points)
    POS = np.asarray(det.all_points); POSf = jnp.asarray(POS); bounds = get_detector_bounds(det)
    dp_data = load_detector_params(args.physics, num_sensors=ND)
    dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(2.5)))
    data_sim = setup_event_simulator(args.geom, args.nbuf, temperature=None, K=8, is_data=True,
                                     hit_mode='realistic', physics_config=args.physics,
                                     default_detector_params=dp_data, particle='muon',
                                     wavelength_mode=True, charge_resolution=None, **GRID)
    # Band-consistent recon: sample the model over the PhotonSim Cherenkov emission band
    # [274.91, 673.83] nm so QE applies to the true band (GEANT4-consistent; see campaign_recon
    # REVAL_RESULTS.md). Vertex/direction hit the SIREN-emitter floor (~15 cm / sub-degree);
    # a residual energy bias (~few %) reflects SIREN-vs-GEANT4 emission fidelity, not the fit.
    band = tuple(args.cherenkov_band) if args.cherenkov_band and args.cherenkov_band[1] > 0 else None
    pred = setup_event_simulator(args.geom, args.nph, temperature=0.1, K=8, hit_mode='per_photon',
                                 physics_config=args.physics, default_detector_params=True,
                                 particle='muon', wavelength_mode=True, pos_grad_threshold=8,
                                 n_grad_iters=8, cherenkov_emission_band=band, **GRID)
    model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
    dummy = track_from_vec9(jnp.asarray(vec9_from_track(1000., [0, 0, 0], [0, 0, 1], t0=0.)))

    def make_seed(vtx, t0g, e0, ocf, otf):
        c2 = hierarchical_direction_search_cone(pred, jnp.asarray(vtx), t0g, POSf, otf, ocf,
                                                (ocf, otf), e0, 3, 8, 90., 0.5, 0)
        dg = np.array([np.sin(c2['best_theta'])*np.cos(c2['best_phi']),
                       np.sin(c2['best_theta'])*np.sin(c2['best_phi']), np.cos(c2['best_theta'])])
        return vec9_from_track(e0, np.asarray(vtx), dg, t0=t0g)

    rows = []
    print(f'{ND} sensors | reconstructing events {args.start}..{args.start+args.events-1} from {os.path.basename(args.root)}')
    print(f'{"ev":>4s}{"vtx(cm)":>9s}{"dir(deg)":>9s}{"dE(MeV)":>9s}{"t0(ns)":>8s}')
    for ev in range(args.start, args.start + args.events):
        raw = read_photon_data_from_photonsim(args.root, ev)
        energy = float(raw['energy']) if 'energy' in raw else args.energy
        raw, vtx_true, dir_true = rand_tf(raw, ev, args.fidr, args.fidz)
        th9, d = truth9(vtx_true, dir_true, energy)
        pd, _ = pad_photon_data(raw, args.nbuf)
        c, t = jax.lax.stop_gradient(data_sim(dummy, jax.random.PRNGKey(7000 + ev), pd))
        oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
        ocf, otf = jnp.asarray(oc), jnp.asarray(ot)
        e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1/jnp.sqrt(3)), jnp.pi/4, 0.,
                                      POSf, otf, ocf, (ocf, otf), 1000., 700., 12, 0)['best_energy']
        p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.zeros(3), 0., 0., bounds,
                                               n_div=5, t0_n_div=5, levels=6, verbosity=0)
        seedA = make_seed(np.asarray(p1['best_position']), float(p1['best_t0']), e0, ocf, otf)
        seedB = make_seed(*seed_vertex_time(POS, oc, ot), e0, ocf, otf)
        res, _ = fit_track_multistart(model, oc, ot, [seedA, seedB], nkeys=args.nkeys, niters=args.niters)
        vtx_err = float(np.linalg.norm(res[1:4] - th9[1:4]) * 100)
        dir_err = float(np.degrees(np.arccos(np.clip(vec9_dir(res) @ d, -1, 1))))
        dE = float(res[0] - energy); dt0 = float(res[8])
        rows.append((ev, vtx_err, dir_err, dE, dt0))
        np.savez(os.path.join(args.out, f'event_{ev}.npz'), truth=th9, fit=np.asarray(res),
                 vtx_err_cm=vtx_err, dir_err_deg=dir_err, dE_mev=dE, dt0_ns=dt0)
        print(f'{ev:>4d}{vtx_err:>9.1f}{dir_err:>9.2f}{dE:>+9.1f}{dt0:>+8.2f}', flush=True)

    a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f'\nresolution over {len(rows)} events (median | RMS):')
    for j, (name, unit) in enumerate([('vertex', 'cm'), ('direction', 'deg'), ('energy', 'MeV'), ('t0', 'ns')]):
        print(f'  {name:10s} {np.median(np.abs(a[:,j])):7.2f} | {np.sqrt(np.mean(a[:,j]**2)):7.2f}  {unit}')
    wanderers = int((a[:, 0] > 40).sum())
    print(f'  wanderers (vtx > 40 cm): {wanderers}/{len(rows)}')
    print(f'per-event npz written to {args.out}/')


if __name__ == '__main__':
    main()
