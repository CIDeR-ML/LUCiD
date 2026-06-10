"""One calibration config → CRB (+ optional recovery / shot-noise) → JSON. GPU worker.

Env:
  NPH (photons), K, GRID(1=full/0=reduced), SRC (source-combo key), INTENS,
  RECOVER(1: implicit-engine recovery fit), SHOT(1: shot-noise scatter, M seeds),
  M, STEPS, NB_H, EPS(0.375=Anscombe), BAKE_K, POLYAK, TAG (output filename stem).
Writes grid_out/<TAG>.json with {nph, src, crb:{param:sigma}, recover:{...}, shot:{...}}.
"""
import os, sys, json, time
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150) if os.environ.get('GRID', '1') == '1' \
    else dict(n_cap=100, n_angular=150, n_height=100)
NPH = int(float(os.environ.get('NPH', '1e6')))
K = int(os.environ.get('K', '8'))
SRC = os.environ.get('SRC', 'laser_iso')
INTENS = float(os.environ.get('INTENS', '1e8'))
RECOVER = os.environ.get('RECOVER', '0') == '1'
SHOT = os.environ.get('SHOT', '0') == '1'
M = int(os.environ.get('M', '4'))
STEPS = int(os.environ.get('STEPS', '100'))
NB_H = int(os.environ.get('NB_H', '2'))
EPS = float(os.environ.get('EPS', '1e-8'))
BAKE_K = os.environ.get('BAKE_K', '0') == '1'
POLYAK = int(os.environ.get('POLYAK', '0'))
PERT = float(os.environ.get('PERT', '0.15'))
TAG = os.environ.get('TAG', f'{SRC}_N{NPH:.0e}')
OUT = os.path.join(_HERE, 'grid_out'); os.makedirs(OUT, exist_ok=True)

FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
LABEL = ['g', 'L_R', 'L_M', 'L_abs', 'wall', 'sensor', 'qe']


def make_sources(det, H, R):
    """Return the list of source objects for the SRC key."""
    def L(pos, d): return laser_source(position=pos, direction=d, intensity=INTENS)
    def I(pos): return isotropic_source(position=pos, intensity=INTENS)
    top, bot = H/2 - 0.1, -H/2 + 0.1
    reg = {
        'laser_down': [L([0, 0, top], [0, 0, -1])],
        'laser_up': [L([0, 0, bot], [0, 0, 1])],
        'laser_wall': [L([R - 0.1, 0, 0], [-1, 0, 0])],
        'laser_diag': [L([0, 0, top], [0.6, 0, -0.8])],
        'iso_center': [I([0, 0, 0])],
        'iso_off': [I([R/2, 0, 0])],
        'iso_top': [I([0, 0, top])],
        'laser_iso': [L([0, 0, top], [0, 0, -1]), I([0, 0, 0])],
        'multi_laser': [L([0, 0, top], [0, 0, -1]), L([0, 0, bot], [0, 0, 1]),
                        L([R - 0.1, 0, 0], [-1, 0, 0])],
        'multi_laser_iso': [L([0, 0, top], [0, 0, -1]), L([0, 0, bot], [0, 0, 1]),
                            L([R - 0.1, 0, 0], [-1, 0, 0]), I([0, 0, 0])],
        'iso_ring': [I([R/2, 0, 0]), I([-R/2, 0, 0]), I([0, R/2, 0]), I([0, -R/2, 0])],
        'all': [L([0, 0, top], [0, 0, -1]), L([0, 0, bot], [0, 0, 1]),
                L([R - 0.1, 0, 0], [-1, 0, 0]), I([0, 0, 0]), I([R/2, 0, 0])],
    }
    return reg[SRC]


def main():
    t0 = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    R = det.r if hasattr(det, 'r') else 6.0
    dp = DetectorParams.from_flat(
        scatter_length=70., mie_scatter_length=3000., g=0.9,
        wall_reflection_rate=.2, sensor_reflection_rate=.2, absorption_length=60.,
        qe=0.07, qe_corrections=jnp.ones(NS))
    srcs = make_sources(det, H, R)
    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False, **GK)

    out = dict(tag=TAG, src=SRC, nph=NPH, intens=INTENS, k_iter=K,
               n_sources=len(srcs), grid=GK)

    prob = build_calibration_problem(sim, srcs, dp, FIELDS, key=jax.random.PRNGKey(1), eps=EPS)
    c = crb(prob['source_models'], prob['theta_true'], NS, nb_h=NB_H)
    out['crb'] = {LABEL[i]: float(c['sigma'][i]) for i in range(len(FIELDS))}
    out['t_crb'] = time.time() - t0

    if RECOVER:
        rng = np.random.default_rng(0)
        start = prob['theta0'] + rng.uniform(-PERT, PERT, prob['theta0'].shape)
        res = fit(prob['source_models'], prob['truth_charge'], start, NS,
                  steps=STEPS, refresh=15, nb_h=NB_H, bake_k=BAKE_K, polyak=POLYAK)
        truth = np.exp(prob['theta0'])
        out['recover'] = {LABEL[i]: dict(truth=float(truth[i]), rec=float(res['theta'][i]),
                                         ferr=float(abs(res['theta'][i]/truth[i]-1)))
                          for i in range(len(FIELDS))}

    if SHOT:
        sim_data = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                         use_expected_value=False, hit_mode='realistic',
                                         apply_smearing=False, wavelength_mode=False, **GK)
        dpk = dp  # truth_k = ones here (focus on globals); could randomize
        rec = np.zeros((M, len(FIELDS)))
        for m in range(M):
            tshot = [np.asarray(sim_data(s, dpk, jax.random.PRNGKey(100 + m * 17 + j))[0])
                     for j, s in enumerate(srcs)]
            r = fit(prob['source_models'], tshot, prob['theta_true'], NS, steps=STEPS,
                    refresh=max(20, STEPS // 2), nb_h=NB_H, seed=m,
                    bake_k=BAKE_K, polyak=POLYAK, eps=EPS)
            rec[m] = r['theta']
        truth = np.exp(prob['theta0'])
        out['shot'] = {LABEL[i]: dict(bias=float(rec[:, i].mean()/truth[i]-1),
                                      sigma=float(rec[:, i].std()/truth[i]),
                                      crb=float(c['sigma'][i]))
                       for i in range(len(FIELDS))}
        out['shot_M'] = M

    out['t_total'] = time.time() - t0
    with open(os.path.join(OUT, f'{TAG}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[{TAG}] done in {out["t_total"]:.0f}s; CRB qe={out["crb"]["qe"]*100:.2f}% '
          f'L_abs={out["crb"]["L_abs"]*100:.2f}%', flush=True)


if __name__ == '__main__':
    main()
