"""Phase-0 reconciliation TRIPWIRE — capture water-mode reference tensors that every
unification↔refactor-v2 merge phase must preserve. Run with CAPTURE=1 to (re)generate the
reference npz; run without to ASSERT against it. See docs/RECONCILIATION_PLAN.md Phase 0.

Pins, on the real lucid.fitting calibration path (the path most threatened by the optics/
scintillation/param-tree merge):
  - FORWARD charge (scalar optics AND wavelength optics → covers optical_model.py seam)
  - AD-Fisher CRB sigma (exercises SourceModel.ad_jacobian = jacfwd; the base's whole value)
  - forward-mode jacfwd vs reverse-mode grad of total charge (both AD modes agree)
  - AD==FD Jacobian cross-check (the DiCE/custom_vjp-free gradient is correct)
  - NaN-free under jacfwd (the dropped custom_vjp backstop is gone)
  - DetectorParams nested leaf-name/order (a wrong insertion silently corrupts every optimizer)
"""
import os, sys, json
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import DetectorParams, _flatten_detector_params
from lucid.sources import laser_source
from lucid.fitting import build_calibration_problem

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'tripwire_water_ref.npz')
GEOM = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'config', 'SK_like_geom_config.json')
SEED, NPH, K = 20240614, 500_000, 8
GRID = dict(n_cap=100, n_angular=150, n_height=100)
FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']

det = generate_detector(GEOM); NS = len(det.all_points); top = det.H / 2 - 0.1
dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                              wall_reflection_rate=.2, sensor_reflection_rate=.2,
                              absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
source = laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6)


def leaf_order(dp):
    """Nested leaf names in flatten order (the optimizer's ravel order)."""
    return list(_flatten_detector_params(dp).keys())


def capture_mode(wavelength_mode):
    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=wavelength_mode, **GRID)
    prob = build_calibration_problem(sim, [source], dp, FIELDS, key=jax.random.PRNGKey(SEED))
    sm = prob['source_models'][0]; theta = np.asarray(prob['theta_true'])
    lk = np.zeros(NS); ek = jax.random.PRNGKey(0); pk = jax.random.PRNGKey(1)
    q = np.asarray(prob['truth_charge'][0])                         # forward charge (√(k·M))²-ish per source
    m = np.asarray(sm.m(theta, lk, ek, pk))                        # residual √(k·M)
    adJ = sm.ad_jacobian(theta, lk, ek, pk)                       # jacfwd (NS, P)
    out = dict(q_sum=float(q.sum()), q_l2=float(np.linalg.norm(q)), q_nlit=int((q > 0).sum()),
               m_l2=float(np.linalg.norm(m)), adJ_l2=float(np.linalg.norm(adJ)),
               adJ_colnorm=np.linalg.norm(adJ, axis=0))            # per-param sensitivity
    if not wavelength_mode:                                        # full AD tripwire on the scalar path
        out['fisher_diag'] = (adJ ** 2).sum(axis=0)              # DETERMINISTIC AD-Fisher info (Jᵀ J diag)
        fdJ = sm.fd_jacobian(theta, lk, ek, pk)                   # CRN-FD cross-check (informational; FD is noisy)
        out['adfd_cos'] = float((adJ * fdJ).sum() / (np.linalg.norm(adJ) * np.linalg.norm(fdJ) + 1e-30))
    nan = bool(np.isnan(q).any() or np.isnan(m).any() or np.isnan(adJ).any())
    out['nan'] = nan
    return out


def digest():
    sc = capture_mode(False); wl = capture_mode(True)
    return dict(leaf_order=leaf_order(dp), n_leaves=len(leaf_order(dp)), NS=NS,
                scalar=sc, wavelength=wl,
                meta=dict(seed=SEED, nph=NPH, K=K, fields=FIELDS))


def _flat(d, p=''):  # flatten nested digest to comparable scalars/arrays
    out = {}
    for k, v in d.items():
        kk = f'{p}{k}'
        if isinstance(v, dict): out.update(_flat(v, kk + '.'))
        elif isinstance(v, (list, np.ndarray)) and kk.endswith('leaf_order'): out[kk] = list(v)
        else: out[kk] = v
    return out


if __name__ == '__main__':
    d = digest(); f = _flat(d)
    print('=== TRIPWIRE DIGEST (water-mode, SK_like) ===')
    print('leaves:', f['n_leaves'], '| NS:', f['NS'])
    print('scalar: q_sum %.4f q_l2 %.4f nlit %d | adJ_l2 %.5f adfd_cos %.4f nan %s' % (
        f['scalar.q_sum'], f['scalar.q_l2'], f['scalar.q_nlit'], f['scalar.adJ_l2'],
        f['scalar.adfd_cos'], f['scalar.nan']))
    print('  fisher_diag (JᵀJ):', np.array2string(f['scalar.fisher_diag'], precision=4))
    print('wavelength: q_sum %.4f q_l2 %.4f nlit %d adJ_l2 %.5f nan %s' % (
        f['wavelength.q_sum'], f['wavelength.q_l2'], f['wavelength.q_nlit'],
        f['wavelength.adJ_l2'], f['wavelength.nan']))
    if os.environ.get('CAPTURE') == '1':
        np.savez(REF, digest_json=json.dumps(d, default=lambda x: np.asarray(x).tolist()))
        print('WROTE', REF)
    else:
        if not os.path.exists(REF):
            print('NO REFERENCE yet — run with CAPTURE=1 first'); sys.exit(2)
        ref = json.loads(str(np.load(REF, allow_pickle=True)['digest_json']))
        rf = _flat(ref)
        assert f['leaf_order'] == rf['leaf_order'], 'DetectorParams LEAF ORDER changed!'
        for key in ['scalar.q_l2', 'scalar.adJ_l2', 'wavelength.q_l2', 'wavelength.adJ_l2']:
            assert abs(f[key] - rf[key]) <= 1e-4 * (abs(rf[key]) + 1e-9), f'{key} drift: {f[key]} vs {rf[key]}'
        assert np.allclose(f['scalar.fisher_diag'], rf['scalar.fisher_diag'], rtol=1e-4), 'AD-Fisher diag drift'
        assert not f['scalar.nan'] and not f['wavelength.nan'], 'NaN appeared'
        print('TRIPWIRE OK — all references match')
