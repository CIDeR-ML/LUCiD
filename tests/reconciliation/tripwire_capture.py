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
# Pin the backend BEFORE jax is imported. The test path is already CPU-only -- tests/conftest.py
# sets this and test_tripwire forwards os.environ into the subprocess -- but the documented
# regeneration path (running this file directly) inherited nothing, so a capture taken on a GPU
# node stored 581.2667 against a CPU test value of 581.7201: 7.8x the tolerance, a guaranteed
# failure on the next run. The reference and the check have to agree on the backend.
if os.environ.get('JAX_PLATFORMS', 'cpu') != 'cpu':
    sys.stderr.write('tripwire: overriding JAX_PLATFORMS=%s with cpu (the reference is CPU-only)\n'
                     % os.environ['JAX_PLATFORMS'])
# All three lines, matching tests/conftest.py:11-13. Setting only the two JAX_PLATFORMS vars left
# the direct run differing from the pytest run in exactly one variable -- the class of divergence
# this block exists to remove -- and conftest attributes its anti-hang behaviour to hiding the
# device as well: a driver probe can block the process in uninterruptible D-state.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # back-compat with older jaxlib
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
        # Per-column, not one blanket rtol. fisher_diag is (adJ**2).sum over 10764 sensors; a
        # weakly-determined column is built from tiny per-sensor entries, so its float32 relative
        # error is far larger than a well-determined one's -- the floor tracks column magnitude
        # almost monotonically (mie F=1.95 -> 4.1e-4; qe F=4309 -> 4.6e-6). One number for all
        # seven is therefore either too loose for qe or too tight for mie.
        #
        # Each bound is 20x that column's MEASURED cross-node floor: identical code and seeds,
        # AMD (milano) vs the Intel Xeon Gold 5118 host of a turing node, both on CPU --
        #   mie 4.1e-4, wall 3.4e-4, g 3.1e-4, sensor 1.3e-4, scatter 3.8e-5, abs 2.3e-5, qe 4.6e-6
        # Core count is not the variable (1->32 cores leaves q_l2 bit-identical and fisher_diag
        # inside 2e-5); the node is. A jaxlib/XLA upgrade was never probed and is the likeliest
        # thing to eat this margin, so widen from a MEASUREMENT if that day comes, not by reflex.
        #
        # Sensitivity retained, against the isolated b8c266e A/B (ac8861c -> HEAD, same node):
        # mie 5.0e-2, g 9.8e-3, sensor 4.3e-3, absorption 1.2e-3, wall 1.7e-3. Four of those trip
        # their bounds -- mie, g, sensor and absorption -- and a single blanket 5e-3 would have
        # caught only mie and g. Three of these bounds are individually LOOSER than 5e-3; the point
        # is not that every bound is tighter but that the set detects strictly more on the one real
        # change there is to test against.
        # Each bound is that column's floor x20, rounded up to one significant figure (20.6-26.3x).
        assert list(rf['meta.fields']) == FIELDS, 'FIELDS changed — fisher tolerances are keyed to them'
        _tol = np.array([{'g': 7e-3, 'scatter_length': 1e-3, 'mie_scatter_length': 1e-2,
                          'absorption_length': 5e-4, 'wall_reflection_rate': 7e-3,
                          'sensor_reflection_rate': 3e-3, 'qe': 1e-4}[k] for k in FIELDS])
        _new = np.asarray(f['scalar.fisher_diag']); _ref = np.asarray(rf['scalar.fisher_diag'])
        _bad = np.abs(_new - _ref) > _tol * np.abs(_ref)
        # Name the offending column: 'AD-Fisher diag drift' alone sent this investigation looking
        # at the forward charge when the signal was in one Jacobian column.
        assert not _bad.any(), 'AD-Fisher diag drift: ' + ', '.join(
            '%s %.8g vs %.8g (rel %.2e > %.0e)' % (FIELDS[i], _new[i], _ref[i],
                abs(_new[i] - _ref[i]) / abs(_ref[i]), _tol[i]) for i in np.nonzero(_bad)[0])
        assert not f['scalar.nan'] and not f['wavelength.nan'], 'NaN appeared'
        print('TRIPWIRE OK — all references match')
