"""Exact muon-gun truth: the gun fires from the ORIGIN (0,0,0) in +z (verified — most-upstream
primary-Cherenkov emission is (0,0,0) for every event), and rand_tf applies a known (R, sh) about
the photon-origin centroid c. So the exact truth is that transform of (origin, +z) — no PCA.
Shared by rescore_truth / aggregate_ms / plots so every tool scores against the same real truth."""
import numpy as np, uproot
ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
FIDR, FIDZ = 12., 12.


def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def _params(ev):                                          # replicate rand_tf's RNG order EXACTLY
    rng = np.random.default_rng(100003 + ev)
    beta = np.degrees(np.arccos(rng.uniform(-1, 1)))
    al = rng.uniform(0, 2 * np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
    rr = FIDR * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
    sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-FIDZ, FIDZ)]) * 100.0
    return _rotax(axis, beta), sh


def exact_truth(evs):
    """evs: iterable of event ids -> {ev: (vtx_m (3,), dir (3,))} for the exact transformed gun truth."""
    evs = list(evs)
    tree = uproot.open(ROOT)['OpticalPhotons']
    A = tree.arrays(['PhotonPosX', 'PhotonPosY', 'PhotonPosZ'],
                    entry_start=min(evs), entry_stop=max(evs) + 1, library='np')
    base = min(evs); out = {}
    for ev in evs:
        O = np.column_stack([A['PhotonPosX'][ev - base], A['PhotonPosY'][ev - base],
                             A['PhotonPosZ'][ev - base]]) / 10.
        c = O.mean(0); R, sh = _params(ev)
        out[ev] = (((np.zeros(3) - c) @ R.T + c + sh) / 100.0, np.array([0, 0, 1.]) @ R.T)
    return out


def vec9_dir(v):
    st, ct, sp, cp = float(v[4]), float(v[5]), float(v[6]), float(v[7])
    nt = np.hypot(st, ct); npp = np.hypot(sp, cp); st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp
    return np.array([st * cp, st * sp, ct])


def fit_err_exact(fit9, vtx_true, dir_true):
    """[vtx cm, dir deg, dE MeV, dt0 ns] of a 9-vec fit vs exact truth (energy/t0 truth: 1050, 0)."""
    dv = np.linalg.norm(fit9[1:4] - vtx_true) * 100
    dd = float(np.degrees(np.arccos(np.clip(vec9_dir(fit9) @ dir_true, -1, 1))))
    return np.array([dv, dd, float(fit9[0] - 1050.), float(fit9[8] - 0.)])
