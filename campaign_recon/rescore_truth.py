"""Re-score saved fits against the EXACT truth (muon gun = origin, +z, transformed by rand_tf),
instead of derive_truth's PCA-of-photon-origins proxy. PCA fits the AVERAGE of the multiple-
scattered track (vertex pulled by lateral scatter, direction = path-average not initial), so it
disagrees with the true initial track by ~cm / ~deg — at the level of our quoted resolution.
rand_tf is deterministic per event, so we recompute the exact truth and re-score offline."""
import os, glob, numpy as np, uproot
ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('OUT', os.path.join(HERE, 'out_ms100'))
FIDR, FIDZ = 12., 12.


def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def rand_tf_params(ev):                                   # replicate rand_tf's RNG order EXACTLY
    rng = np.random.default_rng(100003 + ev)
    beta = np.degrees(np.arccos(rng.uniform(-1, 1)))
    al = rng.uniform(0, 2 * np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
    rr = FIDR * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
    sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-FIDZ, FIDZ)]) * 100.0
    return _rotax(axis, beta), sh


def vec9_dir(v):
    st, ct, sp, cp = float(v[4]), float(v[5]), float(v[6]), float(v[7])
    nt = np.hypot(st, ct); npp = np.hypot(sp, cp); st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp
    return np.array([st * cp, st * sp, ct])


def ang(u, v): return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


# centroids c = mean of ALL photon origins (cm), per event — needed because rand_tf rotates about c
tree = uproot.open(ROOT)['OpticalPhotons']
files = sorted(glob.glob(os.path.join(OUT, 'ev*.npz')))
evs = [int(os.path.basename(f)[2:5]) for f in files]
A = tree.arrays(['PhotonPosX', 'PhotonPosY', 'PhotonPosZ'], entry_start=min(evs), entry_stop=max(evs) + 1, library='np')
base = min(evs)

rows = []
for f, ev in zip(files, evs):
    O = np.column_stack([A['PhotonPosX'][ev - base], A['PhotonPosY'][ev - base], A['PhotonPosZ'][ev - base]]) / 10.
    c = O.mean(0)                                          # cm
    R, sh = rand_tf_params(ev)
    vtx_true = ((np.zeros(3) - c) @ R.T + c + sh) / 100.0  # exact gun vertex (origin) transformed, m
    dir_true = np.array([0, 0, 1.]) @ R.T                  # exact gun direction (+z) transformed
    d = np.load(f); fit = d['fit']; th_pca = d['truth']; d_pca = d['tdir']
    fd = vec9_dir(fit)
    rows.append(dict(
        v_exact=np.linalg.norm(fit[1:4] - vtx_true) * 100, v_pca=np.linalg.norm(fit[1:4] - th_pca[1:4]) * 100,
        a_exact=ang(fd, dir_true), a_pca=ang(fd, d_pca),
        proxy_v=np.linalg.norm(th_pca[1:4] - vtx_true) * 100, proxy_a=ang(d_pca, dir_true)))

R_ = {k: np.array([r[k] for r in rows]) for k in rows[0]}
N = len(rows)
print(f'=== Re-score {N} fits: EXACT gun truth (origin,+z transformed) vs PCA proxy (derive_truth) ===\n')
print(f'{"":26s} {"median":>8s} {"mean":>8s} {"RMS":>8s}')
for lab, ke in [('vertex |Δ| vs PCA  (cm)', 'v_pca'), ('vertex |Δ| vs EXACT (cm)', 'v_exact'),
                ('direction vs PCA   (deg)', 'a_pca'), ('direction vs EXACT (deg)', 'a_exact')]:
    v = R_[ke]; print(f'{lab:26s} {np.median(v):8.2f} {v.mean():8.2f} {np.sqrt((v**2).mean()):8.2f}')
print(f'\nPROXY ERROR (derive_truth PCA vs exact gun truth): '
      f'vertex median {np.median(R_["proxy_v"]):.1f}cm (mean {R_["proxy_v"].mean():.1f}), '
      f'direction median {np.median(R_["proxy_a"]):.2f}° (mean {R_["proxy_a"].mean():.2f})')

# ---------- figure: PCA-proxy vs exact-truth scoring ----------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
hb = np.linspace(0, 60, 25)
ax[0].hist(R_['v_pca'], bins=hb, alpha=0.55, color='#999', label=f'vs PCA proxy (med {np.median(R_["v_pca"]):.1f})')
ax[0].hist(R_['v_exact'], bins=hb, alpha=0.7, color='#3477b8', label=f'vs EXACT gun (med {np.median(R_["v_exact"]):.1f})')
ax[0].set_xlabel('vertex |Δ| (cm)'); ax[0].set_ylabel('events'); ax[0].legend(); ax[0].grid(alpha=0.25)
ax[0].set_title('Vertex: ~unchanged (proxy err 3.5cm washes out)')
ab = np.linspace(0, 6, 25)
ax[1].hist(R_['a_pca'], bins=ab, alpha=0.55, color='#999', label=f'vs PCA proxy (med {np.median(R_["a_pca"]):.2f}°)')
ax[1].hist(R_['a_exact'], bins=ab, alpha=0.7, color='#c0392b', label=f'vs EXACT gun (med {np.median(R_["a_exact"]):.2f}°)')
ax[1].set_xlabel('direction error (deg)'); ax[1].set_ylabel('events'); ax[1].legend(); ax[1].grid(alpha=0.25)
ax[1].set_title('Direction: 1.7°→1.0° (PCA averaged the scattered track)')
fig.suptitle(f'Resolution vs truth definition (N={N}, argmin two-start) — PCA proxy was 3.5cm/2.3° off the gun', fontsize=12)
fig.tight_layout(); p = os.path.join(OUT, 'fig4_truth_correction.png'); fig.savefig(p, dpi=120); plt.close(fig)
print(f'wrote {p}')
