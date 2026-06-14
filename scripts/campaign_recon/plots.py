"""Distributions of the two-start recon results + per-parameter optimization-trajectory plots.
Reads out_ms100/ (two-start) and out/ (old single-start baseline). Writes PNGs to OUT dir.
Usage: PYTHONPATH=. python scripts/campaign_recon/plots.py
"""
import os, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from truth_exact import exact_truth, fit_err_exact, vec9_dir as _vd
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.environ.get('NEW', os.path.join(HERE, 'out_ms100'))
OLD = os.path.join(HERE, 'out')


def vec9_dir(v):                                    # vectorized form (for trajectory arrays)
    st, ct, sp, cp = v[..., 4], v[..., 5], v[..., 6], v[..., 7]
    nt = np.hypot(st, ct); npp = np.hypot(sp, cp)
    return np.stack([st / nt * cp / npp, st / nt * sp / npp, ct / nt], -1)


MARGIN = float(os.environ.get('MARGIN', '0.01'))    # arbitration: pick time-seed only if >1% lower loss
nf = sorted(glob.glob(os.path.join(NEW, 'ev*.npz')))
D = [np.load(f) for f in nf]
N = len(D)
ev_ids = [int(os.path.basename(f)[2:5]) for f in nf]
ET = exact_truth(ev_ids)                            # exact gun truth per event — score EVERYTHING vs this
# per-seed errors vs EXACT truth, recomputed from saved 9-vecs; margin-gated (prefer A=charge)
errA = np.array([fit_err_exact(d['fitA'], *ET[ev]) for d, ev in zip(D, ev_ids)])
errB = np.array([fit_err_exact(d['fitB'], *ET[ev]) for d, ev in zip(D, ev_ids)])
L = np.array([d['losses'] for d in D]); relgap = (L[:, 0] - L[:, 1]) / np.abs(L[:, 0])
which = (relgap > MARGIN).astype(int)               # winning seed under the 1% margin
fit = np.where(which[:, None] == 1, errB, errA)     # (N,4) margin-selected, vs exact
which_saved = np.array([int(d['which']) for d in D])  # the basin whose trajectory was saved (argmin)
seedA = np.array([fit_err_exact(d['seedA'], *ET[ev])[0] for d, ev in zip(D, ev_ids)])
seedB = np.array([fit_err_exact(d['seedB'], *ET[ev])[0] for d, ev in zip(D, ev_ids)])
# old single-start baseline, re-scored vs exact too
old = np.array([fit_err_exact(np.load(os.path.join(OLD, f'ev{e:03d}.npz'))['fit'], *ET[e])[0]
                if os.path.exists(os.path.join(OLD, f'ev{e:03d}.npz')) else np.nan for e in ev_ids])

# ---------- Figure 1: final-result distributions ----------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
specs = [(0, 'vertex |Δ|  (cm)', (0, 80), 'fit'), (1, 'direction  (deg)', (0, 6), 'fit'),
         (2, 'energy bias  (MeV)', (-60, 60), 'fit'), (3, 'tₒ bias  (ns)', (-3, 3), 'fit')]
for k, (i, lab, rng, _) in enumerate(specs):
    a = ax.flat[k]; v = fit[:, i]
    a.hist(v, bins=24, range=rng, color='#3477b8', alpha=0.85, edgecolor='white', lw=0.4)
    med = np.median(v); p68 = np.percentile(np.abs(v), 68)
    a.axvline(med, color='crimson', lw=1.5, label=f'median {med:.2f}')
    a.set_title(f'{lab}    (68%={p68:.2f}, RMS={np.sqrt((v**2).mean()):.2f})', fontsize=11)
    a.set_xlabel(lab); a.set_ylabel('events'); a.legend(fontsize=9); a.grid(alpha=0.25)
fig.suptitle(f'Two-start recon — final-parameter distributions  (N={N}, GEANT4 data, 2.5 ns TTS)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(NEW, 'fig1_distributions.png'), dpi=120); plt.close(fig)

# ---------- Figure 2: vertex old-vs-new + seed->fit + which-seed ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
m = ~np.isnan(old)
hb = np.linspace(0, max(60, np.nanpercentile(np.r_[old[m], fit[m, 0]], 98)), 25)
ax[0].hist(old[m], bins=hb, alpha=0.55, label=f'OLD single  (med {np.median(old[m]):.0f})', color='#999')
ax[0].hist(fit[m, 0], bins=hb, alpha=0.65, label=f'TWO-start  (med {np.median(fit[m,0]):.0f})', color='#3477b8')
ax[0].set_xlabel('vertex |Δ| (cm)'); ax[0].set_ylabel('events'); ax[0].legend(); ax[0].grid(alpha=0.25)
ax[0].set_title('Vertex resolution: old vs two-start')
# seed -> fit improvement (best seed of the two)
bestseed = np.minimum(seedA, seedB)
ax[1].scatter(bestseed, fit[:, 0], c=which, cmap='coolwarm', s=28, edgecolor='k', lw=0.3)
ax[1].plot([0, 400], [0, 400], 'k--', lw=0.7, alpha=0.5)
ax[1].set_xlabel('best seed vtx |Δ| (cm)'); ax[1].set_ylabel('fit vtx |Δ| (cm)')
ax[1].set_title('seed → fit (color = winning seed: blue=A charge, red=B time)'); ax[1].grid(alpha=0.25)
ax[1].set_xlim(0, 400); ax[1].set_ylim(0, max(60, fit[:, 0].max() * 1.1))
# which seed won, split by inwardness (cWall from exact truth)
cw = np.array([float(ET[ev][1] @ (lambda r: r / (np.linalg.norm(r) + 1e-9))(np.r_[ET[ev][0][:2], 0.]))
               for ev in ev_ids])
ax[2].scatter(cw, fit[:, 0], c=which, cmap='coolwarm', s=28, edgecolor='k', lw=0.3)
ax[2].set_xlabel('cWall (track·outward-radial; <0 = inward)'); ax[2].set_ylabel('fit vtx |Δ| (cm)')
ax[2].set_title(f'B(time) won {int((which==1).sum())}/{N}, A(charge) {int((which==0).sum())}/{N}'); ax[2].grid(alpha=0.25)
ax[2].set_ylim(0, max(60, fit[:, 0].max() * 1.1))
fig.tight_layout(); fig.savefig(os.path.join(NEW, 'fig2_vertex_seed.png'), dpi=120); plt.close(fig)

# ---------- Figure 3: per-parameter optimization trajectories ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for d, ev in zip(D, ev_ids):
    tr = d['traj']; vtx_true, dir_true = ET[ev]; it = np.arange(len(tr))   # score traj vs EXACT truth
    pos_err = np.linalg.norm(tr[:, 1:4] - vtx_true, axis=1) * 100
    dirs = vec9_dir(tr)
    ang = np.degrees(np.arccos(np.clip(dirs @ dir_true, -1, 1)))
    e_err = tr[:, 0] - 1050.; t0_err = tr[:, 8] - 0.
    c = 'crimson' if int(d['which']) == 1 else '#3477b8'   # color = saved (lower-loss) basin of this traj
    ax[0, 0].plot(it, pos_err, c=c, alpha=0.35, lw=0.7)
    ax[0, 1].plot(it, ang, c=c, alpha=0.35, lw=0.7)
    ax[0, 2].plot(it, e_err, c=c, alpha=0.35, lw=0.7)
    ax[1, 0].plot(it, t0_err, c=c, alpha=0.35, lw=0.7)
    ax[1, 1].semilogy(it, d['gnorm'], c=c, alpha=0.35, lw=0.7)
ax[0, 0].set_title('vertex error (cm)'); ax[0, 0].set_ylim(0, 200)
ax[0, 1].set_title('direction error (deg)'); ax[0, 1].set_ylim(0, 12)
ax[0, 2].set_title('energy error (MeV)'); ax[0, 2].set_ylim(-150, 200); ax[0, 2].axhline(0, color='k', lw=0.6)
ax[1, 0].set_title('tₒ error (ns)'); ax[1, 0].set_ylim(-10, 10); ax[1, 0].axhline(0, color='k', lw=0.6)
ax[1, 1].set_title('‖g‖ (scaled gradient norm)')
for a in ax.flat[:5]:
    a.set_xlabel('GN iteration'); a.grid(alpha=0.25)
ax[1, 2].axis('off')
ax[1, 2].text(0.05, 0.6, f'N = {N} events\nblue = charge-seed (A) won\nred = time-seed (B) won\n\n'
              f'final vtx median {np.median(fit[:,0]):.1f} cm\nfinal dir median {np.median(fit[:,1]):.2f}°\n'
              f'final E median {np.median(fit[:,2]):+.1f} MeV\nfinal tₒ median {np.median(fit[:,3]):+.2f} ns',
              fontsize=12, va='top', family='monospace')
fig.suptitle(f'Two-start recon — per-parameter optimization trajectories  (N={N})', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(NEW, 'fig3_trajectories.png'), dpi=120); plt.close(fig)
print(f'wrote fig1_distributions.png, fig2_vertex_seed.png, fig3_trajectories.png to {NEW}  (N={N})')
