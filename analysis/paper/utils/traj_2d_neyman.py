"""Two random-start optimization trajectories to overlay on the Neyman 2D loss surface.

Re-evaluates the SAME Neyman chi^2 loss, keys (PRNGKey(3)/PRNGKey(101), NK=8/8), and per-point
Fisher F = 2 sum (dM/dtheta)^2 / Q as compute_2d_neyman.py, so each iterate sits on the exact
surface that make_2d_neyman.py plots (the paths follow the shown streamlines by construction).

  - GD  (left  panel): backtracking-line-search steepest descent  x <- x - t grad L, run to convergence.
  - GN  (right panel): DAMPED Fisher step (Marquardt lam*diag(F) + Levenberg mu*median(diag F),
                       plain solve, no eigen-floor) x <- x - F^-1 grad L, run to convergence.
                       lam/mu come from CALIB_RECIPE and the step from utils/damping.py, so
                       this arm takes the same step the joint fit takes by construction.

Both converge to the realized minimum (the cyan X, displaced from truth by the independent-key noise).
Saves data/traj2d_neyman.npz : starts(2,2), min(2), gd0,gd1,gn0,gn1 (each (T,2) in metres).
"""
import os, sys, time
from pathlib import Path
# Repo-relative (was a hardcoded absolute path before the 2026-08-12 move into analysis/paper).
REPO = str(Path(__file__).resolve().parents[3]); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, jax, jax.numpy as jnp
from analysis.paper.utils import calibration as C, damping
from analysis.paper.utils import paths
FIGURE = "calib_loss_geometry"
from lucid.detector_params import DetectorParams

# ---- identical setup to compute_2d_neyman.py (single laser, scalar_mix, 405 nm) ----
truth = C.effective_truth(C.REP_WL)
sim = C._cal_sim(); src = C._laser_sources()[0]
tx, ty = float(truth["scatter_length"]), float(truth["absorption_length"])
def build_dp(v):                                                   # v = [lambda_s, lambda_a] differentiable
    return DetectorParams.from_flat(
        scatter_length=v[0], mie_scatter_length=1e6, g=0.9,
        wall_reflection_rate=C.WALL_R_MIX,   sensor_reflection_rate=C.SENSOR_R_MIX,
        wall_fspec=C.WALL_FSPEC,             sensor_fspec=C.SENSOR_FSPEC,
        absorption_length=v[1], qe=truth["qe"], qe_corrections=jnp.ones(C.NS))
# MUST match the surface in compute_2d_neyman.py: the loss floor goes as (1/NK_SIM + 1/NK_TRUTH),
# so a mismatch puts every iterate on a DIFFERENT surface than the one plotted and moves the realized
# minimum. Defaults are compute_2d_neyman.py's own (8/4), so the two agree unless BOTH are overridden
# together. The original figure used 8/8; reproduce it with NK_SIM=8 passed to both scripts.
NK_TRUTH = int(os.environ.get("NK_TRUTH", "8")); NK_SIM = int(os.environ.get("NK_SIM", "4"))
_tkeys = jax.random.split(jax.random.PRNGKey(3), NK_TRUTH)
_skeys = jax.random.split(jax.random.PRNGKey(101), NK_SIM)         # independent of truth -> min displaced by noise
@jax.jit
def _mean_over(v, keys):        # accumulate, do not stack: peak memory then does not scale with NK
    acc = sim(src, build_dp(v), keys[0])[0]
    for k in keys[1:]: acc = acc + sim(src, build_dp(v), k)[0]
    return acc / len(keys)
def M_truth(v): return _mean_over(v, _tkeys)
@jax.jit
def M_fn(v):    return _mean_over(v, _skeys)
Q = jax.lax.stop_gradient(M_truth(jnp.array([tx, ty])))
Winv = 1.0 / jnp.maximum(Q, 1e-9)                                  # Neyman weight 1/Q
Wn = np.asarray(Winv)
@jax.jit
def L(v):                                                          # Neyman chi^2 on charge
    M = M_fn(v); return jnp.sum((M - Q) ** 2 * Winv)
Lg = jax.jit(jax.value_and_grad(L))
Jf = jax.jit(jax.jacfwd(M_fn))
def fisher(v):
    J = np.asarray(Jf(v)); return 2.0 * np.einsum("np,nq,n->pq", J, J, Wn)  # GN / Fisher matrix
# The campaign's damping, read from the recipe rather than restated, and applied through the one
# definition this figure's two halves share — so "the GN arm takes the same step the joint fit
# takes" holds by construction instead of by two files spelling out the same arithmetic.
LAM, MU = float(C.CALIB_RECIPE['LAM']), float(C.CALIB_RECIPE['MU'])
def finv_g(F, g):
    return damping.damped_step(F, g, LAM, MU)

t0 = time.time(); _ = Lg(jnp.array([tx, ty])); _ = fisher(jnp.array([tx, ty]))
print(f"[traj] warm {time.time()-t0:.0f}s", flush=True)

# ---- box + realized minimum from the plotted grid ----
d = np.load(str(paths.data_dir(FIGURE) / "landscape2d_neyman.npz"))
X, Y, Lgrid = d["X"], d["Y"], d["L"]
im, jm = np.unravel_index(np.argmin(Lgrid), Lgrid.shape); mx, my = float(X[im]), float(Y[jm])
xlo, xhi, ylo, yhi = float(X.min()), float(X.max()), float(Y.min()), float(Y.max())

# ---- two starts in diagonally opposite corners: top-left (low lambda_s, high lambda_a) and
#      bottom-right (high lambda_s, low lambda_a), so the two crawls reach the razor valley from
#      opposite sides ----
def corner(fx, fy):
    return np.array([xlo + fx * (xhi - xlo), ylo + fy * (yhi - ylo)])
starts = [corner(0.18, 0.88), corner(0.82, 0.12)]                  # top-left, bottom-right
print(f"[traj] starts {[[round(float(s[0]),1), round(float(s[1]),1)] for s in starts]}", flush=True)

# ---- GD: plain FIXED-STEP gradient descent  x <- x - eta*grad L. Its path is an integral curve of the
#      -grad L field, so it lies exactly on the plotted streamlines. With eta ~ 0.9/lambda_max it clears
#      the stiff lambda_s direction in ~1 step then crawls the soft lambda_a valley at rate (1-1/kappa),
#      ~O(kappa) steps -- the visible ill-conditioning that the Fisher preconditioner removes. ----
def gd(x0, eta, maxit=500, xtol=5e-2, cap=20.0):
    x = np.asarray(x0, float); path = [x.copy()]
    for it in range(maxit):
        _, g = Lg(jnp.array(x)); g = np.asarray(g)
        step = eta * g; n = np.linalg.norm(step)
        if n > cap: step = step * (cap / n)                         # trust region for the far-field
        x = x - step; path.append(x.copy())
        if np.linalg.norm(step) < xtol: break                       # step -> 0: converged to the min
        if it % 25 == 0: print(f"[traj]   gd it{it} x=({x[0]:.1f},{x[1]:.1f}) |step|={np.linalg.norm(step):.3g}", flush=True)
    return np.array(path)

# ---- GN: damped Fisher step (utils/damping.py, no eigen-floor), to convergence ----
def gn(x0, maxit=40, cap=80.0, dtol=5e-2):
    x = np.asarray(x0, float); path = [x.copy()]
    for it in range(maxit):
        _, g = Lg(jnp.array(x)); g = np.asarray(g)
        step = -finv_g(fisher(jnp.array(x)), g)
        n = np.linalg.norm(step)
        if n > cap: step = step * (cap / n)                        # trust region so a far start stays in frame
        x = x + step; path.append(x.copy())
        if np.linalg.norm(step) < dtol: break
    return np.array(path)

xmin = np.array([mx, my])
lam_max = float(np.linalg.eigvalsh(fisher(jnp.array(xmin))).max())   # stiff curvature at the min
eta = 0.9 / lam_max
print(f"[traj] lambda_max(F@min)={lam_max:.4g}  eta={eta:.4g}", flush=True)
res = {}
for i, s in enumerate(starts):
    t0 = time.time(); pg = gd(s, eta)
    print(f"[traj] gd{i}: {len(pg)} steps {time.time()-t0:.0f}s -> ({pg[-1,0]:.1f},{pg[-1,1]:.1f})", flush=True)
    t0 = time.time(); pn = gn(s)
    print(f"[traj] gn{i}: {len(pn)} steps {time.time()-t0:.0f}s -> ({pn[-1,0]:.1f},{pn[-1,1]:.1f})", flush=True)
    res[f"gd{i}"] = pg; res[f"gn{i}"] = pn
out = str(paths.data_dir(FIGURE) / "traj2d_neyman.npz")
np.savez(out, starts=np.array(starts), min=np.array([mx, my]), **res)
print(f"[traj] saved {out}", flush=True)
