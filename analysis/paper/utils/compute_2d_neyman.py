"""Neyman variant of the 2D loss-geometry figure: SAME single laser source & scalar_mix as compute_surf,
but loss = Neyman chi^2 on charge  sum (M-Q)^2/Q, and the preconditioner uses the Fisher / Gauss-Newton
matrix  F = 2 sum (dM/dtheta)^2 / Q  per grid point (from the AD Jacobian) -- as in the calibration fit,
not the finite-diff 2x2 estimate. Linear grid in actual metres.
Saves X, Y, L, Gx, Gy, and the per-point Fisher (Fxx, Fxy, Fyy)."""
import os, sys, time
from pathlib import Path
# Repo-relative (was a hardcoded absolute path before the 2026-08-12 move into analysis/paper).
REPO = str(Path(__file__).resolve().parents[3]); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, jax, jax.numpy as jnp
from analysis.paper.utils import calibration as C
from analysis.paper.utils import paths
FIGURE = "calib_loss_geometry"
from lucid.detector_params import DetectorParams
NGRID = int(os.environ.get("NGRID", "25")); HWF = float(os.environ.get("HWF", "0.55"))
truth = C.effective_truth(C.REP_WL)
sim = C._cal_sim(); src = C._laser_sources()[0]
tx, ty = float(truth["scatter_length"]), float(truth["absorption_length"])
def build_dp(v):                                                   # mirror C._dp, with v=[Ls,La] differentiable
    return DetectorParams.from_flat(
        scatter_length=v[0], mie_scatter_length=1e6, g=0.9,
        wall_reflection_rate=C.WALL_R_MIX,   sensor_reflection_rate=C.SENSOR_R_MIX,
        wall_fspec=C.WALL_FSPEC,             sensor_fspec=C.SENSOR_FSPEC,
        absorption_length=v[1], qe=truth["qe"], qe_corrections=jnp.ones(C.NS))
NK_TRUTH = int(os.environ.get("NK_TRUTH", "8"))                   # truth-data draws
NK_SIM = int(os.environ.get("NK_SIM", "4"))                       # forward draws (cheaper); INDEPENDENT keys
_tkeys = jax.random.split(jax.random.PRNGKey(3), NK_TRUTH)        # -> NO common-random-numbers: the truth is a
_skeys = jax.random.split(jax.random.PRNGKey(101), NK_SIM)        #    fixed noisy dataset, the model is separate,
@jax.jit                                                          #    so the min displaces by the residual noise
def _mean_over(v, keys):
    acc = sim(src, build_dp(v), keys[0])[0]
    for k in keys[1:]: acc = acc + sim(src, build_dp(v), k)[0]
    return acc / len(keys)
def M_truth(v): return _mean_over(v, _tkeys)
@jax.jit
def M_fn(v):    return _mean_over(v, _skeys)
Q = jax.lax.stop_gradient(M_truth(jnp.array([tx, ty])))           # 8-draw truth (independent of the 4-draw model)
Winv = 1.0 / jnp.maximum(Q, 1e-9)                                 # Neyman weight 1/Q
@jax.jit
def L(v):                                                          # Neyman chi^2 on charge
    M = M_fn(v); return jnp.sum((M - Q) ** 2 * Winv)
# PROCESS-LEVEL SHARDING: the grid is embarrassingly parallel, so run N copies (one GPU each via
# CUDA_VISIBLE_DEVICES) — each does its round-robin slice, one point at a time (fits 11GB), saves a partial.
SHARD = int(os.environ.get("SHARD_ID", "0")); NSHARDS = int(os.environ.get("NSHARDS", "1"))

def main():
    Lg = jax.jit(jax.value_and_grad(L)); Jf = jax.jit(jax.jacfwd(M_fn))
    t0 = time.time(); _ = Lg(jnp.array([tx, ty])); _ = Jf(jnp.array([tx, ty]))
    print(f"[neyman] shard {SHARD}/{NSHARDS} warm {time.time()-t0:.0f}s", flush=True)

    X = np.linspace(tx * (1 - HWF), tx * (1 + HWF), NGRID)             # LINEAR, actual metres
    Y = np.linspace(ty * (1 - HWF), ty * (1 + HWF), NGRID)
    Larr = np.zeros((NGRID, NGRID)); Gx = np.zeros((NGRID, NGRID)); Gy = np.zeros((NGRID, NGRID))
    Fxx = np.zeros((NGRID, NGRID)); Fxy = np.zeros((NGRID, NGRID)); Fyy = np.zeros((NGRID, NGRID))
    Wn = np.asarray(Winv)
    IJ = [(i, j) for i in range(NGRID) for j in range(NGRID)]          # index [i=Ls, j=La]
    mine = [idx for idx in range(len(IJ)) if idx % NSHARDS == SHARD]   # this shard's round-robin points
    t0 = time.time()
    for c, idx in enumerate(mine):
        i, j = IJ[idx]; v = jnp.array([X[i], Y[j]])
        lv, g = Lg(v); J = np.asarray(Jf(v))                          # J:(NS,2)
        F = 2.0 * np.einsum("np,nq,n->pq", J, J, Wn)                  # Fisher / GN matrix
        Larr[i, j] = float(lv); Gx[i, j] = float(g[0]); Gy[i, j] = float(g[1])
        Fxx[i, j] = F[0, 0]; Fxy[i, j] = 0.5 * (F[0, 1] + F[1, 0]); Fyy[i, j] = F[1, 1]
        if c % 50 == 0: print(f"[neyman] shard{SHARD} {c}/{len(mine)} ({time.time()-t0:.0f}s)", flush=True)
    base = str(paths.data_dir(FIGURE) / "landscape2d_neyman")
    out = f"{base}.npz" if NSHARDS == 1 else f"{base}_part{SHARD}.npz"
    np.savez(out, X=X, Y=Y, L=Larr, Gx=Gx, Gy=Gy, Fxx=Fxx, Fxy=Fxy, Fyy=Fyy,
             truth_x=tx, truth_y=ty, wl=C.REP_WL)
    print(f"[neyman] shard {SHARD} saved {len(mine)} pts ({time.time()-t0:.0f}s)", flush=True)


# GUARDED: without this, IMPORTING this module ran the full scan. It is invoked by
# `fig_calib_loss_geometry` as `subprocess.run([sys.executable, compute_2d_neyman.py])`,
# which is unchanged -- but an import from any tool started a 625-point Neyman scan, which
# is what happened to a dependency check and had to be killed mid-run.
if __name__ == "__main__":
    main()
