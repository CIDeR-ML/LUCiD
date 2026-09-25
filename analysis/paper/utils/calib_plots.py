"""Rendering for the two calibration figures. Plot-only: every function reads saved ``.npz``
and writes a PDF+PNG pair. No simulation, no fitting. The styling comments record deliberate
choices; keep them in step with any change to the plots.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import matplotlib.cm as cm             # noqa: E402
import matplotlib.patheffects as pe    # noqa: E402

from analysis.paper.utils import damping   # noqa: E402  (numpy only — keeps this module plot-only)
import matplotlib.ticker as mticker    # noqa: E402
from matplotlib.lines import Line2D    # noqa: E402


def _get_cmap(name):
    """matplotlib >= 3.9 removed ``cm.get_cmap``; both spellings return the same colormap."""
    try:
        return cm.get_cmap(name)
    except AttributeError:
        return plt.get_cmap(name)


def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(str(out_dir), f"{stem}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    return png


# ==========================================================================================
# FIGURE B -- joint 19-parameter calibration convergence
# ==========================================================================================
def convergence(data_dir, out_dir, polyak_window=150, xzoom=100, stem="calib_convergence"):
    """2x3 convergence figure: per-wavelength optics + reflection + per-PMT gains.

    Publication style: thick lines, large fonts, descriptive y-axis labels (no panel titles,
    no suptitle). Fast panels zoomed to the first ``xzoom`` iterations; the diffuse panel keeps
    the full range, because the diffuse pair is the only reason the run goes to 600.
    colour = wavelength (optics) / wall-sensor (reflectivity); LINE STYLE = seed; dotted = truth.
    """
    plt.rcParams.update({"font.size": 26, "axes.labelsize": 30, "xtick.labelsize": 23,
                         "ytick.labelsize": 23, "legend.fontsize": 21, "lines.linewidth": 4.2,
                         "axes.linewidth": 2.0, "mathtext.default": "regular"})
    files = sorted(glob.glob(os.path.join(str(data_dir), "crb_part_*.npz")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not files:
        raise FileNotFoundError(f"no crb_part_*.npz in {data_dir} — run the fit first")
    fs = [np.load(f) for f in files]
    truth = np.asarray(fs[0]["truth"]); raw = [np.asarray(f["hist"]) for f in fs]
    T = raw[0].shape[0]; steps = np.arange(T); POLYW = polyak_window
    FRAC = POLYW / T      # window as a FRACTION of progress -> grows continuously, no saturation kink

    def polyak(y):
        """Proportional-window running mean: average the last FRAC of the trajectory so far.
        A fixed window saturates at t=POLYW and the slope changes there (a visible kink on a
        fast-decaying signal). Growing the window in proportion to t removes that transition and
        reaches exactly POLYW at the final step, matching the Polyak readout."""
        out = np.empty_like(y)
        for t in range(len(y)):
            lo = int(t * (1.0 - FRAC))
            out[t] = y[lo:t + 1].mean(0)
        return out

    H = [polyak(h) for h in raw]; nseed = len(H)
    SEED_LS = ["-", "--", "-.", (0, (5, 2)), (0, (1, 1))][:max(nseed, 1)]
    wls = list(np.asarray(fs[0]["wls"])) if "wls" in fs[0] else [337, 375, 405, 445, 473]
    wlmap = _get_cmap("turbo"); wlcol = [wlmap(0.08 + 0.84 * i / 4) for i in range(5)]
    XZOOM = xzoom                                  # fast panels: show only where the action is
    LWT, LWTR = 4.6, 3.0                           # trajectory / truth line widths

    fig, ax = plt.subplots(2, 3, figsize=(24, 15))

    def style(a):
        a.set_xlabel("iteration"); a.grid(alpha=.28, which="both", lw=1.3)
        a.tick_params(width=2.0, length=8)

    # --- optics: colour = wavelength, line style = seed ---
    opt = [("Scattering length  $L_s$  [m]", [0, 3, 6, 9, 12], True),
           ("Absorption length  $L_a$  [m]", [1, 4, 7, 10, 13], True),
           ("Quantum efficiency  QE", [2, 5, 8, 11, 14], False)]
    for a, (ylab, idx, logy) in zip(ax[0], opt):
        for c, j in zip(wlcol, idx):
            for s, h in enumerate(H):
                a.plot(steps, h[:, j], color=c, ls=SEED_LS[s], lw=LWT, alpha=0.9)
            a.axhline(truth[j], color=c, ls=":", lw=LWTR, alpha=0.95)
        if logy:
            a.set_yscale("log")
        a.set_ylabel(ylab); style(a); a.set_xlim(-0.03 * XZOOM, XZOOM)
    ax[0, 0].legend(handles=[Line2D([0], [0], color=wlcol[i], lw=5, label=f"{int(wls[i])} nm")
                             for i in range(5)],
                    loc="upper right", title="colour = λ", framealpha=0.92)
    ax[0, 1].legend(handles=[Line2D([0], [0], color="k", ls=SEED_LS[s], lw=3.4, label=f"seed {s}")
                             for s in range(nseed)]
                            + [Line2D([0], [0], color="k", ls=":", lw=2.6, label="truth")],
                    loc="lower right", framealpha=0.92)

    # --- reflectivity combined; colour = wall/sensor, line style = seed ---
    # Wall/sensor colours are chosen AGAINST the top row's `turbo` wavelength ramp, not in isolation.
    # turbo runs dark-blue -> blue -> cyan -> green -> yellow -> orange -> red, so the previous
    # "#1f77b4 / #2ca02c" (tab blue / tab green) sat INSIDE that gamut: the same two hues carried two
    # different meanings in one figure. PURPLE is the one family turbo never visits, so the wall curve
    # gets a hue that cannot be read as a wavelength. Pairing it with a dark ochre gives the
    # ColorBrewer PuOr endpoints -- a standard colourblind-safe diverging pair, dark and muted against
    # turbo's high-saturation neon (so the row reads as its own family), and separated in LIGHTNESS
    # (rel. luminance ~0.24 vs ~0.35) so the two survive greyscale printing.
    wcol, scol = "#542788", "#B35806"

    def refl_panel(a, jw, js, lw_lab, ls_lab, ylab):
        for j, col in [(jw, wcol), (js, scol)]:
            for s, h in enumerate(H):
                a.plot(steps, h[:, j], color=col, ls=SEED_LS[s], lw=LWT, alpha=0.92)
            a.axhline(truth[j], color=col, ls=":", lw=LWTR, alpha=0.9)
        a.set_yscale("log"); a.set_ylabel(ylab); style(a)
        # These panels span under two decades, so the default log formatter either repeats "10^-1"
        # or (with ScalarFormatter) labels every minor tick and the text collides. Label a sparse
        # 1-2-3-5 progression only (3 included so the truth values ~0.0225/0.025/0.0275 are
        # readable), keep the rest as unlabelled minor ticks for the grid.
        a.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0), numticks=15))
        a.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1),
                                                     numticks=15))
        a.yaxis.set_minor_formatter(mticker.NullFormatter())
        a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.legend(handles=[Line2D([0], [0], color=wcol, lw=5, label=lw_lab),
                          Line2D([0], [0], color=scol, lw=5, label=ls_lab)],
                 loc="best", framealpha=0.92)

    refl_panel(ax[1, 0], 15, 17, r"wall  $R_w^{\rm spec}$", r"sensor  $R_s^{\rm spec}$",
               "Specular reflectivity")
    ax[1, 0].set_xlim(-0.03 * XZOOM, XZOOM)   # specular settles by ~50; diffuse keeps full range
    refl_panel(ax[1, 1], 16, 18, r"wall  $R_w^{\rm diff}$", r"sensor  $R_s^{\rm diff}$",
               "Diffuse reflectivity")

    # --- gains ---
    kh = np.concatenate([np.asarray(f["khat"]) for f in fs])
    kt = np.concatenate([np.asarray(f["truth_k"]) for f in fs])
    g = ax[1, 2]; hb = g.hexbin(kt, kh, gridsize=52, bins="log", cmap="viridis", mincnt=1)
    lim = [min(kt.min(), kh.min()) * 0.99, max(kt.max(), kh.max()) * 1.01]
    g.plot(lim, lim, "r--", lw=3.6, label="recovered = true"); g.set_xlim(lim); g.set_ylim(lim)
    g.set_xlabel("true per-PMT gain  $k$"); g.set_ylabel(r"recovered gain  $\hat k$")
    g.text(0.04, 0.90, f"RMS {(kh/kt-1).std()*100:.2f}%", transform=g.transAxes, fontsize=24,
           bbox=dict(fc="white", ec="0.5", alpha=0.9))
    g.legend(loc="lower right", framealpha=0.92); g.tick_params(width=2.0, length=8)
    cb = fig.colorbar(hb, ax=g); cb.set_label("PMTs", fontsize=22); cb.ax.tick_params(labelsize=18)
    fig.tight_layout()
    out = _save(fig, out_dir, stem)
    plt.close(fig)
    print(f"wrote {out}  ({nseed} seeds, {T} iterations)")
    return out


# ==========================================================================================
# FIGURE A -- 2D loss geometry, raw descent vs Fisher-preconditioned
# ==========================================================================================
def loss_geometry(data_dir, out_dir, lam=0.01, mu=0.1, stem="calib_loss_geometry"):
    """Neyman + Fisher 2D loss-geometry figure. Data = ``landscape2d_neyman.npz``
    (single laser, scalar_mix, Neyman chi^2 loss, AD gradient, per-point Fisher/GN matrix
    F = 2 sum J^T J / Q).

     (left)  raw gradient descent  -grad L
     (right) Fisher-preconditioned -F^-1 grad L   (Gauss-Newton step, per grid point).
    """
    plt.rcParams.update({"font.size": 30, "axes.labelsize": 42, "xtick.labelsize": 34,
                         "ytick.labelsize": 34, "axes.titlesize": 40, "axes.linewidth": 2.6,
                         "mathtext.default": "regular"})
    d = np.load(os.path.join(str(data_dir), "landscape2d_neyman.npz"))
    X, Y, L, Gx, Gy = d["X"], d["Y"], d["L"], d["Gx"], d["Gy"]
    Fxx, Fxy, Fyy = d["Fxx"], d["Fxy"], d["Fyy"]
    tx, ty, wl = float(d["truth_x"]), float(d["truth_y"]), int(d["wl"])
    XX, YY = np.meshgrid(X, Y)
    LAM, MU = lam, mu     # the campaign's damping (CALIB_RECIPE, and traj_2d_neyman.py)

    def finv_g(fxx, fxy, fyy, gx, gy):                             # damped F^-1 g, per point
        # Marquardt lam*diag(F) + Levenberg mu*median(diag F)*I, then a plain solve -- IDENTICAL to the
        # step the joint fit takes, so the streamlines show the actual optimizer field.
        # Previously this eigen-floored the inverse (clip ev at 1e-3*ev_max), which is the LEGACY solver
        # that SOLVER=solve replaced. That floor engaged at 132/2601 grid points and tilted the field by a
        # median 22.4 deg (max 73.4) versus the damped step, so the plotted streamlines disagreed with the
        # GN trajectory overlaid on them -- which traj_2d_neyman.py had already switched to the damping.
        F = np.array([[fxx, fxy], [fxy, fyy]]); g = np.array([gx, gy])
        return damping.damped_step(F, g, LAM, MU)    # one definition, shared with traj_2d_neyman

    Px = np.zeros_like(Gx); Py = np.zeros_like(Gy)
    for i in range(X.size):
        for j in range(Y.size):
            Px[i, j], Py[i, j] = finv_g(Fxx[i, j], Fxy[i, j], Fyy[i, j], Gx[i, j], Gy[i, j])
    # cond of Fisher at the minimum
    i0 = int(np.argmin(np.abs(X - tx))); j0 = int(np.argmin(np.abs(Y - ty)))
    ev0 = np.linalg.eigvalsh(np.array([[Fxx[i0, j0], Fxy[i0, j0]], [Fxy[i0, j0], Fyy[i0, j0]]]))
    cond = ev0.max() / max(ev0.min(), 1e-30)

    # --- optimization trajectories (two random starts), overlaid on top of each panel ---
    _TPATH = os.path.join(str(data_dir), "traj2d_neyman.npz")
    TRAJ = np.load(_TPATH) if os.path.exists(_TPATH) else None
    TCOL = ["#ff7f0e", "#ff2ec4"]                          # start 1 = orange, start 2 = magenta

    def _trim(p, tol=1.5):                                 # drop the jitter tail clustered at the min
        dd = np.linalg.norm(p - p[-1], axis=1)
        k = int(np.argmax(dd < tol))
        return p[:k + 1] if (dd[k] < tol and k > 0) else p

    def overlay(a, trajs, label_starts):
        for c, (col, p) in enumerate(zip(TCOL, trajs)):
            p = _trim(p)
            a.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.6, zorder=7, solid_capstyle="round",
                   path_effects=[pe.Stroke(linewidth=4.6, foreground="k"), pe.Normal()])
            # a dot at every iterate -> the step DENSITY is visible: GD crawls (many dots), GN leaps (few)
            a.plot(p[:, 0], p[:, 1], "o", color=col, ms=6.5, mec="k", mew=0.8, zorder=8, ls="none")
            a.plot(p[0, 0], p[0, 1], marker="o", ms=22, color=col, mec="k", mew=2.2, zorder=9,
                   ls="none", label=(f"start {c+1}" if label_starts else None))

    Lg = np.log10(L.T - L.min() + 1e-6)
    vmin, vmax = np.percentile(Lg, 20), np.percentile(Lg, 99)   # clip: 12-decade razor -> reveal the bowl
    lv = np.linspace(vmin, vmax, 30)
    fig, ax = plt.subplots(1, 2, figsize=(21, 9.4), sharey=True, constrained_layout=True)
    panels = [(-Gx.T, -Gy.T, r"Gradient descent   $-\nabla\mathcal{L}$"),
              (-Px.T, -Py.T, r"Gauss--Newton   $-F^{-1}\nabla\mathcal{L}$")]
    for k, (a, (U, VV, ttl)) in enumerate(zip(ax, panels)):
        cf = a.contourf(XX, YY, Lg, levels=lv, cmap="viridis", vmin=vmin, vmax=vmax, extend="both")
        mag = np.hypot(U, VV) + 1e-30
        a.streamplot(X, Y, U / mag, VV / mag, color="w", density=1.3, linewidth=1.5,
                     arrowsize=1.8, zorder=2)
        if TRAJ is not None:                        # gradient descent (left), Gauss-Newton (right)
            overlay(a, [TRAJ["gd0"], TRAJ["gd1"]] if k == 0 else [TRAJ["gn0"], TRAJ["gn1"]],
                    label_starts=(k == 0))
        a.plot(tx, ty, marker="*", ms=44, color="red", mec="k", mew=1.8, zorder=9, ls="none",
               label="truth")
        a.set(xlabel=r"scattering length  $\lambda_s$  [m]", xlim=(X.min(), X.max()),
              ylim=(Y.min(), Y.max()))
        if k == 0:
            a.set_ylabel(r"absorption length  $\lambda_a$  [m]")
            a.legend(loc="lower left", framealpha=0.92, handletextpad=0.2, borderpad=0.3, fontsize=24)
        a.set_title(ttl, pad=14); a.tick_params(width=2.6, length=11)
    cb = fig.colorbar(cf, ax=ax, fraction=.046, pad=.02)
    cb.set_label(r"$\log_{10}\,\mathcal{L}$", fontsize=40)
    # Ticks on ROUND 0.2 steps rather than on the contour levels (which are a 30-point linspace between
    # two percentiles, hence labels like 4.379). Round values print exactly at "%.1f" = 2 sig figs.
    # Built by arange INSIDE [vmin, vmax] rather than by a locator: a locator also emits the round value
    # just below vmin, which lands in the "extend" arrow and overprints the first in-range label.
    _t = np.arange(np.ceil(vmin / 0.2) * 0.2, vmax + 1e-9, 0.2)
    cb.set_ticks(_t); cb.set_ticklabels([f"{v:.1f}" for v in _t])
    cb.ax.tick_params(labelsize=30)
    out = _save(fig, out_dir, stem)
    plt.close(fig)
    print(f"wrote {out}  cond(F)={cond:.1f}  wl={wl}nm  L=[{L.min():.3g},{L.max():.3g}]")
    return out
