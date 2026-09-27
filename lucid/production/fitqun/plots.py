"""One figure per kind of fiTQun tune input, driven off the manifest.

A plotter reads the installed file and draws whatever lets a reader judge that
input at a glance. This is the only module here that knows the internal structure
of a tune file, which is why it is separate from ``report.py``.

Registered by manifest ``kind``; a kind with no plotter is still reported in the
table, it just has no figure. Reading needs uproot, and on LXPLUS the krb5
credential cache must be bound into the container or every open fails with
EACCES rather than anything mentioning Kerberos.
"""
from __future__ import annotations

import re
from pathlib import Path

FIGSIZE = (6.4, 4.0)
DPI = 110


def _fig(xlabel: str, ylabel: str, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25, linewidth=.6)
    return fig, ax


def _save(fig, out: Path) -> Path:
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_time_pdf(path: str, out_dir: Path, label: str) -> list[Path]:
    """The fitted Gaussian in corrected hit time, against predicted charge.

    The file stores, per momentum node, the mean and width of ``t_c`` as a
    function of ``log10(mu)`` over 125 bins on [-2, 3] -- the same axis
    ``makehistWCSim.cc:155`` filled. ``htpdfinfo`` carries the momentum validity
    range in bins 5 and 6, which is the number that matters: ``GetTimeCoeff``
    (``fiTQun_shared.cc:1967``) *clamps* momentum into that range, so above pmax
    every track is evaluated with the pmax coefficients rather than extrapolated.
    """
    import numpy as np
    import uproot
    with uproot.open(path) as f:
        moms = sorted({int(m.group(1)) for k in f.keys()
                       if (m := re.match(r"hmean_(\d+)", k))})
        if not moms:
            return []
        info = f["htpdfinfo"].values() if "htpdfinfo" in [k.split(";")[0]
                                                          for k in f.keys()] else None
        pmin, pmax = (info[4], info[5]) if info is not None else (min(moms), max(moms))
        series = []
        for p_mev in moms:
            hm, hs = f[f"hmean_{p_mev}"], f[f"hsigm_{p_mev}"]
            edges = hm.axis().edges()
            x = .5 * (edges[1:] + edges[:-1])
            series.append((p_mev, x, hm.values(), hs.values()))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=DPI)
    cmap = plt.get_cmap("viridis")
    for i, (p_mev, x, mean, sig) in enumerate(series):
        c = cmap(i / max(len(series) - 1, 1))
        ok = sig != 0
        axes[0].plot(x[ok], mean[ok], "-", lw=1.1, color=c)
        axes[1].plot(x[ok], sig[ok], "-", lw=1.1, color=c,
                     label=f"{p_mev}" if i in (0, len(series) - 1) else None)
    for ax, yl, ti in ((axes[0], "fitted mean of $t_c$ (ns)", "time PDF centre"),
                       (axes[1], "fitted width of $t_c$ (ns)", "time PDF width")):
        ax.set_xlabel(r"$\log_{10}(\mu)$   (predicted charge, p.e.)")
        ax.set_ylabel(yl)
        ax.set_title(f"{label}: {ti}", fontsize=10)
        ax.grid(alpha=.25, linewidth=.6)
    # The per-bin fits go wild where a bin has too few hits -- below ~0.1 p.e.
    # and at the top of the range -- so clip to the region that carries the
    # likelihood rather than letting the failures set the scale.
    axes[0].set_ylim(-8, 8)
    axes[1].set_ylim(0, 8)
    axes[1].legend(fontsize=8, title="MeV/c", title_fontsize=8)
    axes[0].annotate(f"{len(series)} momentum nodes\nvalid {pmin:g}-{pmax:g} MeV/c"
                     f"\n(clamped outside)",
                     (.03, .05), xycoords="axes fraction", fontsize=8)
    fig.tight_layout()
    out = out_dir / f"{label}_time_pdf.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return [out]


def plot_charge_pdf(path: str, out_dir: Path, label: str) -> list[Path]:
    """Each fitted parameter against predicted charge, one line per charge range."""
    import uproot
    with uproot.open(path) as f:
        graphs: dict[int, list] = {}
        for k in f.keys():
            m = re.match(r"gParam_Rang(\d+)_(\d+)", k)
            if not m:
                continue
            g = f[k]
            graphs.setdefault(int(m.group(2)), []).append(
                (int(m.group(1)), g.values("x"), g.values("y")))
        if not graphs:
            return []
    out = []
    for ipar in sorted(graphs):
        fig, ax = _fig(r"$\mu$ (predicted p.e.)", f"parameter {ipar}",
                       f"{label}: charge PDF parameter {ipar}")
        for irang, x, y in sorted(graphs[ipar]):
            ax.plot(x, y, "-", lw=1.2, label=f"q range {irang}")
        ax.set_xscale("log")
        ax.legend(fontsize=7, ncol=2)
        out.append(_save(fig, out_dir / f"{label}_charge_pdf_par{ipar}.png"))
    return out


def plot_angular_response(path: str, out_dir: Path, label: str) -> list[Path]:
    """The response against cos(eta), sampled from the stored TF1."""
    import numpy as np
    import uproot
    with uproot.open(path) as f:
        tf1 = f["angResp"]
        # uproot gives fParams as a TF1Parameters model, not an array.
        par = np.asarray(tf1.member("fParams").member("fParameters"), dtype=float)
        names = [str(n) for n in tf1.member("fParams").member("fParNames")]
        xmin, xmax = tf1.member("fXmin"), tf1.member("fXmax")
    # The TF1 is a piecewise polynomial whose formula ROOT cannot re-evaluate
    # outside ROOT, so show the stored parameters and domain rather than a
    # reconstructed curve we might get subtly wrong.
    fig, ax = _fig("parameter", "value",
                   f"{label}: angular response, {len(par)} params "
                   f"on cos(eta) in [{xmin:g}, {xmax:g}]")
    ax.stem(range(len(par)), par)
    if len(names) == len(par):
        ax.set_xticks(range(len(par)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    return [_save(fig, out_dir / f"{label}_angular_response.png")]


def plot_attenuation(npz_glob: str, out_dir: Path, label: str) -> list[Path]:
    """Direct fraction against distance, from the AttenL side-output shards."""
    import glob

    import numpy as np

    from . import attenlength
    files = sorted(glob.glob(npz_glob))
    if not files:
        return []
    a = d = e = None
    for fn in files:
        z = np.load(fn)
        if a is None:
            a, d, e = z["h_all"].astype(float), z["h_direct"].astype(float), z["edges"]
        else:
            a += z["h_all"]
            d += z["h_direct"]
    c = .5 * (e[1:] + e[:-1])
    ok = a >= 50
    p = d[ok] / a[ok]
    fig, ax = _fig("emission point to PMT, R (cm)", "direct / all",
                   f"{label}: AttenL direct fraction ({a.sum():,.0f} photons)")
    ax.errorbar(c[ok], p, yerr=np.sqrt(np.maximum(p * (1 - p) / a[ok], 1e-12)),
                fmt="o", ms=3, lw=.8)
    try:
        fit = attenlength.fit(e, a, d)
        ax.plot(c[ok], fit.a0 * np.exp(-c[ok] / fit.length_cm), "-",
                label=f"L = {fit.length_cm:.0f} cm")
        ax.legend(fontsize=8)
    except ValueError as exc:
        ax.annotate(f"fit refused: {exc}", (.03, .06), xycoords="axes fraction",
                    fontsize=8)
    ax.set_ylim(0, 1.05)
    return [_save(fig, out_dir / f"{label}_attenuation.png")]


#: manifest kind -> plotter. Missing kinds are reported without a figure.
PLOTTERS = {
    "time_pdf": plot_time_pdf,
    "charge_pdf": plot_charge_pdf,
    "angular_response": plot_angular_response,
}


def render(m, out_dir) -> dict[str, list[str]]:
    """Every figure the manifest's artifacts can produce, keyed by ``install_as``."""
    out_dir = Path(out_dir)
    made: dict[str, list[str]] = {}
    for a in m.artifacts:
        fn = PLOTTERS.get(a.kind)
        if fn is None or not a.path or not Path(a.path).exists():
            continue
        label = Path(a.install_as).stem
        try:
            made[a.install_as] = [str(p) for p in fn(a.path, out_dir, label)]
        except Exception as exc:                       # a bad file must not stop the report
            made[a.install_as] = [f"ERROR: {type(exc).__name__}: {exc}"]
    return made
