#!/usr/bin/env python3
"""Figure: the t0 correction — photon arrival delay (t - d/c) vs distance, per energy.

Two ingredients, deliberately separated for reproducibility:

  * LINES  — the t0 parametrization that SHIPS with LUCiD
             (``data/<material>/<particle>/t0.json``), evaluated via
             ``predict_t0`` over 0..s_max(E). A plain clone reproduces these.
  * POINTS — raw PhotonSim timing (``PhotonHist_TimeDistanceNorm``), which is NOT
             shipped. ``--generate-data`` makes a minimal 5-events/energy sample
             with the bundled PhotonSim; ``--data-dir`` can instead point at a
             full production scan.

    python fig_t0_correction.py                  # lines (+ points if a sample exists)
    python fig_t0_correction.py --generate-data  # simulate the small sample, then plot
    python fig_t0_correction.py --data-dir <scan> --plot-results   # full-scan points

Energies are chosen uniformly in track length s_max(E) (~--spacing-m apart), from
--energy-min to --energy-max. Run inside the container.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

MATERIAL = 'water'
G4NAME = {'muon': 'mu-', 'electron': 'e-'}
C_MM_PER_NS = 299.792
# defaults = the paper configuration
D_ENERGY_MIN, D_ENERGY_MAX, D_SPACING_M, D_DELAY_CAP = 500.0, 10000.0, 2.0, 0.25


# ------------------------------------------------------------ shipped parametrization
def _shipped(particle):
    """(t0 coeffs, s_max_fn[mm]) — both from the LUCiD-bundled model. No raw data."""
    from lucid.utils import unpack_t0_params, unpack_siren_params
    from lucid.siren.core import build_cherenkov_context
    from lucid.siren.training.inference import SIRENPredictor
    coeffs = unpack_t0_params(particle, MATERIAL)
    scfg = unpack_siren_params(particle, MATERIAL)
    ctx = build_cherenkov_context(SIRENPredictor(scfg['siren_model_path']),
                                  dict(scfg['ray_sampling']))
    return coeffs, ctx.s_max_fn


def _delay_line(E, coeffs, smax_fn):
    """Model delay (ns) vs distance (mm) over 0..s_max(E), from predict_t0."""
    from lucid.sources.siren_rays import predict_t0
    smax = float(smax_fn(E))
    d = np.linspace(0.0, smax, 300)
    t = np.asarray(predict_t0(d, float(E), *coeffs))
    return d, t - d / C_MM_PER_NS


def select_energies(smax_fn, e_min, e_max, spacing_m):
    """Energies whose s_max steps by ~spacing_m metres across [e_min, e_max]."""
    grid = np.arange(e_min, e_max + 1, 50.0)
    out, last = [], None
    for E in grid:
        s_m = float(smax_fn(E)) / 1000.0
        if last is None or (s_m - last) >= spacing_m:
            out.append(float(E)); last = s_m
    return out


# ------------------------------------------------------------------ optional raw points
def _load_points(E, data_dir, smax_mm):
    """(d_mm, delay) from the nearest-energy sample ROOT, or None. Needs PhotonSim's reader."""
    import glob
    cands = glob.glob(f'{data_dir}/*{int(E)}MeV*.root')
    if not cands:
        return None
    sys.path.insert(0, str(_photonsim_dir() / 'tools' / 't0_correction'))
    from calculate_t0 import load_profile, MIN_D_MM
    try:
        d_mm, delay = load_profile(Path(cands[0]), smax_mm)
    except (KeyError, ValueError):
        return None
    return (d_mm, delay) if int((d_mm > MIN_D_MM).sum()) >= 4 else None


def _decimate(values, spacing):
    keep, last = [], -np.inf
    for i, v in enumerate(values):
        if abs(v - last) >= spacing:
            keep.append(i); last = v
    return np.asarray(keep, dtype=int)


# --------------------------------------------------------- minimal 5-event/energy sample
def _photonsim_dir():
    for c in (os.environ.get('LUCID_PHOTONSIM_DIR'),
              REPO_ROOT.parent / 'PhotonSim', '/opt/PhotonSim'):
        if c and (Path(c) / 'tools' / 't0_correction' / 'calculate_t0.py').exists():
            return Path(c)
    raise FileNotFoundError('PhotonSim checkout not found; set $LUCID_PHOTONSIM_DIR')


def generate_sample(particle, energies, n_events, out_dir):
    """Simulate n_events/energy with the bundled PhotonSim (bakes /output/smax so the
    timing histogram is booked). Writes <out_dir>/<particle>_<E>MeV.root."""
    import subprocess
    ps = _photonsim_dir()
    sys.path.insert(0, str(ps / 'tools' / 't0_correction'))
    from generate_energy_scan_macros import _load_smax_row, _eval_smax
    binary = ps / 'build' / 'PhotonSim'
    smax_row, _ = _load_smax_row(ps / 'data', MATERIAL, G4NAME[particle])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for E in energies:
        smax_mm = _eval_smax(smax_row, E)
        root = out_dir / f'{particle}_{int(E)}MeV.root'
        mac = out_dir / f'{particle}_{int(E)}MeV.mac'
        mac.write_text(
            f"/output/filename {root}\n/output/smax {smax_mm:.6f} mm\n/run/initialize\n"
            f"/photon/storeIndividual false\n/gun/particle {G4NAME[particle]}\n"
            f"/gun/energy {E} MeV\n/run/beamOn {n_events}\n")
        print(f'  sim {particle} {int(E)} MeV x{n_events} -> {root.name}', flush=True)
        subprocess.run([str(binary), str(mac)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------------- figure
def make_figure(particle, energies, coeffs, smax_fn, data_dir, delay_cap,
                marker_spacing_ns, out):
    gev = np.array(energies) / 1000.0
    vmax = max(gev.max(), D_ENERGY_MAX / 1000.0)
    norm = LogNorm(vmin=gev.min(), vmax=vmax)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for E in energies:
        c = cmap(norm(E / 1000.0))
        d, delay = _delay_line(E, coeffs, smax_fn)          # shipped parametrization
        ax.plot(d / 1000.0, delay, '-', color=c, lw=1.8, alpha=0.8, zorder=2)
        if data_dir:                                        # optional raw points
            pts = _load_points(E, data_dir, float(smax_fn(E)))
            if pts is not None:
                dp, dyp = pts
                idx = _decimate(dyp, marker_spacing_ns)[1:]
                ax.plot(dp[idx] / 1000.0, dyp[idx], 'o', ms=3.0, alpha=0.95,
                        color=c, zorder=3)
    ax.axhline(0, color='grey', lw=0.4)
    ax.set_xlim(0, None); ax.set_ylim(0, delay_cap)
    ax.set_xlabel('Distance from Origin (m)')
    ax.set_ylabel(r'$t_0$ Correction (ns)')

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax); cbar.set_label('Energy (GeV)')
    cand = np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20])
    ticks = cand[(cand >= gev.min()) & (cand <= vmax)]
    if ticks.size:
        cbar.set_ticks(ticks); cbar.set_ticklabels([f'{t:g}' for t in ticks])

    fig.tight_layout()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / f'{particle}_t0_correction'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {base}.pdf (+png)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true',
                    help='simulate the minimal n-event/energy sample first')
    ap.add_argument('--plot-results', action='store_true', help='(default action)')
    ap.add_argument('--particles', default='muon,electron')
    ap.add_argument('--data-dir', default=None,
                    help='dir of PhotonSim ROOTs for the points (default: the generated sample)')
    ap.add_argument('--events', type=int, default=20, help='events/energy for --generate-data')
    ap.add_argument('--energy-min', type=float, default=D_ENERGY_MIN)
    ap.add_argument('--energy-max', type=float, default=D_ENERGY_MAX)
    ap.add_argument('--spacing-m', type=float, default=D_SPACING_M)
    ap.add_argument('--delay-cap', type=float, default=D_DELAY_CAP)
    ap.add_argument('--marker-spacing-ns', type=float, default=0.01)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else paths.figure_dir()

    for particle in a.particles.split(','):
        particle = particle.strip()
        coeffs, smax_fn = _shipped(particle)
        energies = select_energies(smax_fn, a.energy_min, a.energy_max, a.spacing_m)
        print(f'{particle}: {len(energies)} energies '
              f'({energies[0]:.0f}-{energies[-1]:.0f} MeV), ~{a.spacing_m} m spacing')
        sample_dir = paths.data_dir('t0_correction', 'local') / particle
        if a.generate_data:
            generate_sample(particle, energies, a.events, sample_dir)
        data_dir = a.data_dir or (str(sample_dir) if sample_dir.exists() else None)
        make_figure(particle, energies, coeffs, smax_fn, data_dir,
                    a.delay_cap, a.marker_spacing_ns, out)


if __name__ == '__main__':
    main()
