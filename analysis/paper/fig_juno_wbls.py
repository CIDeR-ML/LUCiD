#!/usr/bin/env python3
"""Figures: JUNO-like WbLS sphere — Cherenkov vs scintillation, and a scintillation parameter's loss.

1. ``sphere_wbls_cher_scint_fraction`` — the per-sensor signed asymmetry
       a = (Q_cher - Q_scint) / (Q_cher + Q_scint)   in [-1, +1]
   on a diverging blue->white->red scale (+1 pure Cherenkov, -1 pure scintillation, white where
   they balance). Two plotly disc scenes side by side (data left, prediction right), shared colorbar.

     * DATA       : real PhotonSim ROOT photons (Cherenkov + scintillation expanded from the dE/dx
                    segments) injected into the source='photons' simulator, one emission process at a time.
     * PREDICTION : forward sim of the same track, Cherenkov-only and scintillation-only, combined.

2. ``sphere_wbls_<param>_loss_<mode>`` — the differentiability of a scintillation parameter
   (S or kB). The forward simulator is re-run at every scan point with a different value of the
   chosen parameter, and the loss gradient dNLL/d(param) is taken by AUTOMATIC DIFFERENTIATION
   through the simulator (not by rescaling a precomputed prediction). The parameter enters as a
   runtime DetectorParams field, so one jax.value_and_grad of the loss is JIT-compiled once and
   evaluated at each scan point.

   Two scannable parameters (--param):
     * S  : scintillation light yield (ph/MeV)         — dL/dx numerator
     * kB : Birks quenching constant (mm/keV)          — dL/dx denominator (Chou/Birks)
       dL/dx = S·(dE/dx) / (1 + kB·(dE/dx) + C·(dE/dx)²)

   Two ways to define the observed event n_i (--data-mode):
     * injected : the DATA-like event — the injected charges of figure 1 (its cache). The
                  gradient's zero crossing recovers the parameter up to data-vs-model fidelity.
     * forward  : a self-consistent pseudo-data event — the SMOOTH forward prediction run once at
                  the nominal value p0 (same sim + key). Since data == model at p0, this is a
                  PERFECT closure: NLL is minimised and the gradient is zero at p0.

   Loss = per-sensor Poisson NLL of the total predicted charge vs the fixed observed event:
       mu_i(p) = forward_sim(track, p)_i   (both emission processes)
       NLL(p)  = sum_i [ mu_i - n_i * ln(mu_i) ]

The expensive forward sims are cached (the per-process charges, then the scan), so the figures
can be re-drawn with --plot-results alone.

    python analysis/paper/fig_juno_wbls.py                 # simulate + cache + plot (S, injected)
    python analysis/paper/fig_juno_wbls.py --generate-data
    python analysis/paper/fig_juno_wbls.py --plot-results  # re-draw from the caches
    # the Birks-constant closure (smooth pseudo-data, perfect recovery):
    python analysis/paper/fig_juno_wbls.py --param kB --data-mode forward

Run inside the container (needs the [viz] extra and a GPU for the sims); see
docs/getting-started/install.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths, provenance      # noqa: E402

CONFIG = REPO_ROOT / 'config'
FIGURE = 'juno_cher_scint'
DET = 'JUNO_wbls'
GEOM = str(CONFIG / f'{DET}_geom_config.json')
PHYS = str(CONFIG / f'{DET}_physics_config.json')
DEFAULT_ROOT = str(REPO_ROOT / 'data' / 'water' / 'muon' / '1000MeV_5events.root')
ENTRY = 0
N_PHOTONS = 2_000_000
ENERGY = 1000.0
SEED = 6
EPS = 1e-6
# The track (both figures) points at an angle, so the ring shows on the sphere.
THETA = float(np.pi / 4)
PHI = float(np.pi / 6)
FRACTION_REL_THRESHOLD = 0.02
DIVERGING = [[0.0, '#2166ac'], [0.5, '#ffffff'], [1.0, '#b2182b']]
CB = dict(colorbar_label='', colorbar_thickness=36, colorbar_len=0.32,
          colorbar_tickfont_size=40, colorbar_tickfont_family='serif', colorbar_x=0.86)
EYE = 0.95                 # camera distance (smaller = the sphere fills more of its scene)
PANEL_GAP = 0.0            # paper fraction between the two scenes
CBAR_GUTTER = 0.07         # paper fraction kept free for the colorbar on the right

# Per-parameter presentation + default scan window. `field` is the ScintillationParams attribute;
# kB's leverage on total light is weak, so its data-like crossing sits above 1.5 p0 -> wider window.
PARAMS = {
    'S':  dict(field='S',  xlabel=r'Scintillation yield  $S / S_0$',
               glabel=r'$d\mathcal{L}/dS$',   hat=r'$\hat{S}=%.3f\,S_0$',
               vfmt='%8.1f', rlabel='S0',  lo=0.5, hi=1.5),
    'kB': dict(field='kB', xlabel=r'Birks constant  $k_B / k_{B,0}$',
               glabel=r'$d\mathcal{L}/dk_B$', hat=r'$\hat{k}_B=%.3f\,k_{B,0}$',
               vfmt='%.6g', rlabel='kB0', lo=0.5, hi=3.5),
}


def _cache_file():
    return paths.data_dir(FIGURE, 'local') / 'charges.npz'


def _scan_file(param, mode):
    return paths.data_dir(f'juno_{param}_loss_{mode}', 'local') / 'scan.npz'


# ---- figure 1: the per-process charges --------------------------------------------------
def _data_asymmetry(root, entry, theta, phi):
    """Inject real ROOT photons (per process), turned about the recorded primary vertex onto
    (theta, phi) -> per-sensor (Q_cher, Q_scint)."""
    import jax
    from lucid.simulation import setup_event_simulator
    from lucid.sources.data_event import place_photons
    from lucid.sources.root_reader import read_event_data_from_photonsim, read_primary
    from lucid.sources.scintillation_photons import scintillation_medium_params
    from lucid.utils import rotation_onto, spherical_to_cartesian
    key = jax.random.PRNGKey(SEED)
    sim = setup_event_simulator(GEOM, N_PHOTONS, source='photons', detection='expected',
                                temperature=0.0, K=6, physics_config=PHYS,
                                default_detector_params=True)
    medium_params = scintillation_medium_params(sim.default_detector_params, sim.det_geom.medium)
    rotation = rotation_onto(np.array([0.0, 0.0, 1.0]), spherical_to_cartesian(theta, phi, np), np)
    primary_vertex, _ = read_primary(root, entry)

    def charges_for(procs):
        pd = place_photons(read_event_data_from_photonsim(
            root, entry, emission_processes=procs, medium_params=medium_params,
            rng=np.random.default_rng(SEED)),
            primary_vertex=primary_vertex, rotation=rotation, vertex=primary_vertex)
        q, _ = jax.lax.stop_gradient(sim(key, pd))
        return np.asarray(q)

    qc, qs = charges_for(('cherenkov',)), charges_for(('scintillation',))
    print(f'DATA  Q: cher={qc.sum():.1f} scint={qs.sum():.1f}', flush=True)
    return qc, qs


def _pred_asymmetry(theta, phi):
    """Forward sim, one process at a time -> per-sensor (Q_cher, Q_scint)."""
    import jax
    import jax.numpy as jnp
    from lucid.detector_params import ParticleParams
    from lucid.simulation import setup_event_simulator
    key = jax.random.PRNGKey(SEED)
    track = ParticleParams(
        energy=jnp.array(ENERGY, dtype=jnp.float32),
        position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        theta=jnp.array(theta, dtype=jnp.float32), phi=jnp.array(phi, dtype=jnp.float32),
        t0=jnp.array(0.0, dtype=jnp.float32))

    def charges_for(procs):
        sim = setup_event_simulator(GEOM, 1_000_000, source='particle', hits='first_arrival',
                                    temperature=0.0, K=6, physics_config=PHYS,
                                    default_detector_params=True, emission_processes=procs)
        q, _ = jax.lax.stop_gradient(sim(track, key))
        return np.asarray(q)

    qc, qs = charges_for(('cherenkov',)), charges_for(('scintillation',))
    print(f'PRED  Q: cher={qc.sum():.1f} scint={qs.sum():.1f}', flush=True)
    return qc, qs


def generate_charges(root, entry):
    qc_d, qs_d = _data_asymmetry(root, entry, THETA, PHI)
    qc_p, qs_p = _pred_asymmetry(THETA, PHI)
    cf = _cache_file(); cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cf, qc_d=qc_d, qs_d=qs_d, qc_p=qc_p, qs_p=qs_p)
    print(f'cached charges -> {cf}')


# ---- figure 2: the loss scan -------------------------------------------------------------
def generate_scan(param, mode, nphot, npoints, srel_lo, srel_hi):
    import jax
    import jax.numpy as jnp
    from lucid.detector_params import ParticleParams, load_physics_config
    from lucid.simulation import setup_event_simulator
    spec = PARAMS[param]; field = spec['field']

    # Forward prediction must emit BOTH processes so the total charge feels the scintillation
    # yield/quenching.
    sim = setup_event_simulator(GEOM, nphot, source='particle', hits='first_arrival',
                                temperature=0.0, K=6, physics_config=PHYS,
                                default_detector_params=False,
                                emission_processes=('cherenkov', 'scintillation'))
    dp0, _, _ = load_physics_config(PHYS)
    p0 = float(getattr(dp0.scintillation, field))
    key = jax.random.PRNGKey(SEED)
    track = ParticleParams(energy=jnp.array(ENERGY, jnp.float32), position=jnp.zeros(3, jnp.float32),
                           theta=jnp.array(THETA, jnp.float32), phi=jnp.array(PHI, jnp.float32),
                           t0=jnp.array(0.0, jnp.float32))

    def charges_at(val):
        dp = dp0._replace(scintillation=dp0.scintillation._replace(**{field: jnp.asarray(val, jnp.float32)}))
        q, _ = sim(track, dp, key)
        return q

    if mode == 'forward':
        # self-consistent pseudo-data: the SMOOTH forward at p0 (same sim + key -> exact closure)
        q0 = np.asarray(jax.lax.stop_gradient(charges_at(p0)))
        mask = q0 > 0.0
        midx = jnp.asarray(np.where(mask)[0]); n = jnp.asarray(q0[mask])
        print(f'closure: {int(mask.sum())} active sensors; pseudo-data = forward({field}0)', flush=True)
    else:  # injected
        z = np.load(_cache_file())
        mask = (z['qc_p'] + z['qs_p']) > 0.0            # predicted-active sensors
        midx = jnp.asarray(np.where(mask)[0]); n = jnp.asarray((z['qc_d'] + z['qs_d'])[mask])
        print(f'injected data: likelihood over {int(mask.sum())} sensors', flush=True)
    print(f'{field}0 = {p0:.6g}; NPHOT={nphot}; window [{srel_lo:.3g}, {srel_hi:.3g}] {field}0', flush=True)

    def nll(val):
        dp = dp0._replace(scintillation=dp0.scintillation._replace(**{field: val}))
        charges, _ = sim(track, dp, key)                # re-simulate prediction at val
        mu = jnp.clip(charges[midx], EPS, None)
        return jnp.sum(mu - n * jnp.log(mu))

    vg = jax.jit(jax.value_and_grad(nll))               # JIT once
    scan = np.linspace(srel_lo * p0, srel_hi * p0, npoints)
    losses = np.empty(npoints); grads = np.empty(npoints)
    for i, val in enumerate(scan):
        l, g = vg(jnp.asarray(val, jnp.float32))
        losses[i] = float(l); grads[i] = float(g)
        print(('  ' + field + '=' + spec['vfmt'] + ' (%4.2f ' + spec['rlabel'] + ')  NLL=%.5e  '
               'd' + field + '=%+.5e') % (val, val / p0, losses[i], grads[i]), flush=True)

    sf = _scan_file(param, mode); sf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sf, val=scan, p0=p0, loss=losses, grad=grads, param=param, mode=mode)
    print(f'cached scan -> {sf}')


# ---- plotting ---------------------------------------------------------------------------
def _panel_fig(detector, qc, qs, show_colorbar):
    from lucid.visualization import event_display_3d
    total = qc + qs
    qmax = float(total.max())
    lit = np.where(total > FRACTION_REL_THRESHOLD * qmax)[0]
    asym = (qc[lit] - qs[lit]) / total[lit]
    print(f'  asym on {len(lit)} sensors, range [{asym.min():.2f}, {asym.max():.2f}]')
    cb = dict(CB, colorbar_x=0.985, colorbar_len=0.55)
    return event_display_3d(
        detector, lit, asym, np.zeros(len(lit)), show_all_sensors=True, log_scale=False,
        show_colorbar=show_colorbar, dark_theme=False, plot_time=False, colorscale=DIVERGING,
        surface_color='lightgray', inactive_color='lightgray', inactive_opacity=0.15,
        cmin=-1.0, cmax=1.0, colorbar_tickvals=[-1.0, 0.0, 1.0],
        colorbar_ticktext=['-1', '0', '1'], return_fig=True, **cb)


def plot_fraction(out):
    from plotly.subplots import make_subplots
    from lucid.geometry import generate_detector
    cf = _cache_file()
    if not cf.exists():
        print(f'[skip] no charge cache at {cf} — run --generate-data first'); return
    z = np.load(cf)
    detector = generate_detector(GEOM)
    eye_d = dict(x=EYE, y=EYE, z=EYE)
    f_data = _panel_fig(detector, z['qc_d'], z['qs_d'], show_colorbar=False)
    f_pred = _panel_fig(detector, z['qc_p'], z['qs_p'], show_colorbar=True)
    sub = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        horizontal_spacing=0.0)
    for tr in f_data.data:
        tr.update(scene='scene'); sub.add_trace(tr, row=1, col=1)
    for tr in f_pred.data:
        tr.update(scene='scene2'); sub.add_trace(tr, row=1, col=2)
    sd = f_data.layout.scene.to_plotly_json(); sd.pop('domain', None)
    sp = f_pred.layout.scene.to_plotly_json(); sp.pop('domain', None)
    sub.layout.scene.update(sd); sub.layout.scene2.update(sp)
    half = (1.0 - CBAR_GUTTER - PANEL_GAP) / 2.0
    sub.layout.scene.update(domain=dict(x=[0.0, half], y=[0.0, 1.0]), camera=dict(eye=eye_d))
    sub.layout.scene2.update(domain=dict(x=[half + PANEL_GAP, 2 * half + PANEL_GAP],
                                         y=[0.0, 1.0]), camera=dict(eye=eye_d))
    sub.update_layout(width=2200, height=1100, showlegend=False, paper_bgcolor='white',
                      plot_bgcolor='white', margin=dict(l=0, r=0, t=0, b=0))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'sphere_wbls_cher_scint_fraction'
    for ext in ('pdf', 'png'):
        sub.write_image(f'{base}.{ext}', scale=2)
    print(f'wrote {base}.pdf (+png)')


def plot_scan(param, mode, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = 13
    spec = PARAMS[param]
    sf = _scan_file(param, mode)
    if not sf.exists():
        print(f'[skip] no scan cache at {sf} — run --generate-data first'); return
    z = np.load(sf)
    val, p0, loss, grad = z['val'], float(z['p0']), z['loss'], z['grad']
    r = val / p0
    i = np.where(np.diff(np.sign(grad)) != 0)[0]
    r_min = None
    if len(i):
        j = i[0]
        r_min = r[j] - grad[j] * (r[j + 1] - r[j]) / (grad[j + 1] - grad[j])
    dloss = loss - loss.min()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1], hspace=0.08),
                                   facecolor='white')
    ax1.plot(r, dloss, '-', color='cornflowerblue', lw=2.4, zorder=1)
    ax1.plot(r, dloss, 'o', color='navy', ms=6, zorder=3)
    ax1.set_ylabel(r'$\Delta \mathcal{L}$')
    ax2.axhline(0.0, color='0.5', lw=1.0)
    ax2.plot(r, grad, '-', color='cornflowerblue', lw=2.4, zorder=1)
    ax2.plot(r, grad, 'o', color='navy', ms=6, zorder=3)
    ax2.set_ylabel(spec['glabel'])
    ax2.set_xlabel(spec['xlabel'])
    if r_min is not None:
        for ax in (ax1, ax2):
            ax.axvline(r_min, color='0.5', ls=':', lw=1.2)
        ax2.plot(r_min, 0.0, '*', color='#c0392b', ms=15, zorder=4, label=spec['hat'] % r_min)
        ax2.legend(frameon=False, loc='lower right')

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / f'sphere_wbls_{param}_loss_{mode}'
    for ext in ('pdf', 'png'):
        fig.savefig(f'{base}.{ext}', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {base}.pdf (+png)' + (f'; {param}_hat = {r_min:.4f} {param}0' if r_min else
                                        '; no zero crossing in range'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--entry', type=int, default=ENTRY)
    ap.add_argument('--param', choices=['S', 'kB'], default='S',
                    help='scintillation parameter to scan (default S)')
    ap.add_argument('--data-mode', choices=['injected', 'forward'], default='injected',
                    help="'injected' = the real data-like event (the charge cache); "
                         "'forward' = self-consistent smooth pseudo-data (perfect closure)")
    ap.add_argument('--nphot', type=int, default=1_000_000, help='rays of the scan forward')
    ap.add_argument('--npoints', type=int, default=11)
    ap.add_argument('--srel-lo', type=float, default=None,
                    help='scan lower bound in units of p0 (default: per-param)')
    ap.add_argument('--srel-hi', type=float, default=None,
                    help='scan upper bound in units of p0 (default: per-param)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    spec = PARAMS[a.param]
    lo = spec['lo'] if a.srel_lo is None else a.srel_lo
    hi = spec['hi'] if a.srel_hi is None else a.srel_hi
    out = Path(a.out) if a.out else paths.figure_dir()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_charges(a.root, a.entry)
        generate_scan(a.param, a.data_mode, a.nphot, a.npoints, lo, hi)
        for cache in (_cache_file(), _scan_file(a.param, a.data_mode)):
            provenance.write_run_record(cache.parent, __file__, a)
    if a.plot_results or both:
        plot_fraction(out)
        plot_scan(a.param, a.data_mode, out)
        provenance.write_run_record(out, __file__, a)


if __name__ == '__main__':
    main()
