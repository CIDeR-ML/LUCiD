#!/usr/bin/env python3
"""Figure: JUNO-like WbLS sphere — differentiability of a scintillation parameter (S or kB).

The forward simulator is re-run at every scan point with a different value of the chosen
scintillation parameter, and the loss gradient dNLL/d(param) is taken by AUTOMATIC
DIFFERENTIATION through the simulator (not by rescaling a precomputed prediction). The
parameter enters as a runtime DetectorParams field, so one jax.value_and_grad of the loss is
JIT-compiled once and evaluated at each scan point.

Two scannable parameters (--param):
  * S  : scintillation light yield (ph/MeV)         — dL/dx numerator
  * kB : Birks quenching constant (mm/keV)          — dL/dx denominator (Chou/Birks)
    dL/dx = S·(dE/dx) / (1 + kB·(dE/dx) + C·(dE/dx)²)

Two ways to define the observed event n_i (--data-mode):
  * injected : the DATA-like event — real PhotonSim photons (Cherenkov + scintillation from the
               dE/dx segments) injected into the is_data sim, cached by fig_juno_cher_scint_fraction.
               The gradient's zero crossing recovers the parameter up to data-vs-model fidelity.
  * forward  : a self-consistent pseudo-data event — the SMOOTH forward prediction run once at the
               nominal value p0 (same sim + key). Since data == model at p0, this is a PERFECT
               closure: NLL is minimised and the gradient is zero at p0.

Loss = per-sensor Poisson NLL of the total predicted charge vs the fixed observed event:
    mu_i(p) = forward_sim(track, p)_i   (both emission processes)
    NLL(p)  = sum_i [ mu_i - n_i * ln(mu_i) ]

    # the scintillation-yield calibration (injected data-like event):
    python analysis/paper/fig_juno_scint_loss.py --param S  --data-mode injected
    # the Birks-constant closure (smooth pseudo-data, perfect recovery):
    python analysis/paper/fig_juno_scint_loss.py --param kB --data-mode forward
    # --generate-data / --plot-results split as usual; both by default.

`injected` mode needs the charge cache from fig_juno_cher_scint_fraction.py. Run in the
container (GPU).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

CONFIG = REPO_ROOT / 'config'
DET = 'JUNO_wbls'
GEOM = str(CONFIG / f'{DET}_geom_config.json')
PHYS = str(CONFIG / f'{DET}_physics_config.json')
ENERGY = 1000.0
SEED = 6
EPS = 1e-6
CHARGES = paths.data_dir('juno_cher_scint', 'local') / 'charges.npz'   # from the display figure

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


def _scan_file(param, mode):
    return paths.data_dir(f'juno_{param}_loss_{mode}', 'local') / 'scan.npz'


def generate_data(param, mode, theta, phi, nphot, npoints, srel_lo, srel_hi):
    import jax
    import jax.numpy as jnp
    from lucid.detector_params import ParticleParams, load_physics_config
    from lucid.simulation import setup_event_simulator
    spec = PARAMS[param]; field = spec['field']

    # Forward prediction must emit BOTH processes so the total charge feels the scintillation
    # yield/quenching. Patch make_medium in every module that binds it (see fig_juno_cher_scint).
    import importlib
    import lucid.wavelength.medium as _wl_medium
    _orig = _wl_medium.make_medium
    def _both(material, *a, **k):
        return _orig(material, *a, **k)._replace(emission_processes=('cherenkov', 'scintillation'))
    for m in ('lucid.wavelength.medium', 'lucid.simulation.simulator',
              'lucid.geometry.detector_geometry'):
        mod = importlib.import_module(m)
        if hasattr(mod, 'make_medium'):
            mod.make_medium = _both

    sim = setup_event_simulator(GEOM, nphot, temperature=0.0, K=6, is_data=False,
                                detector_type='Sphere', max_candidates_per_ray=4,
                                physics_config=PHYS, default_detector_params=False,
                                hit_mode='aggregated')
    dp0, _, _ = load_physics_config(PHYS)
    p0 = float(getattr(dp0.scintillation, field))
    key = jax.random.PRNGKey(SEED)
    track = ParticleParams(energy=jnp.array(ENERGY, jnp.float32), position=jnp.zeros(3, jnp.float32),
                           theta=jnp.array(theta, jnp.float32), phi=jnp.array(phi, jnp.float32),
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
        if not CHARGES.exists():
            raise SystemExit(f'[error] charge cache {CHARGES} missing — run '
                             f'fig_juno_cher_scint_fraction.py --generate-data first')
        z = np.load(CHARGES)
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


def plot_results(param, mode, out):
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
    ap.add_argument('--param', choices=['S', 'kB'], default='S',
                    help='scintillation parameter to scan (default S)')
    ap.add_argument('--data-mode', choices=['injected', 'forward'], default='injected',
                    help="'injected' = the real data-like event (needs the charge cache); "
                         "'forward' = self-consistent smooth pseudo-data (perfect closure)")
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--theta', type=float, default=float(np.pi / 4))
    ap.add_argument('--phi', type=float, default=float(np.pi / 6))
    ap.add_argument('--nphot', type=int, default=1_000_000)
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
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.param, a.data_mode, a.theta, a.phi, a.nphot, a.npoints, lo, hi)
    if a.plot_results or both:
        plot_results(a.param, a.data_mode, Path(a.out) if a.out else paths.figure_dir())


if __name__ == '__main__':
    main()
