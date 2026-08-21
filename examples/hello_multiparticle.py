"""hello_multiparticle — load a MULTI-PARTICLE PhotonSim event and draw it.

The other examples all carry one particle. This one reads a GEANT4 event containing two
muons that share a vertex and fly off in independent directions, transports their photons
through SK-like water, and draws the result — two overlapping Cherenkov rings.

Nothing about the transport is multi-particle-aware: the data path consumes a flat photon
list, so an event with N primaries costs exactly what its photon count costs. What the
reader *does* give you is the split — ``particles[i]['photon_indices']`` says which photons
each primary made, so the same event can be redrawn one particle at a time.

Run:  python examples/hello_multiparticle.py --split     (writes hello_multiparticle*.png)
      python examples/hello_multiparticle.py --event 0 --split   (rings that overlap)
"""
import argparse
import os

import jax
import numpy as np

from lucid.detector_params import ParticleParams
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources.event_io import pad_photon_data
from lucid.sources.root_reader import read_particle_data_from_photonsim
from lucid.visualization import create_detector_display, unroll_layout

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
# Full grid so every PMT is reachable (a coarse grid drops sensors -> exact-zero -> white holes).
GRID = dict(n_cap=150, n_angular=250, n_height=150)
DEFAULT_FILE = 'data/water/dimuon/1000MeV_100events.root'
PDG = {13: 'mu-', -13: 'mu+', 11: 'e-', -11: 'e+', 22: 'gamma', 211: 'pi+', -211: 'pi-'}


def as_photon_data(ev, keep=None):
    """Photon dict for the is_data simulator, optionally restricted to one particle's photons.

    The reader hands back positions in METRES; the data-mode simulator divides its input by
    100 (simulator.py, "cm to m"), so convert here. ``keep`` is an index array from
    ``particles[i]['photon_indices']``.
    """
    sel = slice(None) if keep is None else np.asarray(keep)
    return {'photon_origins': np.asarray(ev['photon_origins'])[sel] * 100.0,   # m -> cm
            'photon_directions': np.asarray(ev['photon_directions'])[sel],
            'photon_times': np.asarray(ev['photon_times'])[sel],
            'wavelengths': np.asarray(ev['photon_wavelengths'])[sel]}


def display_by_particle(detector, per_particle_q, labels, out, title=None):
    """Colour every PMT by WHICH primary lit it, brightness by how much charge it got.

    One hue per primary, lightened toward white as the PMT's charge falls (percentile-clipped
    so a few bright PMTs don't flatten the rest); a PMT shared between primaries takes the hue
    of whichever contributed more, so overlap regions read as a colour boundary, not a blend.
    Built on ``unroll_layout`` so the seam and cap placement match the standard display.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection
    from matplotlib.lines import Line2D

    lay = unroll_layout(detector)
    Q = np.stack(per_particle_q)                       # (n_particles, n_sensors)
    tot = Q.sum(0)
    lit = tot > 0
    owner = Q.argmax(0)                                # which primary dominates each PMT

    # Brightness by charge, applied as a blend toward white at FULL opacity — using alpha
    # instead would blend against the background and wash the hues out.
    hi = np.percentile(tot[lit], 99.0) if lit.any() else 1.0
    f = (np.clip(tot / max(hi, 1e-9), 0.0, 1.0) ** 0.5)[:, None]

    ZERO = np.array([0.9, 0.9, 0.9, 1.0])              # same grey the canonical display uses
    hues = plt.cm.tab10(np.arange(len(per_particle_q)) % 10)
    white = np.array([1.0, 1.0, 1.0, 1.0])
    colors = np.tile(ZERO, (len(tot), 1))
    for i in range(len(per_particle_q)):
        m = lit & (owner == i)
        colors[m] = white + (hues[i] - white) * f[m]
        colors[m, 3] = 1.0

    # Match create_detector_display exactly: circles one PMT-pitch wide in DATA units, the
    # same padding, and a figure whose aspect follows the unrolled layout.
    d = lay['diameter']
    x_min, x_max = lay['x'].min() - d, lay['x'].max() + d
    y_min, y_max = lay['y'].min() - d, lay['y'].max() + d
    fig_w = 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * (y_max - y_min) / (x_max - x_min)))
    ax.add_collection(EllipseCollection(
        widths=d, heights=d, angles=0, units='x', facecolors=colors,
        offsets=np.column_stack((lay['x'], lay['y'])), transOffset=ax.transData,
        edgecolors='none'))
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=hues[i], label=l)
                       for i, l in enumerate(labels)],
              loc='upper right', frameon=False, fontsize=9)
    for ext in out:
        fig.savefig(ext, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', default=DEFAULT_FILE, help=f'PhotonSim ROOT file [{DEFAULT_FILE}]')
    ap.add_argument('--event', type=int, default=93,
                    help='event index [93: 164 deg opening, two well-separated barrel rings]')
    ap.add_argument('--split', action='store_true',
                    help='also draw one display per primary, using its own photons')
    ap.add_argument('--out', default='hello_multiparticle', help='output file prefix')
    ap.add_argument('--ext', default='png', help='output format(s), comma separated (png,pdf)')
    args = ap.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(
            f'no such file: {args.file}\n'
            'This example needs a PhotonSim ROOT file with >1 primary per event. Generate one\n'
            'with two muons sharing a vertex (PhotonSim macro):\n'
            '    /gun/randomDirection true\n'
            '    /gun/clearPrimaries\n'
            '    /gun/addPrimary mu- 1000 MeV\n'
            '    /gun/addPrimary mu- 1000 MeV\n'
            'then point --file at the result. Any multi-primary file works.')
    ev = read_particle_data_from_photonsim(args.file, args.event)
    parts = ev['particles']
    n_phot = np.asarray(ev['photon_origins']).shape[0]

    # ---- truth ------------------------------------------------------------------------
    # Every primary is its own entry; `category`/`creator_process` come from LUCiD's
    # categorizer, so this same block reads a 1-particle file or a 10-particle one.
    print(f'{args.file}  event {args.event}: {ev["n_particles"]} primaries, {n_phot} photons')
    dirs = []
    for i, p in enumerate(parts):
        ti = p['track_info']
        d = np.asarray(ti['direction'])
        dirs.append(d)
        print(f'  [{i}] {PDG.get(int(ti["pdg"]), ti["pdg"]):>5s}  {ti["energy"]:7.1f} MeV  '
              f'vertex {np.round(ti["position"], 3)} m  dir {np.round(d, 3)}  '
              f'{len(p["photon_indices"])} photons  ({ti["creator_process"]})')
    if len(dirs) == 2:
        cosop = float(np.clip(dirs[0] @ dirs[1], -1, 1))
        print(f'  opening angle {np.degrees(np.arccos(cosop)):.1f} deg')
    # The file's own PrimaryEnergy is per-primary, not the event sum — add it up here.
    print(f'  total primary energy {sum(float(p["track_info"]["energy"]) for p in parts):.1f} MeV')

    # ---- transport --------------------------------------------------------------------
    # One buffer sized to the WHOLE event, reused for the per-particle subsets: pad_photon_data
    # truncates to nbuf (a[:nbuf]) and masks beyond the true N, so a single compiled simulator
    # serves every pass and the subsets cost no extra compile.
    sim = setup_event_simulator(GEOM, n_phot, temperature=None, K=4, is_data=True,
                                hit_mode='realistic', charge_resolution=None, particle='muon',
                                physics_config=PHYS, default_detector_params=True,
                                wavelength_mode=True, **GRID)
    dummy = ParticleParams.from_cartesian(energy=1000., position=[0., 0., 0.],
                                          direction=[0., 0., 1.], t0=0.)  # ignored in data mode
    display = create_detector_display(GEOM, sparse=False)

    def run(photon_dict, tag, label):
        pd, n = pad_photon_data(photon_dict, n_phot)
        q, t = jax.lax.stop_gradient(sim(dummy, jax.random.PRNGKey(0), pd))
        q, t = np.asarray(q), np.asarray(t)
        print(f'  {label}: {n} photons -> {(q > 0).sum()} PMTs lit, {q.sum():.0f} pe')
        for ext in args.ext.split(','):
            display(q, np.where(q > 0, t, 0.), file_name=f'{args.out}{tag}.{ext.strip()}',
                    perc_min=0.0, perc_max=99.5)
        return q

    print('transporting...')
    q_all = run(as_photon_data(ev), '', 'all primaries')
    if args.split:
        per = [run(as_photon_data(ev, p['photon_indices']), f'_p{i}', f'primary {i} only')
               for i, p in enumerate(parts)]
        # The per-particle passes transport disjoint photon sets, so their charges add up to
        # the combined event apart from per-pass QE/Poisson sampling (hit_mode='realistic'
        # draws per photon). A few per-mille is the expected size of that difference.
        s = sum(q.sum() for q in per)
        print(f'  split total {s:.0f} pe vs combined {q_all.sum():.0f} pe '
              f'({100 * (s - q_all.sum()) / q_all.sum():+.2f} %, sampling noise)')

        # One picture, every primary at once, coloured by origin.
        shared = sum(1 for j in range(len(q_all))
                     if sum(q[j] > 0 for q in per) > 1)
        print(f'  {shared} PMTs lit by more than one primary')
        labels = [f'[{i}] {PDG.get(int(p["track_info"]["pdg"]), "?")} '
                  f'{p["track_info"]["energy"]:.0f} MeV' for i, p in enumerate(parts)]
        display_by_particle(generate_detector(GEOM), per, labels,
                            [f'{args.out}_byparticle.{e.strip()}' for e in args.ext.split(',')],
                            title=f'event {args.event}: {len(parts)} primaries, '
                                  f'coloured by which one lit each PMT')
    print(f'wrote {args.out}.png' + (' and per-primary displays' if args.split else ''))


if __name__ == '__main__':
    main()
