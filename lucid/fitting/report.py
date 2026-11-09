"""Console tables for fit results: aligned unicode text that copy-pastes cleanly.

``print(track_table(theta, truth=truth, dir_of=vec9_dir))`` summarises a track fit.
"""
import numpy as np


def rule(text='', width=64, char='━'):
    """A titled horizontal rule, e.g. ``━━ track reconstruction ━━━━━``."""
    if not text:
        return char * width
    head = f'{char}{char} {text} '
    return head + char * max(0, width - len(head))


def table(headers, rows, *, title=None, aligns=None, indent='  '):
    """Return an aligned text table. Numeric-looking columns right-align by default.

    ``rows`` is a list of stringifiable tuples; ``aligns`` optionally overrides per-column
    alignment with ``'l'``/``'r'``. Rendered with a light rule under the header.
    """
    rows = [[('' if c is None else str(c)) for c in r] for r in rows]
    ncol = len(headers)
    if aligns is None:
        aligns = []
        for j in range(ncol):
            col = [r[j] for r in rows]
            aligns.append('r' if col and all(_looks_numeric(c) for c in col if c) else 'l')
    w = [len(str(headers[j])) for j in range(ncol)]
    for r in rows:
        for j in range(ncol):
            w[j] = max(w[j], len(r[j]))

    def _fmt(cells):
        return indent + '  '.join(
            (cells[j].rjust(w[j]) if aligns[j] == 'r' else cells[j].ljust(w[j]))
            for j in range(ncol))

    out = []
    if title:
        out.append(indent + title)
    out.append(_fmt([str(h) for h in headers]))
    out.append(indent + '─' * (sum(w) + 2 * (ncol - 1)))
    out.extend(_fmt(r) for r in rows)
    return '\n'.join(out)


def _looks_numeric(s):
    try:
        float(str(s).replace('%', '').replace('+', '').replace('°', '')
              .split()[0] if str(s).split() else str(s))
        return True
    except (ValueError, IndexError):
        return False


# ---- domain summaries -------------------------------------------------------


def track_table(theta, *, truth=None, dir_of=None):
    """Format a reconstructed 9-vector track: energy, vertex, direction, t0 (+errors vs truth).

    ``dir_of(vec9) -> unit 3-vector`` decodes the direction encoding (pass ``vec9_dir``).
    """
    theta = np.asarray(theta)
    e, vx, vy, vz, t0 = theta[0], theta[1], theta[2], theta[3], theta[8]
    rows = [['energy', f'{e:.1f} MeV'], ['vertex x', f'{vx:+.3f} m'],
            ['vertex y', f'{vy:+.3f} m'], ['vertex z', f'{vz:+.3f} m'],
            ['t0', f'{t0:+.2f} ns']]
    tail = []
    if truth is not None:
        truth = np.asarray(truth)
        for k, i in enumerate([0, 1, 2, 3]):
            err = theta[i] - truth[i]
            unit = 'MeV' if i == 0 else 'cm'
            rows[k].append(f'{err * (1 if i == 0 else 100):+.1f} {unit}')
        rows[4].append(f'{theta[8] - truth[8]:+.2f} ns')
        vtx = float(np.linalg.norm((theta[1:4] - truth[1:4]))) * 100
        tail.append(f'  |vertex error| = {vtx:.1f} cm')
        if dir_of is not None:
            c = float(np.clip(np.dot(dir_of(theta), dir_of(truth)), -1, 1))
            tail.append(f'  direction error = {np.degrees(np.arccos(c)):.2f}°')
    headers = ['parameter', 'fit'] + (['error'] if truth is not None else [])
    return '\n'.join([rule('track reconstruction'), table(headers, rows)] + tail)
