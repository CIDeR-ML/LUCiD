"""Pretty, dependency-light console output for fits and Fisher/CRB summaries.

One house style for the whole toolkit so ``fit``, ``fit_track``, ``crb`` and the scripts all
print the same way. No hard dependency beyond ``tqdm`` (already required): progress bars degrade
to a plain iterator when disabled or off-TTY, and tables are aligned unicode text that copy-paste
cleanly. Everything here is opt-in — pass ``verbose=True`` to a fitter; the default is silent, so
notebooks and tests are unaffected.
"""
import sys
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is a declared dependency
    tqdm = None


def progress(iterable, *, desc='', total=None, verbose=False):
    """Wrap ``iterable`` in a tqdm bar when ``verbose`` (and tqdm is available), else pass through.

    Use the returned object's ``.set_postfix_str(...)`` inside the loop for a live metric; it is a
    no-op on the plain fallback so call sites never branch.
    """
    if verbose and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total, leave=True,
                    bar_format='  {desc} {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    class _Plain:
        def __init__(self, it):
            self._it = iter(it)

        def __iter__(self):
            return self._it

        def set_postfix_str(self, *_a, **_k):
            pass

    return _Plain(iterable)


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


def emit(text, *, verbose=True, stream=None):
    """Print ``text`` only when ``verbose``. Central switch so call sites stay one-liners."""
    if verbose:
        print(text, file=stream or sys.stdout, flush=True)


# ---- domain summaries -------------------------------------------------------

def crb_table(result, names, *, as_percent=True):
    """Format a CRB summary: per-parameter fractional 1σ and the strongest degeneracy."""
    sigma = np.asarray(result['sigma'])
    cov = np.asarray(result['cov'])
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(d, d)
    scale, unit = (100.0, '%') if as_percent else (1.0, '')
    rows = [[names[i], f'{sigma[i] * scale:.2f}{unit}'] for i in range(len(names))]
    ev = np.linalg.eigvalsh(np.asarray(result['fisher']))
    cond = ev.max() / max(ev.min(), 1e-300)
    iu = np.triu_indices(len(names), 1)
    k = int(np.argmax(np.abs(corr[iu]))) if len(iu[0]) else 0
    out = [rule('Cramér–Rao bound (fractional 1σ at truth)'),
           table(['parameter', '1σ'], rows)]
    if len(iu[0]):
        out.append(f'  Fisher condition number {cond:.1e}   strongest degeneracy: '
                   f'{names[iu[0][k]]} ↔ {names[iu[1][k]]} = {corr[iu][k]:+.2f}')
    return '\n'.join(out)


def calib_table(fit_theta, names, *, truth=None, sigma=None):
    """Format a calibration result: fit value, optional truth/error, optional CRB σ."""
    fit_theta = np.asarray(fit_theta)
    headers = ['parameter', 'fit']
    if truth is not None:
        headers += ['truth', 'error']
    if sigma is not None:
        headers += ['CRB σ']
    truth = None if truth is None else np.asarray(truth)
    sigma = None if sigma is None else np.asarray(sigma)
    rows = []
    for i, nm in enumerate(names):
        row = [nm, f'{fit_theta[i]:.4g}']
        if truth is not None:
            frac = (fit_theta[i] - truth[i]) / truth[i] * 100 if truth[i] else np.nan
            row += [f'{truth[i]:.4g}', f'{frac:+.1f}%']
        if sigma is not None:
            row += [f'{sigma[i] * 100:.1f}%']
        rows.append(row)
    out = [rule('calibration fit'), table(headers, rows)]
    if truth is not None and sigma is not None:
        within = np.abs(fit_theta - truth) <= sigma * np.abs(truth) * 1.0
        out.append(f'  {int(within.sum())}/{len(names)} parameters within 1σ CRB'
                   + ('  ✓' if within.all() else ''))
    return '\n'.join(out)


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
