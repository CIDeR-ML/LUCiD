"""A trained SIREN emitter: one directory, one record, one evaluator.

An emitter lives in ``data/<material>/<particle>/<kind>/`` (``kind`` is ``'cherenkov'`` or
``'dedx'``) as two files:

- ``emitter.json`` (tracked): the network shape, the input domain, the target decoding, the
  sampling grid, the fitted curves s_max(E) and, for Cherenkov, N_photons(E), the arrival-time fit
  t0 and the emission band, plus the provenance and training recipe;
- ``weights.npz`` (installed by ``scripts/download_data.py``): the network parameters.

:func:`load_emitter` reads them into one :class:`Emitter`; :func:`write_emitter` is the only
output path (the trainer's and the promotion's), and :func:`validate_record` checks every record
it writes and :func:`read_emitter` reads, so a model written is a model the loader reads.
:func:`density` is the one evaluator (normalise the inputs, run the net, decode the target),
shared by the ray samplers, the validators and the training plots.

A training run (``lucid-train-siren``) writes an emitter without the t0 fit and the emission band;
``lucid-siren-promote`` adds them and installs it in ``data/``. :func:`load_emitter` reads only
promoted emitters.

A material that has no emitters of its own names the set it runs in its material JSON
(``config/materials/<material>.json``, key ``"siren"``: wbls and ice run ``"water"``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze

from lucid.siren.net import SIREN
from lucid.utils import base_dir_path

SCHEMA = 1
KINDS = ('cherenkov', 'dedx')
PARTICLES = ('muon', 'electron')
PHOTONSIM_PARTICLE = {'muon': 'mu-', 'electron': 'e-'}     # the tables' and PhotonSim's names
T0_FORM = 'stretched_exp_delay'
_AXIS2 = {'cherenkov': 'angle_rad', 'dedx': 'dedx_kev_per_mm'}

# The provenance block. The trainer writes TRAINED; lucid-siren-promote adds PROMOTED. A key whose
# value was never recorded holds null.
PROVENANCE_TRAINED = ('material', 'particle', 'table', 'steps', 'losses', 'init', 'commit',
                      'trained_at_utc')
PROVENANCE_PROMOTED = ('run', 'selection', 'smax_fit', 'promoted_at_utc', 'promoted_commit')
LOSSES = ('final_train', 'final_val', 'best_val')


@dataclass(frozen=True, eq=False)
class Emitter:
    """One trained SIREN emitter, as :func:`read_emitter` builds it from its directory.

    ``params`` is a pytree: the ray functions take it as an argument, so the caller decides whether
    it is traced or closed over. Everything else is a plain Python value or closure, never traced.
    """
    kind: str                                   # 'cherenkov' | 'dedx'
    net: SIREN
    params: FrozenDict
    lo: tuple                                   # domain lower bounds: (E MeV, axis2, s/s_max)
    hi: tuple                                   # axis2: opening angle (rad) or dE/dx (keV/mm)
    log10_range: tuple                          # (log10_min, log10_max) of the decoded target
    target_offset: float                        # the training offset: the net learns log10(t + offset)
    grid_bins: int                              # the ray samplers' first-pass grid, per axis
    n_photons: Optional[Callable]               # Cherenkov: N_photons(E_MeV), traceable
    t0: Optional[tuple]                         # Cherenkov: (log10 A, log10 lambda_mm, beta), or None
                                                # for one trained without a t0 fit
    c_mm_per_ns: Optional[float]                # Cherenkov: the t0 fit's speed of light
    emission_band: Optional[tuple]              # Cherenkov: (lo, hi) nm, the band N_photons(E)
                                                # counts, or None for one recorded without it
    meta: dict = field(repr=False)              # emitter.json as read; physics reads none of it
    # s_max(E_MeV) in mm. The fits are in mm; the two methods below are the only conversion.
    _s_max_mm: Callable = field(repr=False)

    def along_track_m(self, s_over_smax, energy):
        """Distance along the track in metres at ``s/s_max`` (traceable; the ray samplers)."""
        return s_over_smax * self._s_max_mm(energy) / 1000.0

    def s_max_m(self, energy) -> float:
        """The track length s_max(E) in metres, on the host (placement, validators, figures)."""
        return float(self._s_max_mm(float(energy))) / 1000.0


# The parameters of each s_max(E) form (the columns of PhotonSim's smax_fit.csv; keep in step with
# FORMS in PhotonSim/tools/smax/analyze_smax.py). "piecewise": the low piece in (a, b1, b2, E0), the
# high piece in the *_hi columns, joined at e_join_mev.
SMAX_PARAMS = {
    "A*E^B":            ("A", "B"),
    "smooth_two_power": ("a", "b1", "b2", "E0"),
    "piecewise":        ("a", "b1", "b2", "E0", "e_join_mev", "a_hi", "b1_hi", "b2_hi", "E0_hi"),
}


def make_smax_fn(smax: dict, xp=jnp) -> Callable:
    """Build a ``s_max(E_mev) -> mm`` closure from an ``s_max_mm`` block (``{'form', 'params'}``;
    params as floats or CSV strings).

    The ``form`` is fixed per model, so it is dispatched in Python here. The ``piecewise`` branch
    selects with ``xp.where``: with ``xp=jax.numpy`` the closure is traceable, with ``xp=numpy`` a
    float energy stays float64. Mirrors ``_eval_smax`` in ``PhotonSim/tools/smax/analyze_smax.py``.
    """
    form = smax['form']
    p = smax['params']
    if form == 'A*E^B':
        A, B = float(p['A']), float(p['B'])
        return lambda E: A * E ** B
    if form == 'smooth_two_power':
        a, b1, b2, E0 = (float(p[k]) for k in ('a', 'b1', 'b2', 'E0'))
        return lambda E: a * E ** b1 / (1.0 + (E / E0) ** (b1 - b2))
    if form == 'piecewise':
        ej = float(p['e_join_mev'])
        a, b1, b2, E0 = (float(p[k]) for k in ('a', 'b1', 'b2', 'E0'))
        ah, bh1, bh2, Eh0 = (float(p[k])
                             for k in ('a_hi', 'b1_hi', 'b2_hi', 'E0_hi'))

        def _piecewise(E):
            low = a * E ** b1 / (1.0 + (E / E0) ** (b1 - b2))
            high = ah * E ** bh1 / (1.0 + (E / Eh0) ** (bh1 - bh2))
            return xp.where(E < ej, low, high)

        return _piecewise
    raise ValueError(f"unknown smax form: {form!r}")


def make_n_photons_fn(n_photons: dict) -> Callable:
    """Build a jittable ``N_photons(E_mev)`` closure from an ``n_photons`` block.

    ``N_photons(E)`` is the ABSOLUTE SCALE of the Cherenkov emitter: the net supplies a normalised
    PMF, this scalar the magnitude. The reconstruction differentiates through it
    (``MODEL_RECIPE.energy_scale='nphot'``), so its shape matters as much as its value.

    ``N = exp(sum_k c_k t^k)``, ``t = (ln E - u0)/du``, fitted with equal fractional weight at every
    energy: strictly positive and smooth, so it needs no clamp and is safe to differentiate twice.
    THE CENTRED BASIS IS LOAD-BEARING: in a raw ``ln E`` basis Horner sums large terms of
    alternating sign to produce ``lnN ~ 12``, and float32 cancellation then costs more than the
    fit error. Centred, float32 matches float64.
    """
    if n_photons['form'] != 'logpoly':
        raise ValueError(f"unknown n_photons form: {n_photons['form']!r}")
    co = jnp.asarray([float(v) for v in n_photons['coeffs']])       # ASCENDING, centred basis
    u0 = float(n_photons['u0'])
    du = float(n_photons['du'])

    def _logpoly(E):
        t = (jnp.log(E) - u0) / du
        acc = co[-1]
        for k in range(co.shape[0] - 2, -1, -1):                    # Horner, descending
            acc = acc * t + co[k]
        return jnp.exp(acc)

    return _logpoly


# ---- the evaluator ----------------------------------------------------------------------------

# The training/inference contract: the trainer feeds the net normalize_inputs(x) and fits it to
# encode_target(t); inference reads decode_target(net(normalize_inputs(x))). The emitter records
# the encoding's constants (target.log10_min, log10_max, offset) and the domain.

@jax.jit
def normalize_inputs(inputs, energy_min, energy_max, axis2_min, axis2_max, r_min, r_max):
    """``(n, 3)`` inputs ``[E, axis2, s/s_max]`` mapped linearly to ``[-1, 1]`` per column."""
    energy = inputs[:, 0]
    axis2 = inputs[:, 1]
    r = inputs[:, 2]
    normalized_energy = 2.0 * (energy - energy_min) / (energy_max - energy_min) - 1.0
    normalized_axis2 = 2.0 * (axis2 - axis2_min) / (axis2_max - axis2_min) - 1.0
    normalized_r = 2.0 * (r - r_min) / (r_max - r_min) - 1.0
    return jnp.stack([normalized_energy, normalized_axis2, normalized_r], axis=1)


def target_range(targets, offset):
    """``(log10_min, log10_max)``, the range of ``log10(t + offset)`` over a table's targets: the
    constants of :func:`encode_target`, recorded as ``target.log10_min`` and ``log10_max``."""
    log_targets = np.log10(np.asarray(targets) + offset)
    return log_targets.min(), log_targets.max()


def encode_target(targets, log10_min, log10_max, offset):
    """The training target: ``(log10(t + offset) - log10_min) / (log10_max - log10_min)``, in
    ``[0, 1]`` over the table. NumPy, in the dtype of ``targets``; :func:`decode_target` inverts it."""
    return (np.log10(np.asarray(targets) + offset) - log10_min) / (log10_max - log10_min)


@jax.jit
def decode_target(raw, log10_min, log10_max, offset):
    """The net's ``[0, 1]`` output back to the target, the inverse of :func:`encode_target`.
    Clipped at 0: an empty bin decodes to about 0, a float32 rounding below it to exactly 0."""
    log_predictions = raw * (log10_max - log10_min) + log10_min
    return jnp.maximum(10 ** log_predictions - offset, 0.0)


def density(e: Emitter, params, energy, axis2, s_over_smax):
    """The emitter's density at the points ``(axis2, s/s_max)`` for one ``energy`` (MeV).

    ``axis2`` is the opening angle (rad) for a Cherenkov emitter or dE/dx (keV/mm) for a dE/dx
    emitter. Returns decoded, non-negative weights.
    """
    grid = jnp.stack([
        jnp.full_like(axis2, energy),   # energy (MeV)
        axis2,                          # opening angle (Cherenkov) or dE/dx (dedx)
        s_over_smax,                    # s / s_max  in [0, 1]
    ], axis=1)
    normalized = normalize_inputs(grid, e.lo[0], e.hi[0], e.lo[1], e.hi[1], e.lo[2], e.hi[2])
    raw = e.net.apply(params, normalized)
    return decode_target(jnp.squeeze(raw), e.log10_range[0], e.log10_range[1], e.target_offset)


# ---- files ------------------------------------------------------------------------------------

def read_weights(path) -> FrozenDict:
    """``weights.npz`` (flat keys ``params_<layer>_Dense_0_<kernel|bias>``) as the flax tree."""
    nested = {}
    with np.load(path) as f:
        for k in f.files:
            layer, name = k[len('params_'):].rsplit('_Dense_0_', 1)
            nested.setdefault(layer, {}).setdefault('Dense_0', {})[name] = jnp.array(f[k])
    return freeze({'params': nested})


def write_weights(path, params) -> None:
    """The inverse of :func:`read_weights`."""
    flat = {f'params_{layer}_Dense_0_{name}': np.asarray(v)
            for layer, dense in params['params'].items()
            for name, v in dense['Dense_0'].items()}
    np.savez(path, **flat)


def _numbers(v, n=None) -> bool:
    """``v`` is a list of finite numbers (of length ``n``)."""
    return (isinstance(v, list) and len(v) > 0 and (n is None or len(v) == n)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)
                    for x in v))


def _number(v) -> bool:
    return _numbers([v])


def validate_record(record: dict, *, promoted: bool = False) -> None:
    """Check an ``emitter.json`` record; raise ``ValueError`` naming every problem.

    The structure every emitter has (schema, kind, net, domain, target, grid, s_max(E) and, for
    Cherenkov, N_photons(E)), the t0 fit and emission band when present, and the provenance keys
    the trainer writes. With ``promoted`` (an installed emitter, :func:`load_emitter`): a Cherenkov
    emitter must also carry ``t0`` and ``emission_band_nm``, and the provenance the promotion adds.
    """
    bad = []
    if record.get('schema') != SCHEMA:
        bad.append(f"schema {record.get('schema')!r}, this LUCiD reads schema {SCHEMA}")
    kind = record.get('kind')
    if kind not in KINDS:
        bad.append(f"kind {kind!r}, expected one of {KINDS}")
    net = record.get('net', {})
    if not (isinstance(net.get('hidden_features'), int) and net['hidden_features'] > 0
            and isinstance(net.get('hidden_layers'), int) and net['hidden_layers'] >= 0
            and _number(net.get('w0'))):
        bad.append(f"net {net!r}: needs hidden_features > 0, hidden_layers >= 0 and w0")
    dom = record.get('domain', {})
    for axis in ('energy_mev', _AXIS2.get(kind, 'angle_rad'), 's_over_smax'):
        if not (_numbers(dom.get(axis), 2) and dom[axis][0] < dom[axis][1]):
            bad.append(f"domain.{axis} {dom.get(axis)!r}: needs [lo, hi] with lo < hi")
    tg = record.get('target', {})
    if not (_number(tg.get('log10_min')) and _number(tg.get('log10_max'))
            and tg['log10_min'] < tg['log10_max'] and _number(tg.get('offset'))
            and tg['offset'] > 0):
        bad.append(f"target {tg!r}: needs log10_min < log10_max and offset > 0")
    if not (isinstance(record.get('grid_bins'), int) and record['grid_bins'] > 0):
        bad.append(f"grid_bins {record.get('grid_bins')!r}: needs an integer > 0")
    sm = record.get('s_max_mm', {})
    if sm.get('form') not in SMAX_PARAMS:
        bad.append(f"s_max_mm.form {sm.get('form')!r}, expected one of {sorted(SMAX_PARAMS)}")
    else:
        missing = [k for k in SMAX_PARAMS[sm['form']] if k not in sm.get('params', {})]
        if missing:
            bad.append(f"s_max_mm.params lacks {missing} of form {sm['form']!r}")
    has = {k: k in record for k in ('n_photons', 't0', 'emission_band_nm')}
    if kind == 'dedx':
        bad += [f"a dE/dx emitter has no {k}" for k, v in has.items() if v]
    if kind == 'cherenkov':
        nph = record.get('n_photons', {})
        if not (nph.get('form') == 'logpoly' and _numbers(nph.get('coeffs'))
                and _number(nph.get('u0')) and _number(nph.get('du')) and nph['du'] != 0):
            bad.append("n_photons: needs form 'logpoly', coeffs, u0 and du != 0")
        if has['t0']:
            t = record['t0']
            if t.get('form') != T0_FORM:
                bad.append(f"unsupported t0 form {t.get('form')!r}; expected {T0_FORM!r}")
            if not (_number(t.get('c_mm_per_ns')) and t['c_mm_per_ns'] > 0):
                bad.append("t0.c_mm_per_ns: needs a speed > 0")
            for k in ('log10_A', 'log10_lambda_mm', 'beta'):
                if not _numbers(t.get(k), 4):
                    bad.append(f"t0.{k}: needs 4 cubic coefficients in log10 E")
        elif promoted:
            bad.append("a Cherenkov emitter needs its t0 fit (lucid-siren-promote --t0)")
        if has['emission_band_nm']:
            band = record['emission_band_nm']
            if not (_numbers(band, 2) and 0 < band[0] < band[1]):
                bad.append(f"emission_band_nm {band!r}: needs [lo, hi] nm with 0 < lo < hi")
        elif promoted:
            bad.append("a Cherenkov emitter needs emission_band_nm "
                       "(lucid-siren-promote --emission-band-nm)")
    prov = record.get('provenance')
    if not isinstance(prov, dict):
        bad.append("no provenance block")
    else:
        need = PROVENANCE_TRAINED + (PROVENANCE_PROMOTED if promoted else ())
        missing = [k for k in need if k not in prov]
        if missing:
            bad.append(f"provenance lacks {missing}"
                       + (" (install an emitter with lucid-siren-promote)" if promoted else ""))
        losses = prov.get('losses')
        if 'losses' in prov and not (isinstance(losses, dict) and all(k in losses for k in LOSSES)):
            bad.append(f"provenance.losses {losses!r}: needs {LOSSES}")
    if not isinstance(record.get('training', {}).get('recipe'), dict):
        bad.append("no training.recipe")
    if bad:
        raise ValueError("invalid emitter record: " + "; ".join(bad))


def read_emitter(model_dir, *, promoted: bool = False) -> Emitter:
    """The emitter in ``model_dir`` (``emitter.json`` + ``weights.npz``), its record checked by
    :func:`validate_record` (``promoted``: as an installed emitter)."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / 'emitter.json').read_text())
    try:
        validate_record(meta, promoted=promoted)
    except ValueError as exc:
        raise ValueError(f"{model_dir}/emitter.json: {exc}") from None
    kind = meta['kind']
    dom = meta['domain']
    axes = (dom['energy_mev'], dom[_AXIS2[kind]], dom['s_over_smax'])
    t0 = c_mm_per_ns = None
    if 't0' in meta:                            # a Cherenkov emitter's; the simulator needs it
        t = meta['t0']
        t0 = tuple(jnp.asarray(t[k]) for k in ('log10_A', 'log10_lambda_mm', 'beta'))
        c_mm_per_ns = float(t['c_mm_per_ns'])
    return Emitter(
        kind=kind,
        net=SIREN(out_features=1, **meta['net']),
        params=read_weights(model_dir / 'weights.npz'),
        lo=tuple(float(a[0]) for a in axes),
        hi=tuple(float(a[1]) for a in axes),
        log10_range=(float(meta['target']['log10_min']), float(meta['target']['log10_max'])),
        target_offset=float(meta['target']['offset']),
        grid_bins=int(meta['grid_bins']),
        n_photons=make_n_photons_fn(meta['n_photons']) if kind == 'cherenkov' else None,
        t0=t0,
        c_mm_per_ns=c_mm_per_ns,
        emission_band=(tuple(float(v) for v in meta['emission_band_nm'])
                       if 'emission_band_nm' in meta else None),
        meta=meta,
        _s_max_mm=make_smax_fn(meta['s_max_mm']),
    )


def write_emitter(model_dir, record: dict, params, *, promoted: bool = False) -> None:
    """Write ``record`` (the ``emitter.json`` content) and ``params`` into ``model_dir``, after
    :func:`validate_record` (``promoted``: as an installed emitter)."""
    record = {'schema': SCHEMA, **record}
    validate_record(record, promoted=promoted)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'emitter.json').write_text(json.dumps(record, indent=1) + '\n')
    write_weights(model_dir / 'weights.npz', params)


def emitter_dir(particle: str, material: str, kind: str) -> Path:
    """``data/<material>/<particle>/<kind>/``."""
    if particle not in PARTICLES or kind not in KINDS:
        raise ValueError(f"no SIREN emitter {particle!r}/{kind!r}; particles {PARTICLES}, "
                         f"kinds {KINDS}")
    return Path(base_dir_path()) / 'data' / material / particle / kind


def load_emitter(particle: str, material: str, kind: str) -> Emitter:
    """The ``kind`` emitter trained for ``particle`` in ``material`` (the emitter set's name, as a
    material JSON's ``"siren"`` key gives it). Only a promoted emitter loads."""
    return read_emitter(emitter_dir(particle, material, kind), promoted=True)
