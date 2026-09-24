"""Photosensor charge response f(q | mu) — the ``<mu>_pdf.root`` stage-1 inputs.

fiTQun's charge PDF is a property of the photosensor and its electronics
alone: no geometry, no tracks, no light propagation. The reference tune gets
it by running WCSim with ``/mygen/pmtPoisson true`` + ``/mygen/poissonMean
<mu>`` — every PMT is handed Poisson(mu) photoelectrons and digitised — and
histogramming the resulting charges per mu.

LUCiD models the same response in :mod:`lucid.simulation.digitizer` (the SPE
spectrum, the integration window, the discriminator threshold), so this module
reproduces that generator against LUCiD's digitizer instead of WCSim's. It
calls the production digitizer rather than re-deriving it, so a change to the
detector response shows up in the next tune instead of silently diverging
from it.

The discriminator is LUCiD's own: a single sharp threshold on the digitised
charge, applied identically to SK and HK. WCSim instead uses an SK-specific
measured S-curve; the deliberate difference is recorded in
:mod:`lucid.simulation.digitizer`. What matters here is that the table
describes the response LUCiD actually has, since that is what fiTQun is being
tuned to.

Output per mu, matching ``Utilities/chrgpdf/workdir/makeChargePDFplot.C`` so
``gen2d.cc`` and everything downstream run unchanged:

    hchpdf2, hchpdf3   observed charge q (p.e.), one entry per digitised hit
    hctr               bin 2, 3: number of active PMTs; bin 10: their sum
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from lucid.simulation.digitizer import (  # noqa: F401  re-exported for tests
    apply_discriminator,
    apply_readout_resolution,
    digitize_event,
    resolve_model_config,
)

from . import binning, rootio


# Cap on the per-pass photoelectron list. At the top of the reference mu grid
# (~1090 p.e.) a few thousand PMTs times a few tens of shots is O(10^8)
# photoelectrons, so the shots are batched rather than flattened in one go.
_MAX_PE_PER_PASS = 20_000_000


def sample_charges(mu: float, n_pmt: int, n_events: int, model: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Digitised charges from ``n_events`` shots of Poisson(mu) p.e. on ``n_pmt`` PMTs.

    Each PMT sees its photoelectrons simultaneously (they are one "hit"), so
    every photoelectron is placed at t=0 and the integration window collapses
    to a single digit per PMT — the same thing WCSim's pmtPoisson generator
    produces. Going through the production digitizer rather than calling the
    SPE sampler directly is deliberate: it means the threshold, the window and
    the charge model are whatever the detector actually uses.
    """
    per_shot = max(1.0, mu) * n_pmt
    batch = max(1, min(n_events, int(_MAX_PE_PER_PASS // per_shot)))
    out = []
    remaining = n_events
    while remaining > 0:
        n = min(batch, remaining)
        remaining -= n
        npe = rng.poisson(mu, size=(n, n_pmt))
        hit = npe > 0
        if not hit.any():
            continue
        # One photon entry per photoelectron, tagged with a unique (shot, pmt)
        # index so the digitizer treats each PMT-shot separately.
        flat_sensor = np.repeat(np.flatnonzero(hit.reshape(-1)), npe[hit])
        zeros = np.zeros(flat_sensor.size, dtype=np.float64)
        res = digitize_event(flat_sensor, zeros, np.ones_like(zeros),
                             n_sensors=n * n_pmt, model=model)
        q, _ = apply_readout_resolution(res.digit_pe_true, res.digit_time, model, rng)
        q = np.asarray(q, dtype=np.float64)[apply_discriminator(q, model)]
        out.append(q)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def build(mu: float, *, n_pmt: int, n_events: int, model: Union[str, dict, None],
          seed: int, q_edges: Optional[np.ndarray] = None) -> dict:
    """The ROOT objects for one mu point of the charge-PDF scan."""
    resolved = resolve_model_config(model)
    if resolved.get("charge_model") != "spe":
        raise ValueError(
            f"digitizer model {resolved['model']!r} has no physical SPE charge "
            "response; the charge PDF would describe the legacy smear, not a PMT")
    q_edges = binning.charge_q_edges() if q_edges is None else np.asarray(q_edges)
    rng = np.random.default_rng(seed)

    q = sample_charges(mu, n_pmt, n_events, resolved, rng)
    counts, _ = np.histogram(q, bins=q_edges)
    # Charge above the last edge goes in the ROOT overflow bin, which
    # gen2d.cc copies explicitly (for j=1; j<=nqbns+1). Dropping it would
    # lose the high-charge tail and skew the GetEntries() normalisation.
    overflow = float((q >= q_edges[-1]).sum())
    n_fills = float(counts.sum()) + overflow

    # hctr counts every PMT that could have fired, hit or not -- it is the
    # denominator of P(hit | mu).
    n_active = float(n_pmt) * n_events
    ctr = np.zeros(10, dtype=np.float64)
    ctr[1] = ctr[2] = n_active
    ctr[9] = ctr[1] + ctr[2]

    objects = {}
    for name in ("hchpdf2", "hchpdf3"):
        # Unit-weight fills, so Sumw2 equals the content -- matching what
        # makeChargePDFplot.C's Sumw2() produces.
        objects[name] = rootio.th1(
            name, q_edges, counts.astype(np.float64), sumw2=counts.astype(np.float64),
            title="Old PMT Charge PDF", xtitle="q (p.e.)",
            overflow=overflow, sumw2_flow=(0.0, overflow), entries=n_fills)
    # hctr is filled once per event with weight n_pmt, so its Sumw2 is
    # n_events * n_pmt^2, not the content.
    objects["hctr"] = rootio.th1(
        "hctr", np.linspace(0.5, 10.5, 11), ctr, sumw2=ctr * float(n_pmt),
        title="Total # of active PMTs", entries=float(n_events))
    return {"objects": objects, "n_hits": n_fills,
            "n_active": n_active, "overflow": overflow}


def write_mu_point(out_dir, mu: float, label: Optional[str] = None, **kwargs) -> Path:
    """Write ``<mu>_pdf.root`` for one mu, named as the reference chain expects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(mu, **kwargs)
    # gen2d.cc opens the file by the literal token from mutbl.txt, so "1.0"
    # must stay "1.0"; f"{mu:g}" would write "1" and the lookup would miss.
    path = out_dir / f"{label or f'{mu:g}'}_pdf.root"
    rootio.write(path, result["objects"])
    return path


def run_scan(out_dir, *, n_pmt: int, n_events: int, model, seed: int,
             mu_values: Optional[np.ndarray] = None, verbose: bool = True) -> list[Path]:
    """Write the whole mu scan. Each point gets its own seed stream."""
    if mu_values is None:
        labels = binning.charge_mu_labels()
        mu_values = np.array([float(t) for t in labels])
    else:
        mu_values = np.asarray(mu_values)
        labels = [f"{m:g}" for m in mu_values]
    paths = []
    for i, (mu, label) in enumerate(zip(mu_values, labels)):
        # Fewer events suffice at large mu (every PMT fires); mirrors the
        # reference scan's 80 -> 20 event schedule at mu > 30.
        n = n_events if mu <= 30 else max(1, n_events // 4)
        paths.append(write_mu_point(out_dir, float(mu), label=label, n_pmt=n_pmt,
                                    n_events=n, model=model, seed=seed + i))
        if verbose:
            print(f"  mu={mu:<8g} -> {paths[-1].name}", flush=True)
    return paths

