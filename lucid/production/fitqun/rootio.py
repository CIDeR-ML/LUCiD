"""Writers for the ROOT objects the fiTQun tuning chain reads.

fiTQun and the ``Utilities`` tuning code are ROOT programs; LUCiD is not. This
module is the seam: it builds ROOT ``TH1D``/``TH2D``/``TH3F``/``TGraph``
objects with uproot so the tables LUCiD produces drop straight into the
existing chain with no conversion step.

uproot writes histograms out of the box; it has no ``TGraph`` *writer*, only a
reader, so :func:`tgraph` assembles the model by hand from uproot's own
``TGraph`` v4 model and its bases. Round-tripping through uproot's reader is
the check that the byte layout is right (``tests/test_fitqun_rootio.py``).

Bin contents are passed **without** under/overflow and the helpers pad them,
since every table here is defined on its stated range.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import uproot
from uproot.models.TAtt import (
    Model_TAttFill_v1,
    Model_TAttLine_v1,
    Model_TAttMarker_v2,
)
from uproot.models.TGraph import Model_TGraph_v4
from uproot.models.TList import Model_TList
from uproot.models.TNamed import Model_TNamed
from uproot.models.TObject import Model_TObject
from uproot.writing.identify import to_TAxis, to_TH1x, to_TH2x, to_TH3x


def _blank(cls, version: int, members: dict, bases=()):
    """An unread uproot model — the shape ``Model._to_writable`` expects."""
    obj = cls.__new__(cls)
    obj._cursor = None
    obj._file = None
    obj._parent = None
    obj._concrete = None
    obj._num_bytes = None
    obj._instance_version = version
    obj._is_memberwise = False
    obj._bases = list(bases)
    obj._members = dict(members)
    return obj


def _axis(name: str, edges: np.ndarray, title: str = ""):
    edges = np.asarray(edges, dtype=np.float64)
    uniform = np.allclose(np.diff(edges), edges[1] - edges[0])
    return to_TAxis(
        fName=name, fTitle=title, fNbins=len(edges) - 1,
        fXmin=float(edges[0]), fXmax=float(edges[-1]),
        # ROOT treats an empty fXbins as "uniform"; variable binning must carry
        # the explicit edges or every lookup silently uses the wrong bin.
        fXbins=None if uniform else edges,
    )


def _pad(values: np.ndarray) -> np.ndarray:
    """Surround bin contents with the zero under/overflow ROOT expects."""
    return np.pad(np.asarray(values), [(1, 1)] * values.ndim, mode="constant")


def th1(name: str, edges, values, *, sumw2=None, title: str = "",
        xtitle: str = "", dtype=np.float64):
    values = np.asarray(values, dtype=dtype)
    data = _pad(values).reshape(-1)
    errs = _pad(np.asarray(sumw2, dtype=np.float64)).reshape(-1) if sumw2 is not None else None
    centres = 0.5 * (np.asarray(edges)[1:] + np.asarray(edges)[:-1])
    sumw = float(values.sum())
    return to_TH1x(
        fName=name, fTitle=title, data=data,
        fEntries=sumw, fTsumw=sumw, fTsumw2=float((values**2).sum()),
        fTsumwx=float((values * centres).sum()),
        fTsumwx2=float((values * centres**2).sum()),
        fSumw2=errs, fXaxis=_axis("xaxis", edges, xtitle),
    )


def th2(name: str, xedges, yedges, values, *, sumw2=None, title: str = "",
        xtitle: str = "", ytitle: str = "", dtype=np.float64):
    """``values`` is indexed ``[ix, iy]``; ROOT's global bin runs x fastest."""
    values = np.asarray(values, dtype=dtype)
    data = _pad(values).T.reshape(-1)
    errs = _pad(np.asarray(sumw2, dtype=np.float64)).T.reshape(-1) if sumw2 is not None else None
    xc = 0.5 * (np.asarray(xedges)[1:] + np.asarray(xedges)[:-1])
    yc = 0.5 * (np.asarray(yedges)[1:] + np.asarray(yedges)[:-1])
    sumw = float(values.sum())
    wx, wy = values.sum(axis=1), values.sum(axis=0)
    return to_TH2x(
        fName=name, fTitle=title, data=data,
        fEntries=sumw, fTsumw=sumw, fTsumw2=float((values**2).sum()),
        fTsumwx=float((wx * xc).sum()), fTsumwx2=float((wx * xc**2).sum()),
        fTsumwy=float((wy * yc).sum()), fTsumwy2=float((wy * yc**2).sum()),
        fTsumwxy=float((values * xc[:, None] * yc[None, :]).sum()),
        fSumw2=errs,
        fXaxis=_axis("xaxis", xedges, xtitle), fYaxis=_axis("yaxis", yedges, ytitle),
    )


def th3(name: str, xedges, yedges, zedges, values, *, title: str = "",
        xtitle: str = "", ytitle: str = "", ztitle: str = "", dtype=np.float32):
    """``values`` is indexed ``[ix, iy, iz]``. Defaults to ``TH3F`` (float32),
    which is what fiTQun's ``LoadProfiles`` casts the I_n tables to."""
    values = np.asarray(values, dtype=dtype)
    data = _pad(values).transpose(2, 1, 0).reshape(-1)
    xc = 0.5 * (np.asarray(xedges)[1:] + np.asarray(xedges)[:-1])
    yc = 0.5 * (np.asarray(yedges)[1:] + np.asarray(yedges)[:-1])
    zc = 0.5 * (np.asarray(zedges)[1:] + np.asarray(zedges)[:-1])
    v64 = values.astype(np.float64)
    sumw = float(v64.sum())
    wx, wy, wz = v64.sum(axis=(1, 2)), v64.sum(axis=(0, 2)), v64.sum(axis=(0, 1))
    return to_TH3x(
        fName=name, fTitle=title, data=data,
        fEntries=sumw, fTsumw=sumw, fTsumw2=float((v64**2).sum()),
        fTsumwx=float((wx * xc).sum()), fTsumwx2=float((wx * xc**2).sum()),
        fTsumwy=float((wy * yc).sum()), fTsumwy2=float((wy * yc**2).sum()),
        fTsumwz=float((wz * zc).sum()), fTsumwz2=float((wz * zc**2).sum()),
        fTsumwxy=float((v64 * xc[:, None, None] * yc[None, :, None]).sum()),
        fTsumwxz=float((v64 * xc[:, None, None] * zc[None, None, :]).sum()),
        fTsumwyz=float((v64 * yc[None, :, None] * zc[None, None, :]).sum()),
        fSumw2=None,
        fXaxis=_axis("xaxis", xedges, xtitle), fYaxis=_axis("yaxis", yedges, ytitle),
        fZaxis=_axis("zaxis", zedges, ztitle),
    )


def tgraph(name: str, x, y, title: str = "Graph"):
    """A ``TGraph``, assembled from uproot's read model (it has no writer)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"tgraph {name}: x and y differ in length ({x.shape} vs {y.shape})")

    tnamed = _blank(Model_TNamed, 1, {"fName": name, "fTitle": title},
                    [_blank(Model_TObject, 1, {"@fUniqueID": 0, "@fBits": 0})])
    # fFunctions is a real (empty) TList rather than a null pointer: ROOT's
    # TGraph copy constructor — which is how fiTQun loads these — dereferences it.
    functions = _blank(Model_TList, 5, {"fName": "", "fSize": 0, "fLast": -1},
                       [_blank(Model_TObject, 1, {"@fUniqueID": 0, "@fBits": 0})])
    functions._data = []
    functions._options = {}

    return _blank(Model_TGraph_v4, 4, {
        "fNpoints": len(x), "fX": x, "fY": y,
        "fFunctions": functions, "fHistogram": None,
        "fMinimum": -1111.0, "fMaximum": -1111.0,
    }, [tnamed,
        _blank(Model_TAttLine_v1, 1, {"fLineColor": 1, "fLineStyle": 1, "fLineWidth": 1}),
        _blank(Model_TAttFill_v1, 1, {"fFillColor": 1, "fFillStyle": 1001}),
        _blank(Model_TAttMarker_v2, 2, {"fMarkerColor": 1, "fMarkerStyle": 1, "fMarkerSize": 1.0})])


def write(path, objects: dict, compression: Optional[object] = None) -> None:
    """Write ``{key: object}`` to ``path``, replacing any existing file."""
    with uproot.recreate(path, compression=compression) as f:
        for key, obj in objects.items():
            f[key] = obj
