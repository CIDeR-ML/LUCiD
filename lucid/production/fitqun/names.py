"""fiTQun's file and parameter naming contract, in one place.

Every one of these strings was read out of ``fiTQun_shared.cc``; none of them is
documented anywhere else, and getting one wrong produces a silent fallback or a
segfault rather than an error message. They live here so a tune for a new
detector is a matter of naming it, not of re-reading the C++.

The two names a tune is identified by:

``config``
    fiTQun's ``fiTQun.WCSimConfig``. It appears bare in the angular-response and
    time-PDF file names and **underscore-prefixed** in the scattering table's,
    because ``fiTQun_shared`` builds the latter as ``Form("_%s", config)``.
``pmt_type``
    fiTQun's ``fiTQun.WCSimPMTType``, in every per-photosensor file name.

Line references are to the copy this was taken from, fiTQun v6r0.
"""
from __future__ import annotations

# fiTQun_shared.cc:59 -- the hypothesis order of every fq1r* output array, so
# the muon is index 2. Getting this wrong silently scores the wrong hypothesis.
PID_ORDER = (22, 11, 13, 211, 321, 2212, 48)

#: Particles a tune supplies Cherenkov profiles and time PDFs for.
TUNED_PDGS = (11, 13, 211)

#: Scalars fiTQun builds as ``fiTQun.<key>WCSim`` (fiTQun_shared.cc:228-330,
#: runfiTQun.cc:309). The shipped parameters file defines only the SK<n> forms,
#: so a WCSim tune must provide every one of these.
SCALAR_KEYS = (
    "PeakThr", "GainCorrFact", "WaterAttenuationLength", "DarkRate",
    "QEEff", "QEEffCorr", "QmuConvFact",
    "PIDCutEPipa0", "PIDCutEPipa1", "PIDCutMuPip0", "PIDCutMuPip1",
    "RCCut1Ea0", "RCCut1Mua0", "RCCuta0", "RCCuta1",
)


def pid_index(pdg: int) -> int:
    """Index of ``pdg`` on the hypothesis axis of fiTQun's output arrays."""
    return PID_ORDER.index(int(pdg))


def cprofile(pdg: int, *, fitted: bool = False) -> str:
    """``CProf_<pdg>[_fit]_WCSim.root`` (fiTQun_shared.cc:1536-1539, 1680-1683).

    The fitted form is a momentum fit of the raw tables; a tune that ships raw
    tables must also set ``fiTQun.UseFitCProfile = 0`` or fiTQun looks for the
    fitted file and never mentions the raw one.
    """
    return f"CProf_{int(pdg)}{'_fit' if fitted else ''}_WCSim.root"


def charge_pdf(pmt_type: str) -> str:
    """``cPDFpar_<type>.root`` (fiTQun_shared.cc:1487-1498).

    Only when ``fiTQun.WCSimConfig`` is set; otherwise fiTQun inserts an SK era
    and looks for ``cPDFpar_sk<n>_<type>.root``.
    """
    return f"cPDFpar_{pmt_type}.root"


def angular_response(config: str, pmt_type: str) -> str:
    """``angResp_<config>_<type>.root`` (fiTQun_shared.cc:651).

    Must contain a ``TF1`` named ``angResp``; ``fit_cos.C`` produces it, and the
    histogram it fits is the hardcoded ``angRespAll_100``, so the 100 cm shell is
    not optional.
    """
    return f"angResp_{config}_{pmt_type}.root"


def scattable_6d(config: str) -> str:
    """``fiTQun_scattablesF_<config>.root`` (fiTQun_shared.cc:443)."""
    return f"fiTQun_scattablesF_{config}.root"


def scattable_3d(config: str) -> str:
    """``fiTQun_scattable3d_<config>.root`` (fiTQun_shared.cc:471).

    Read only when ``fiTQun.UseScatteredLight`` is 2, 3 or 4. The shipped default
    is 4 ("6D for the 1R fit, 3D otherwise"), so a tune with only the 6D table
    must set it to 1.
    """
    return f"fiTQun_scattable3d_{config}.root"


def time_pdf(pdg: int, config: str, pmt_type: str) -> str:
    """``<pdg>_tpdfpar_<config>_<type>.root`` (fiTQun_shared.cc:1622).

    Holds ``htpdfparmn``/``htpdfparsg``. fiTQun refuses to start without it even
    though the charge prediction never reads it.
    """
    return f"{int(pdg)}_tpdfpar_{config}_{pmt_type}.root"


def scalar_key(name: str) -> str:
    """``fiTQun.<name>WCSim`` -- the detector suffix is literal, not the config."""
    if name not in SCALAR_KEYS:
        raise KeyError(f"{name} is not one of fiTQun's scalar parameters")
    return f"fiTQun.{name}WCSim"


def required_files(config: str, pmt_type: str, *, pdgs=TUNED_PDGS,
                   with_3d: bool = False) -> list:
    """Every file fiTQun opens for this tune, in the order it opens them."""
    files = [charge_pdf(pmt_type), scattable_6d(config)]
    if with_3d:
        files.append(scattable_3d(config))
    files.append(angular_response(config, pmt_type))
    for pdg in pdgs:
        files.append(cprofile(pdg))
        files.append(time_pdf(pdg, config, pmt_type))
    return files
