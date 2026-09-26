"""Every tuning constant pinned to the reference artefact it came from.

These are not behavioural tests. Each number below was taken from a specific
line of the fiTQun tuning chain, and the failure mode they guard against is a
plausible-looking value drifting in with no reference behind it -- which has
happened, and is invisible until a table is already built on it. The citation in
each assertion is the point of the test; the value is just what it says.

Reference tree: WCSim_v1.12.15/fiTQunTuning/Utilities (``cprofile/*`` live in
that repo's git history, not its unpacked working tree).
"""
import numpy as np

from lucid.production.fitqun import binning, cprofile, scattable


# --- scattering table: scatTableLooper.C -------------------------------------

def test_scattable_bin_counts():
    """scatTableLooper.C:215-223 and the constructions at 282-296.

    nzbinss=35, nrbinss=16, nangbins=nctbins=16, and dimension 2 is the PMT
    coordinate: nzbinst=35 over (zmin, zmax) on the barrel, nrbinst=16 over
    (0, rmax) on a cap. mPMT halves that axis to 16 / 8.
    """
    assert scattable.NBINS_SIDE == (35, 16, 35, 16, 16, 16)
    assert scattable.NBINS_CAP == (35, 16, 16, 16, 16, 16)
    assert scattable.NBINS_SIDE_MPMT == (35, 16, 16, 16, 16, 16)
    assert scattable.NBINS_CAP_MPMT == (35, 16, 8, 16, 16, 16)
    # TScatTable's own cap; exceeding it aborts the C++ side at load.
    for nbins in (scattable.NBINS_SIDE, scattable.NBINS_CAP,
                  scattable.NBINS_SIDE_MPMT, scattable.NBINS_CAP_MPMT):
        assert max(nbins) <= 50


def test_scattable_axis_bounds_are_the_pmt_enclosed_volume():
    """rmax = cylRadius - tuberadius, zmax = tubezpos - tuberadius (:161-163),
    with the 1.00001 angular padding the constructions carry."""
    b = scattable.axis_bounds("sidescattable", det_radius_cm=400.0,
                              det_halfheight_cm=500.0, pmt_radius_cm=10.0)
    assert b[0] == (-490.0, 490.0)        # source z
    assert b[1] == (0.0, 390.0)           # source r
    assert b[2] == (-490.0, 490.0)        # barrel PMT z
    cap = scattable.axis_bounds("topscattable", det_radius_cm=400.0,
                                det_halfheight_cm=500.0, pmt_radius_cm=10.0)
    assert cap[2] == (0.0, 390.0)         # cap PMT r
    assert b[4] == (-1.00001, 1.00001)
    assert np.isclose(b[3][1], np.pi * 1.00001)

    # fiTQun.cc's live GetScatRatio selects the table by PMT orientation.
    assert scattable.SURFACE_DIRZ_CUT == 0.8


# --- Cherenkov profile: cprofile/integcprofile.cc and genhist.cc -------------

def test_integral_table_grid():
    """integcprofile.cc: nR0bin=401, nth0bin=201, R0max=5000, with
    R0binEdgs[i]=i*R0max/(nR0bin-1) and th0binEdgs[i]=i*2/(nth0bin-1)-1."""
    assert (binning.N_R0_POINTS, binning.N_COSTH0_POINTS) == (401, 201)
    r0, c0 = binning.r0_points(), binning.costh0_points()
    assert (r0[0], r0[-1]) == (0.0, 5000.0) and np.isclose(r0[1] - r0[0], 12.5)
    assert (c0[0], c0[-1]) == (-1.0, 1.0) and np.isclose(c0[1] - c0[0], 0.01)

    # The closing edge the shipped tables carry: 401 bins to 5012.5, 201 to 1.01.
    r0_ax = cprofile._edges_from_low(r0)
    c0_ax = cprofile._edges_from_low(c0)
    assert len(r0_ax) == 402 and np.isclose(r0_ax[-1], 5012.5)
    assert len(c0_ax) == 202 and np.isclose(c0_ax[-1], 1.01)
    # The momentum axis closes at +1 instead (mombEdgs[nmom]=mombEdgs[nmom-1]+1).
    mom_ax = cprofile._edges_from_low(np.array([120.0, 9900.0, 10000.0]), close=1.0)
    assert mom_ax[-1] == 10001.0


def test_emission_profile_axes():
    """The 2.5 cm s bin width is pinned by genhist.cc reporting s_max as a bin
    centre together with the shipped gsthr values (all odd multiples of 1.25)."""
    s = binning.s_edges()
    assert (binning.S_MIN_CM, binning.S_MAX_CM, binning.N_S_BINS) == (-500.0, 5000.0, 2200)
    assert np.isclose(s[1] - s[0], 2.5)
    # Every bin centre is then an odd multiple of 1.25, as gsthr requires.
    centres = 0.5 * (s[1:] + s[:-1])
    assert np.all(np.abs(np.round(centres / 1.25) % 2) == 1)
    assert binning.N_COSTH_BINS == 500


def test_smax_quantile():
    """genhist.cc: ``if (Itmp>0.9) { thrs=GetBinCenter(i); break; }``."""
    assert cprofile._SMAX_QUANTILE == 0.90

    # A flat emitter over [0, 100] cm: the 90% point is the bin holding 90 cm,
    # reported at its centre.
    edges = np.linspace(0.0, 200.0, 81)          # 2.5 cm bins
    marginal = np.where(0.5 * (edges[1:] + edges[:-1]) < 100.0, 1.0, 0.0)
    smax = cprofile._smax_from_hist(marginal, edges, cprofile._SMAX_QUANTILE)
    assert np.isclose(smax, 88.75)               # bin [87.5, 90) centre
    assert np.isclose(np.round(smax / 1.25) % 2, 1)

    # On the real axis, 0 is a bin edge, so light emitted within the first
    # 2.5 cm reports s_max = 1.25 -- which is what the shipped e- gsthr carries
    # for p = 1..5 MeV/c, the shortest tracks in the whole tune.
    real = binning.s_edges()
    assert 0.0 in real
    point = np.zeros(len(real) - 1)
    point[np.searchsorted(real, 0.0, side="right") - 1] = 1.0
    assert cprofile._smax_from_hist(point, real, cprofile._SMAX_QUANTILE) == 1.25


def test_cprofile_momentum_grids_are_per_particle():
    """Recovered from each CProf_<pdg>_fit_WCSim.root's gNphot. The starts are
    the Cherenkov threshold, fiTQun_shared::pCherenkovThr with nphase=1.334."""
    expected = {11: (659, 1.0), 13: (551, 120.0), 211: (532, 156.0)}
    for pdg, (n, first) in expected.items():
        mom = binning.cprofile_momenta(pdg)
        assert len(mom) == n, pdg
        assert mom[0] == first and mom[-1] == 10000.0
        assert np.all(np.diff(mom) > 0)

    # The electron grid is the muon grid plus 108 points below its threshold --
    # the cells a single shared grid would silently drop.
    e, mu = binning.cprofile_momenta(11), binning.cprofile_momenta(13)
    assert set(mu).issubset(set(e))
    assert sum(1 for m in e if m < 120.0) == 108

    # CprofileMomRepList.dat is the muon grid, and only the muon grid.
    reps_mom, _ = binning.cprofile_momentum_reps()
    assert np.array_equal(reps_mom, mu)


# --- charge PDF: chrgpdf/mutbl.txt and qbins_sk1.txt ------------------------

def test_charge_pdf_grids():
    mu = binning.charge_mu_grid()
    assert len(mu) == 202 and mu[0] == 0.1 and mu[-1] == 1090.0

    # makeChargePDFplot.C breaks its read loop on 1500 *before* incrementing, so
    # nqbins=480 while qbinEdg[480]=1500 still reaches the TH1D constructor.
    q = binning.charge_q_edges()
    assert len(q) == 481 and q[0] == 0.0 and q[-1] == 1500.0
    assert np.all(np.diff(q) > 0)


# --- time PDF and angular response ------------------------------------------

def test_timepdf_axes():
    """makehistWCSim.cc: TH2D("htimepdf","",400,-100,100,125,-2.,3.)."""
    assert (binning.TPDF_T_MIN, binning.TPDF_T_MAX, binning.TPDF_N_T_BINS) == (-100.0, 100.0, 400)
    assert (binning.TPDF_LOGMU_MIN, binning.TPDF_LOGMU_MAX,
            binning.TPDF_N_LOGMU_BINS) == (-2.0, 3.0, 125)


def test_angular_response_shells():
    """angularResponsePlotter_v1.C: nBins=25 over [0,1]; the !isNuPRISM branch
    sets dr=50 and r=100..1500 step 100, the other 50..300 step 50."""
    assert binning.ANGRESP_N_BINS == 25
    assert binning.ANGRESP_SHELL_DR_CM == 50.0
    assert binning.ANGRESP_SHELL_RADII_CM == tuple(float(r) for r in range(100, 1501, 100))
    assert binning.ANGRESP_SHELL_RADII_NUPRISM_CM == tuple(float(r) for r in range(50, 301, 50))
    e = binning.angresp_edges()
    assert (e[0], e[-1], len(e)) == (0.0, 1.0, 26)
