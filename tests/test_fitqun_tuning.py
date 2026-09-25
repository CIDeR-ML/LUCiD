"""fiTQun tuning-input generators: file formats and the integral-table numerics.

Two things are worth pinning here. The ROOT writers have to produce objects
fiTQun's C++ can actually read, so they are round-tripped through uproot's
reader (an independent implementation of the same format). And the Cherenkov
profile's integral tables are a numerical integration with a sign convention
that is easy to get backwards, so they are checked against cases with a closed
form.
"""
import numpy as np
import pytest

from lucid.production.fitqun import angular, binning, cprofile, rootio, scattable, timepdf

uproot = pytest.importorskip("uproot")


# --- ROOT writers ------------------------------------------------------------

def test_histograms_round_trip(tmp_path):
    """Values, variable bin edges and axis order survive a write/read cycle."""
    xe = np.array([0.0, 1.0, 2.0, 4.0, 8.0])      # deliberately non-uniform
    ye = np.linspace(-1.0, 1.0, 4)
    ze = np.array([0.0, 10.0, 100.0])
    v1 = np.array([1.0, 2.0, 3.0, 4.0])
    v2 = np.arange(12.0).reshape(4, 3)
    v3 = np.arange(24.0).reshape(4, 3, 2)

    path = tmp_path / "hists.root"
    rootio.write(path, {
        "h1": rootio.th1("h1", xe, v1, sumw2=v1),
        "h2": rootio.th2("h2", xe, ye, v2),
        "h3": rootio.th3("h3", xe, ye, ze, v3),
    })

    with uproot.open(path) as f:
        assert f.classnames() == {"h1;1": "TH1D", "h2;1": "TH2D", "h3;1": "TH3F"}
        np.testing.assert_allclose(f["h1"].values(), v1)
        np.testing.assert_allclose(f["h1"].axis().edges(), xe)
        np.testing.assert_allclose(f["h1"].errors(), np.sqrt(v1))
        # A transposed write would still round-trip a square array; these are not.
        np.testing.assert_allclose(f["h2"].values(), v2)
        np.testing.assert_allclose(f["h3"].values(), v3)
        np.testing.assert_allclose(f["h3"].axis(2).edges(), ze)


def test_tgraph_round_trip(tmp_path):
    """uproot has no TGraph writer; rootio.tgraph builds the model by hand."""
    x = np.array([1.0, 2.5, 7.0])
    y = np.array([10.0, -3.0, 0.5])
    path = tmp_path / "graph.root"
    rootio.write(path, {"g": rootio.tgraph("g", x, y, "yield")})

    with uproot.open(path) as f:
        g = f["g"]
        assert f.classnames() == {"g;1": "TGraph"}
        assert g.member("fNpoints") == 3
        np.testing.assert_allclose(g.values("x"), x)
        np.testing.assert_allclose(g.values("y"), y)
        # fiTQun loads these via TGraph's copy constructor, which dereferences
        # fFunctions -- a null there would crash it.
        assert g.member("fFunctions") is not None


def test_tgraph_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        rootio.tgraph("g", [1.0, 2.0], [1.0])


# --- Cherenkov profile -------------------------------------------------------

def _uniform_cell(pdg=13, momentum=1000.0, s_max=100.0, n_s=200, n_c=200,
                  s_lo=0.0):
    """A profile flat in s over [s_lo, s_max] and flat in cos(theta)."""
    s_edges = np.linspace(s_lo, s_max, n_s + 1)
    c_edges = np.linspace(-1.0, 1.0, n_c + 1)
    density = np.full((n_s, n_c), 1.0 / ((s_max - s_lo) * 2.0))
    return cprofile.ProfileCell(
        pdg=pdg, momentum_mev=momentum, s_edges=s_edges, costh_edges=c_edges,
        density=density, s_max_cm=s_max, n_photons=1234.0, n_events=10)


def test_integral_tables_against_closed_form():
    """For a profile flat in both variables, I_n has a closed form.

    g = 1/(2 s_max) everywhere, so I_n = int_0^smax s^n /(2 s_max) ds
    = s_max^n / (2 (n+1)) independently of R0 and cos(theta0).
    """
    cell = _uniform_cell(s_max=100.0)
    # A coarse grid keeps the closed-form check fast; the production grid is
    # 401 x 201 (binning.r0_points / costh0_points).
    I, iso1, iso2, r0, c0, mom = cprofile.integral_tables(
        [cell], r0=np.linspace(0.0, 5000.0, 21), costh0=np.linspace(-1.0, 1.0, 21))

    assert I.shape == (3, len(r0), len(c0), 1)
    expected = [100.0**n / (2.0 * (n + 1)) for n in range(3)]
    for n, want in enumerate(expected):
        np.testing.assert_allclose(I[n, :, :, 0], want, rtol=2e-3)

    # The isotropic moments use the s marginal, which integrates to 1, so they
    # are the plain moments of a uniform distribution on [0, s_max].
    np.testing.assert_allclose(iso1[0], 100.0 / 2.0, rtol=2e-3)
    np.testing.assert_allclose(iso2[0], 100.0**2 / 3.0, rtol=2e-3)


def test_integral_tables_pick_the_right_angle_slice():
    """I_n follows cos(theta(s)) along the line of sight, not cos(theta0).

    Concentrate the profile in the most-forward cos(theta) bin. A PMT far
    downstream of the vertex (cos theta0 = +1, R0 >> s_max) is looked at from
    every point of the track at cos(theta(s)) = +1, so it collects the whole
    profile; one directly upstream (cos theta0 = -1) is at cos(theta(s)) = -1
    throughout and collects none of it. Getting the sign of
    cos(theta) = (R0 cos theta0 - s)/R backwards swaps the two.
    """
    s_max, n_s, n_c = 100.0, 100, 200
    c_edges = np.linspace(-1.0, 1.0, n_c + 1)
    s_edges = np.linspace(0.0, s_max, n_s + 1)
    ds, dc = s_edges[1] - s_edges[0], c_edges[1] - c_edges[0]
    density = np.zeros((n_s, n_c))
    density[:, -1] = 1.0            # cos(theta) in the top bin only
    density /= density.sum() * ds * dc
    cell = cprofile.ProfileCell(pdg=13, momentum_mev=500.0, s_edges=s_edges,
                                costh_edges=c_edges, density=density,
                                s_max_cm=s_max, n_photons=1.0, n_events=1)

    # Probe R0 = 5000 cm at cos theta0 = -1 (index 0) and +1 (index 1).
    I, _, _, _, _, _ = cprofile.integral_tables(
        [cell], r0=np.array([5000.0]), costh0=np.array([-1.0, 1.0]))

    upstream = I[0, 0, 0, 0]
    downstream = I[0, 0, 1, 0]
    assert upstream == pytest.approx(0.0, abs=1e-12)
    # All the density sits in one cos(theta) bin of width dc, so integrating it
    # along a line of sight that stays inside that bin returns 1/dc.
    assert downstream == pytest.approx(1.0 / dc, rel=1e-9)


def test_cprofile_file_has_everything_fitqun_loads(tmp_path):
    """fiTQun_shared::LoadProfiles reads these objects by name; all must exist."""
    cells = [_uniform_cell(momentum=p, s_max=0.2 * p, n_s=20, n_c=20)
             for p in (200.0, 500.0, 1000.0)]
    path = tmp_path / "CProf_13_WCSim.root"
    cprofile.write_cprofile(path, 13, cells)

    with uproot.open(path) as f:
        names = {k.split(";")[0] for k in f.keys()}
        assert {"hI3d_0", "hI3d_1", "hI3d_2", "hI_iso_1", "hI_iso_2",
                "gNphot", "gsthr"} <= names
        assert f.classnames()["hI3d_0;1"] == "TH3F"   # LoadProfiles casts to TH3F
        # The momentum axis is read off bin low edges, so those must be the
        # momenta the tables were evaluated at.
        np.testing.assert_allclose(f["hI3d_0"].axis(2).edges()[:3], [200.0, 500.0, 1000.0])
        np.testing.assert_allclose(f["gsthr"].values("y"), [40.0, 100.0, 200.0])


def test_profile_cell_round_trips_and_merges(tmp_path):
    a = _uniform_cell()
    a.n_events = 10
    b = _uniform_cell()
    b.n_events = 30
    b.n_photons = 2234.0

    path = a.save(tmp_path / "cell.npz")
    back = cprofile.ProfileCell.load(path)
    np.testing.assert_allclose(back.density, a.density)
    assert back.s_max_cm == a.s_max_cm

    merged = a + b
    assert merged.n_events == 40
    # Event-weighted, not a plain mean.
    assert merged.n_photons == pytest.approx((1234.0 * 10 + 2234.0 * 30) / 40)


# --- angular response --------------------------------------------------------

def test_cos_eta_is_one_at_normal_incidence():
    source = np.array([[0.0, 0.0, 100.0], [0.0, 70.7, 70.7]])
    sensor = np.zeros((2, 3))
    axis = np.tile([0.0, 0.0, 1.0], (2, 1))
    R, c = angular.cos_eta(source, sensor, axis)
    np.testing.assert_allclose(R, [100.0, 99.98], rtol=1e-3)
    np.testing.assert_allclose(c, [1.0, 0.7071], rtol=1e-3)


def test_angular_measure_keeps_only_the_shell(tmp_path):
    """Photons outside [r-dr, r+dr] must not enter the histogram."""
    n = 6
    sensor = np.tile([0.0, 300.0, 0.0], (n, 1))          # barrel sensor
    axis = np.tile([0.0, -1.0, 0.0], (n, 1))             # facing inward
    radii = np.array([50.0, 95.0, 100.0, 105.0, 150.0, 400.0])
    source = sensor + axis * radii[:, None]              # head-on at each radius

    edges, counts, sumw2 = angular.measure(
        source, sensor, axis, shell_r_cm=100.0, shell_dr_cm=10.0,
        det_radius_cm=400.0, det_halfheight_cm=500.0, n_bins=25)
    assert counts.sum() == 3                              # 95, 100, 105
    assert counts[-1] == 3                                # all at cos eta = 1

    values, _ = angular.normalise(counts, sumw2)
    assert values[-1] == pytest.approx(1.0)

    out = angular.write_angular_response(tmp_path / "ang.root", edges, counts,
                                         sumw2, shell_r_cm=100.0)
    with uproot.open(out) as f:
        assert "angRespAll_100;1" in f.keys()


def test_angular_rejects_inconsistent_detector_extent():
    sensor = np.array([[0.0, 0.0, 0.0]])                  # nowhere near a surface
    axis = np.array([[0.0, 0.0, 1.0]])
    source = np.array([[0.0, 0.0, 100.0]])
    with pytest.raises(ValueError, match="clear of both"):
        angular.measure(source, sensor, axis, shell_r_cm=100.0, shell_dr_cm=10.0,
                        det_radius_cm=4000.0, det_halfheight_cm=5000.0)


# --- time PDF ----------------------------------------------------------------

def test_corrected_time_is_zero_for_light_from_the_midpoint():
    """A photon emitted at the track midpoint travelling straight to the PMT
    at the group velocity has, by construction, t_c = 0."""
    vertex = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    s_max, n_water = 200.0, 1.38
    pmt = np.array([[300.0, 0.0, 100.0]])

    mid = vertex + 0.5 * s_max * direction
    r_mid = np.linalg.norm(pmt[0] - mid)
    t_hit = 0.5 * s_max / timepdf.C_CM_PER_NS + r_mid * n_water / timepdf.C_CM_PER_NS

    tc = timepdf.corrected_time(np.array([t_hit]), pmt, vertex_cm=vertex,
                                direction=direction, s_max_cm=s_max, n_water=n_water)
    np.testing.assert_allclose(tc, [0.0], atol=1e-9)


def test_timepdf_accumulator_bins_and_merges(tmp_path):
    acc = timepdf.TimePdfAccumulator()
    acc.fill(np.array([0.0, 10.0, 10.0]), np.array([1.0, 100.0, 0.0]))
    # mu = 0 has no log10 and is dropped, so two hits survive.
    assert acc.counts.sum() == 2

    other = timepdf.TimePdfAccumulator()
    other.fill(np.array([0.0]), np.array([1.0]))
    merged = acc + other
    assert merged.counts.sum() == 3
    assert merged.n_events == 2

    out = merged.write(tmp_path / "cell_hist.root")
    with uproot.open(out) as f:
        assert f.classnames() == {"htimepdf;1": "TH2D"}
        h = f["htimepdf"]
        assert h.axis(0).low == -100.0 and h.axis(0).high == 100.0
        assert h.axis(1).low == -2.0 and h.axis(1).high == 3.0


def test_cell_name_matches_reference_layout():
    assert timepdf.cell_name(11, 1000.0, 3) == "11_1000_0_3_0"


# --- scattering tables -------------------------------------------------------

def test_scattable_index_order_is_dimension_zero_fastest():
    """TScatTable::GetIndex walks dimension 0 fastest; flat() must match it."""
    nbins = (2, 3, 1, 1, 1, 1)
    bounds = ((0.0, 2.0), (0.0, 3.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    t = scattable.ScatTable("side", nbins, bounds)
    t.table[1, 2, 0, 0, 0, 0] = 7.0
    # index = i_zs + n_zs * i_rs = 1 + 2*2 = 5
    assert t.flat()[5] == 7.0


def test_scattable_names_match_fitqun_lookup():
    """fiTQun_shared.cc looks these up literally; short names return null."""
    assert scattable.SURFACES == ("topscattable", "botscattable", "sidescattable")


def test_scattable_surface_split_is_by_orientation():
    """fiTQun's live GetScatRatio cuts on PMTdir_z, not PMT position."""
    dirs = np.array([-0.95, -0.5, 0.0, 0.5, 0.95])
    got = list(scattable.surface_for(dirs))
    assert got == ["topscattable", "sidescattable", "sidescattable",
                   "sidescattable", "botscattable"]


def test_scattable_axis_bounds_follow_the_reference_formula():
    side = scattable.axis_bounds("sidescattable", det_radius_cm=1690.0,
                                 det_halfheight_cm=1810.0, pmt_radius_cm=25.4)
    cap = scattable.axis_bounds("topscattable", det_radius_cm=1690.0,
                                det_halfheight_cm=1810.0, pmt_radius_cm=25.4)
    # source axes are the PMT-enclosed volume
    assert side[0] == pytest.approx((-1784.6, 1784.6))
    assert side[1] == pytest.approx((0.0, 1664.6))
    # the PMT axis follows the surface: z on the barrel, radius on a cap
    assert side[2] == pytest.approx((-1784.6, 1784.6))
    assert cap[2] == pytest.approx((0.0, 1664.6))
    # angles carry the reference's 1.00001 padding
    assert side[4] == pytest.approx((-1.00001, 1.00001))
    assert side[3][1] == pytest.approx(np.pi * 1.00001)


def test_scattable_ratio_is_6d_over_spread_4d():
    """DivideUnnormalized4D: direct light is 4D and spread over the
    n_ct*n_phi direction cells before dividing -- not an elementwise 6D divide."""
    nb = (2, 1, 1, 1, 2, 2)                      # 4 direction cells
    bd = ((0.0, 2.0),) + ((0.0, 1.0),) * 5
    scattered = scattable.ScatTable("s", nb, bd)
    direct = scattable.ScatTable("d", nb, bd)
    scattered.table[0, 0, 0, 0] = 1.0            # 1 in every direction cell
    direct.table[0, 0, 0, 0] = 2.0               # 2 each -> 8 total direct
    ratio = scattered.ratio_to(direct)
    # direct4d = 8/4 = 2 per cell, so the ratio is 1/2 everywhere in that bin
    np.testing.assert_allclose(ratio.table[0, 0, 0, 0], 0.5)
    # a bin with no direct light is zeroed, not infinite
    assert np.all(ratio.table[1] == 0.0)


def test_scattable_fill_and_ratio():
    nbins = (4, 1, 1, 1, 1, 1)
    bounds = ((0.0, 4.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    direct = scattable.ScatTable("direct", nbins, bounds)
    scattered = scattable.ScatTable("scattered", nbins, bounds)
    zeros = np.zeros(3)
    direct.fill(np.array([0.5, 1.5, 2.5]), zeros, zeros, zeros, zeros, zeros,
                weights=np.array([10.0, 20.0, 40.0]))
    scattered.fill(np.array([0.5, 1.5, 2.5]), zeros, zeros, zeros, zeros, zeros,
                   weights=np.array([1.0, 2.0, 0.0]))

    # One direction cell, so the 4D spread is a no-op and the ratio is plain.
    ratio = scattered.ratio_to(direct)
    np.testing.assert_allclose(ratio.table[:3, 0, 0, 0, 0, 0], [0.1, 0.1, 0.0])
    # An empty direct bin gives 0, not a division by zero.
    assert ratio.table[3, 0, 0, 0, 0, 0] == 0.0


def test_scattable_rejects_too_many_bins():
    with pytest.raises(ValueError, match="at most 50 bins"):
        scattable.ScatTable("side", (51, 1, 1, 1, 1, 1), ((0.0, 1.0),) * 6)


def test_scattable_hdf5_round_trip(tmp_path):
    nbins = (3, 2, 1, 1, 1, 1)
    bounds = ((-100.0, 100.0), (0.0, 50.0)) + ((0.0, 1.0),) * 4
    t = scattable.ScatTable("side", nbins, bounds)
    t.table[:] = np.arange(6.0).reshape(nbins)

    path = scattable.write_hdf5(tmp_path / "scat.h5", {"side": t})
    back = scattable.read_hdf5(path)["side"]
    np.testing.assert_allclose(back.table, t.table)
    assert back.nbins == nbins
    assert back.bounds == bounds


# --- binning provenance ------------------------------------------------------

def test_reference_grids_load():
    mom, reps = cprofile.binning.cprofile_momenta()
    assert mom[0] > 0 and np.all(np.diff(mom) > 0) and reps.sum() > 0

    mu = binning.charge_mu_grid()
    assert mu[0] == 0.1 and np.all(np.diff(mu) > 0)

    # 1500 is the real final edge: makeChargePDFplot.C breaks its read loop on
    # it before incrementing, so nqbins=480 but qbinEdg[480]=1500 is still used.
    q = binning.charge_q_edges()
    assert q[0] == 0.0 and q[-1] == 1500.0
    assert len(q) - 1 == 480
    assert np.all(np.diff(q) > 0)

    # gen2d.cc opens files by mutbl.txt's literal tokens, so "1.0" must survive.
    labels = binning.charge_mu_labels()
    assert len(labels) == len(mu) and "1.0" in labels and "0.1" in labels

    for pdg in (11, 13, 211):
        assert len(binning.timepdf_momenta(pdg)) > 40


def test_kinetic_energy_matches_the_reference_macro():
    """Pins the p -> kinetic-energy convention against the reference tune.

    The generated macro for the e-, p = 1000 MeV/c time-PDF cell carries
    ``/gps/energy 999.48913056049147697903 MeV``. That is sqrt(p^2+m^2) - m
    with the rounded m = 0.511 MeV those scripts used; we carry the PDG value,
    so the convention must agree exactly and the numbers only to the precision
    the mass difference allows.
    """
    m_reference = 0.511
    exact = np.sqrt(1000.0**2 + m_reference**2) - m_reference
    assert exact == pytest.approx(999.48913056049147697903, rel=1e-15)
    assert binning.kinetic_energy_mev(11, 1000.0) == pytest.approx(exact, rel=1e-8)


# --- PhotonSim macro generation ----------------------------------------------

def test_profile_macro_never_disables_cherenkov():
    """The whole reason macros.py names processes instead of numbering them.

    PhotonSim's mu- process table has Cerenkov at index 8, which is where
    WCSim's has muon capture; a macro carrying WCSim's numbers deletes the
    muon's own light. Nothing generated here may switch Cerenkov off.
    """
    from lucid.production.fitqun import macros

    for pdg in (11, 13, 211):
        text = macros.profile_macro(pdg=pdg, momentum_mev=1000.0,
                                    output_path="out.root", n_events=10)
        assert "Cerenkov" not in text
        assert "/particle/process/inactivate" not in text   # index-based form
        assert "/gun/momentumAmp 1000 MeV" in text
        assert f"/gun/particle {binning.PDG_NAMES[pdg]}" in text

    muon = macros.profile_macro(pdg=13, momentum_mev=500.0,
                                output_path="out.root", n_events=10)
    assert "/process/inactivate Decay" in muon
    assert "/process/inactivate muMinusCaptureAtRest" in muon


def test_profile_macro_rejects_unknown_particle():
    from lucid.production.fitqun import macros
    with pytest.raises(ValueError, match="no fiTQun hypothesis"):
        macros.profile_macro(pdg=2212, momentum_mev=1000.0,
                             output_path="out.root", n_events=10)


# --- charge PDF ---------------------------------------------------------------

def test_charge_pdf_occupancy_follows_poisson_times_the_discriminator(tmp_path):
    """P(hit|mu) must sit *below* the Poisson curve by the discriminator loss.

    A PMT fires if it gets at least one photoelectron AND the digitised charge
    clears the threshold. If the cut were applied to the photoelectron count
    instead, it would be inert and this would come out exactly 1 - exp(-mu) --
    which is the bug this pins against.
    """
    from lucid.production.fitqun import chargepdf
    from lucid.simulation.digitizer import _sample_spe_charge, resolve_model_config

    model = resolve_model_config("ski")
    rng = np.random.default_rng(11)
    for mu in (0.5, 5.0):
        # Independent expectation: Poisson occupancy times the SPE survival
        # fraction at that photoelectron multiplicity.
        n = rng.poisson(mu, 400_000)
        q = np.zeros(n.size)
        q[n > 0] = _sample_spe_charge(n[n > 0].astype(float), model["spe"], rng)
        expected = (q >= model["threshold_pe"]).mean()
        assert expected < 1.0 - np.exp(-mu)          # the cut really bites

        res = chargepdf.build(mu, n_pmt=4000, n_events=8, model="ski", seed=7)
        assert res["n_hits"] / res["n_active"] == pytest.approx(expected, abs=0.01)

    # Charges below the discriminator must not appear in the table at all.
    path = chargepdf.write_mu_point(tmp_path, 0.5, n_pmt=4000, n_events=8,
                                    model="ski", seed=7)
    with uproot.open(path) as f:
        h = f["hchpdf2"]
        edges, values = h.axis().edges(), h.values()
        assert values[edges[1:] <= model["threshold_pe"]].sum() == 0
        ctr = f["hctr"].values()
        assert ctr[1] == ctr[2] == 4000 * 8 and ctr[9] == ctr[1] + ctr[2]


def test_charge_pdf_batching_preserves_the_distribution():
    """Batching bounds memory. It redraws rather than replaying, so the check
    is that both give the same occupancy and mean, not the same draws."""
    from lucid.production.fitqun import chargepdf

    mu, n_pmt, n_events = 2.0, 500, 20
    model = chargepdf.resolve_model_config("ski")

    one_pass = chargepdf.sample_charges(mu, n_pmt, n_events,
                                        model, np.random.default_rng(3))
    original = chargepdf._MAX_PE_PER_PASS
    try:
        chargepdf._MAX_PE_PER_PASS = 2000       # forces several batches
        batched = chargepdf.sample_charges(mu, n_pmt, n_events,
                                           model, np.random.default_rng(3))
    finally:
        chargepdf._MAX_PE_PER_PASS = original

    assert batched.size == pytest.approx(one_pass.size, rel=0.05)
    assert batched.mean() == pytest.approx(one_pass.mean(), rel=0.05)
    assert (batched >= model["threshold_pe"]).all()


def test_charge_pdf_rejects_a_model_without_an_spe_spectrum():
    from lucid.production.fitqun import chargepdf
    with pytest.raises(ValueError, match="no physical SPE charge"):
        chargepdf.build(1.0, n_pmt=10, n_events=1, model="basic", seed=0)


def test_charge_pdf_filenames_match_the_reference_spelling(tmp_path):
    """gen2d.cc opens '<mu>_pdf.root' using the literal text of mutbl.txt.

    Nine entries are whole numbers written as "1.0".."9.0"; formatting the
    float gives "1".."9" and gen2d.cc then opens a missing file and
    dereferences the null TFile.
    """
    from lucid.production.fitqun import chargepdf

    for label in ("1.0", "0.1", "10", "1090"):
        p = chargepdf.write_mu_point(tmp_path, float(label), label=label,
                                     n_pmt=20, n_events=1, model="ski", seed=0)
        assert p.name == f"{label}_pdf.root"

    # Every label the scan would emit must round-trip to a real mutbl.txt token.
    labels = set(binning.charge_mu_labels())
    assert {"1.0", "2.0", "9.0"} <= labels


# --- cluster fan-out ----------------------------------------------------------

def test_command_job_exports_env_for_the_whole_chain():
    """`A=1 cmd1 && cmd2` scopes A to cmd1 in bash, which silently broke the
    reduction step of the profile job: PhotonSim ran, then the `python -m
    lucid...` that follows it came up without PYTHONPATH. The dev-checkout
    overrides must be exported, not prefixed."""
    from lucid.production.cluster_common.htcondor import HTCondorAdapter

    adapter = HTCondorAdapter({
        "LUCID_IMAGE_PATH": "/img.sif",
        "LUCID_DEV_PATH": "/dev/LUCiD",
        "PHOTONSIM_DEV_PATH": "/dev/PhotonSim",
        "LOG_BASE_PATH": "/logs",
    })
    body = adapter.render_command_job(
        command="first && second", cell_dir=__import__("pathlib").Path("/out/cell"),
        job_name="j", log_stem="s", partition="")

    args = next(l for l in body.splitlines() if l.startswith("arguments"))
    assert "export PYTHONPATH=/dev/LUCiD" in args
    assert "PHOTONSIM_BIN=/dev/PhotonSim/build/PhotonSim;" in args
    assert args.index("export") < args.index("first && second")
    # HTCondor's arguments is itself a quoted string; an embedded double quote
    # would need doubling, so the command must not carry one.
    assert '"' not in args[args.index("-c"):-1]


def test_profile_cell_command_has_no_nested_quotes():
    import pathlib
    from lucid.production.jobs.fitqun import generate_jobs

    cmd = generate_jobs.cell_command(cell_dir=pathlib.Path("/out/cell"),
                                     pdg=13, momentum=1200.0)
    assert '"' not in cmd and "'" not in cmd
    # The raw ROOT file is only removed once the reduced cell exists.
    assert cmd.index("cprofile accumulate") < cmd.index("rm -f")


def test_events_schedule_ladder():
    from lucid.production.jobs.fitqun.generate_jobs import events_for
    schedule = {"500": 400, "1500": 200, "1e9": 50}
    assert events_for(300, schedule) == 400
    assert events_for(500, schedule) == 400      # inclusive upper bound
    assert events_for(501, schedule) == 200
    assert events_for(9000, schedule) == 50


# --- angular-response driver --------------------------------------------------

def test_sensor_axes_point_inward():
    """A sensor's axis must face into the water, or every cos(eta) flips sign."""
    from lucid.production.fitqun import angular_driver as ad

    R, H = 1690.0, 1810.0
    pos = np.array([
        [R, 0.0, 0.0],        # barrel, +x wall  -> axis -x
        [0.0, -R, 0.0],       # barrel, -y wall  -> axis +y
        [100.0, 0.0, H],      # top cap          -> axis -z
        [100.0, 0.0, -H],     # bottom cap       -> axis +z
    ])
    axes = ad.sensor_axes(pos, det_radius_cm=R, det_halfheight_cm=H)
    np.testing.assert_allclose(axes[0], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(axes[1], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(axes[2], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(axes[3], [0, 0, 1], atol=1e-6)


def test_angular_driver_keeps_only_direct_light():
    """The reference skips isct != 0; here that is the indirect flag."""
    from lucid.production.fitqun import angular_driver as ad

    R, H = 1690.0, 1810.0
    # Positions go in as meters (LUCiD's convention); the cm grid args are
    # fiTQun's and stay as they are.
    sensors = np.array([[0.0, R, 0.0]]) / 100.0
    # Four photons head-on at the shell radius; two of them scattered.
    emission = np.tile([0.0, (R - 100.0) / 100.0, 0.0], (4, 1))
    chunk = {
        "emission_pos": emission,
        "sensor_id": np.zeros(4, dtype=int),
        "detected": np.ones(4, dtype=bool),
        "indirect": np.array([False, True, False, True]),
    }
    _, counts, _ = ad.accumulate(
        [chunk], sensors, shell_r_cm=100.0, det_radius_cm=R,
        det_halfheight_cm=H)
    assert counts.sum() == 2                      # the two direct ones
    assert counts[-1] == 2                        # head-on -> top cos(eta) bin

    # Without the flag, asking for direct light is an error rather than a
    # silent pass-through of indirect photons into the tune.
    with pytest.raises(ValueError, match="indirect"):
        ad.accumulate([{**chunk, "indirect": None}], sensors, shell_r_cm=100.0,
                      det_radius_cm=R, det_halfheight_cm=H)


# --- scattering-table driver --------------------------------------------------

def test_scattable_delta_phi_wraps():
    """ast and phi are TVector3::DeltaPhi differences, wrapped to (-pi, pi]."""
    from lucid.production.fitqun import scattable_driver as sd
    a = np.array([3.0, -3.0, 0.5])
    b = np.array([-3.0, 3.0, 0.2])
    d = sd._delta_phi(a, b)
    assert np.all(np.abs(d) <= np.pi + 1e-12)
    # 3 - (-3) = 6 rad wraps to 6 - 2pi, not 6.
    np.testing.assert_allclose(d[0], 6.0 - 2 * np.pi, atol=1e-9)
    np.testing.assert_allclose(d[2], 0.3, atol=1e-9)


def test_scattable_pmt_coordinate_follows_its_surface():
    """t is the PMT z on the barrel, its distance from the axis on a cap."""
    from lucid.production.fitqun import scattable_driver as sd
    src = np.zeros((2, 3))
    sdir = np.tile([0.0, 0.0, 1.0], (2, 1))
    pmt = np.array([[300.0, 400.0, 1000.0],     # barrel -> t = z = 1000
                    [300.0, 400.0, 1800.0]])    # cap    -> t = rho = 500
    zs, rs, t, ast, ct, phi = sd.coordinates(
        src, sdir, pmt, is_cap=np.array([False, True]))
    np.testing.assert_allclose(t, [1000.0, 500.0])
    np.testing.assert_allclose(ct, [1.0, 1.0])


def test_scattable_driver_splits_direct_from_indirect():
    """One pass, split on the flag -- the reference never runs two productions."""
    from lucid.production.fitqun import scattable_driver as sd

    nb = {s: (4, 2, 2, 2, 2, 2) for s in scattable.SURFACES}
    bd = {s: ((-2000.0, 2000.0), (0.0, 2000.0), (-2000.0, 2000.0),
              (-np.pi, np.pi), (-1.0, 1.0), (-np.pi, np.pi))
          for s in scattable.SURFACES}
    tables = sd.make_tables(nb, bd)

    # fill() takes meters, as LUCiD propagation reports them.
    pmt_positions_m = np.array([[1690.0, 0.0, 0.0]]) / 100.0   # one barrel PMT
    pmt_dir_z = np.array([0.0])                          # barrel orientation
    chunk = {
        "emission_pos": np.zeros((4, 3)),
        "emission_dir": np.tile([0.0, 0.0, 1.0], (4, 1)),
        "sensor_id": np.zeros(4, dtype=int),
        "detected": np.ones(4, dtype=bool),
        "indirect": np.array([True, True, True, False]),
    }
    sd.fill(tables, chunk, pmt_positions_m=pmt_positions_m, pmt_dir_z=pmt_dir_z)

    side = tables["sidescattable"]
    assert side["scattered"].table.sum() == 3
    assert side["direct"].table.sum() == 1
    # The direct partner collapses the two direction axes -- that is what makes
    # it the 4D table DivideUnnormalized4D expects.
    assert side["direct"].nbins[4:] == (1, 1)

    ratios = sd.finalise(tables)
    assert set(ratios) == set(scattable.SURFACES)
    assert np.isfinite(ratios["sidescattable"].table).all()

    # Refuses to guess when the flag is absent.
    with pytest.raises(ValueError, match="indirect"):
        sd.fill(tables, {**chunk, "indirect": None},
                pmt_positions_m=pmt_positions_m, pmt_dir_z=pmt_dir_z)
