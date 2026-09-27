// SadQun — the part of fiTQun that predicts charge, and nothing else.
//
// Everything here is lifted from fiTQun (fiTQun.cc::SnglTrk and the
// fiTQun_shared helpers it calls), not reimplemented, so that the mu this
// produces is the mu fiTQun would produce. Where a branch is dead for our use
// it is omitted rather than ported -- see README.md for the list and why.
#ifndef SADQUN_H
#define SADQUN_H

#include <string>
#include <vector>

namespace sadqun {

// Speed of light, cm/ns -- makehistWCSim.cc's c0.
constexpr double kC0 = 29.9792458;

/// PMT geometry, filled from LUCiD rather than WCSim.
///
/// fiTQun_shared takes these as plain arrays (PMTpos, PMTdir, PMTQE); the only
/// thing WCSim ever did for it was populate them.
struct Geometry {
  int n_pmt = 0;
  std::vector<double> pos;   ///< 3*n_pmt, cm
  std::vector<double> dir;   ///< 3*n_pmt, unit, facing into the water
  std::vector<double> qe;    ///< n_pmt, relative QE (~1)
  double pmt_radius_cm = 0.;
  double det_radius_cm = 0.;
  double det_halfz_cm = 0.;

  /// Read LUCiD's sk_geometry.npz export (see tools/sadqun/npz_to_txt.py).
  static Geometry Load(const std::string& path);
};

/// The Cherenkov emission profile for one particle: the I_n tables plus the
/// per-momentum scalars. This is the raw (UseFitCProfile=0) form, which is
/// what LUCiD writes and therefore the branch SadQun implements.
class Profile {
 public:
  /// Load CProf_<pdg>_WCSim.root as written by lucid.production.fitqun.
  static Profile Load(const std::string& path);

  /// fiTQun_shared::SetTypemom, raw branch: pick the momentum bin, set the
  /// interpolation fraction, and return the per-momentum scalars.
  void SetTypemom(double mom_mev, double* In_iso, double& nphot, double& smax);

  /// fiTQun_shared::InterpolateIn -- trilinear in (R0, costh0, momentum).
  /// Must not be called before SetTypemom.
  void EvalIn(double* In, double R0, double costh0) const;

  double MomentumMin() const { return mom_bins_.front(); }
  double MomentumMax() const { return mom_bins_.back(); }

 private:
  // Axes, stored as the evaluation points fiTQun reads off the bin low edges.
  std::vector<double> r0_bins_, th0_bins_, mom_bins_;
  double r0_binw_recp_ = 0., th0_binw_recp_ = 0.;
  int n_r0_ = 0, n_th0_ = 0, n_mom_ = 0;
  // I_n[n][iR0][ith0][imom], flattened.
  std::vector<double> In_;
  std::vector<double> nphot_, smax_, iso1_, iso2_;
  // Set by SetTypemom, consumed by EvalIn -- fiTQun's imomcur/mombfrac.
  int imom_cur_ = 0;
  double mom_frac_ = 0.;

  double At(int n, int ir, int it, int im) const {
    return In_[((static_cast<size_t>(n) * n_r0_ + ir) * n_th0_ + it) * n_mom_ + im];
  }
  static double Interp1D(const std::vector<double>& x,
                         const std::vector<double>& y, double v);
};

/// The photosensor angular response, epsilon(cos eta), from angResp_<r>.root.
class AngularResponse {
 public:
  static AngularResponse Load(const std::string& path);
  double Eval(double cos_eta) const;

 private:
  std::vector<double> edges_, values_;
};

/// Predicted charge for a single-ring hypothesis: fiTQun::Get1Rmudist.
class ChargePredictor {
 public:
  ChargePredictor(Geometry geom, Profile profile, AngularResponse angresp,
                  double water_atten_cm, double qe_eff);

  /// Track parameters, fiTQun's X[]: vertex (cm), time (ns), theta, phi,
  /// momentum (MeV/c) -- the same 7 makehistWCSim fills from truth.
  struct Track {
    double x, y, z, t, theta, phi, momentum;
  };

  /// Fills mu[i] for every PMT. Returns the fiTQun PC flag convention:
  /// 0 = fully contained. Direct light only (SetScatflg(0)).
  int Get1Rmudist(const Track& trk, std::vector<double>* mu);

 private:
  Geometry geom_;
  Profile profile_;
  AngularResponse angresp_;
  double watt_cm_;   ///< SetWAttL(6800.) in makehistWCSim
  double qe_eff_;    ///< SetQEEff(0.1)

  /// fiTQun_shared::GetToWall -- distance from the vertex to the wall along dir.
  double ToWall(const double* xyz, const double* dir) const;
};

}  // namespace sadqun

#endif  // SADQUN_H
