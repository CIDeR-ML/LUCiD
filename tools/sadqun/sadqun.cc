// SadQun — see sadqun.h. The physics here is fiTQun's, transcribed.
#include "sadqun.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <TF1.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH3F.h>

namespace sadqun {
namespace {

// fiTQun_shared samples the angular-response TF1 onto a fixed table and
// interpolates in it; matching the sampling matters because epsilon() is
// evaluated once per PMT per step and the table, not the function, is what
// fiTQun actually uses.
constexpr int kNEpsilonBins = 1000;

}  // namespace

// ---------------------------------------------------------------- Geometry

Geometry Geometry::Load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("sadqun: cannot open geometry " + path);

  Geometry g;
  std::string key;
  while (in >> key) {
    if (key == "n_pmt") {
      in >> g.n_pmt;
      g.pos.resize(3 * g.n_pmt);
      g.dir.resize(3 * g.n_pmt);
      g.qe.resize(g.n_pmt, 1.0);
    } else if (key == "pmt_radius_cm") {
      in >> g.pmt_radius_cm;
    } else if (key == "det_radius_cm") {
      in >> g.det_radius_cm;
    } else if (key == "det_halfz_cm") {
      in >> g.det_halfz_cm;
    } else if (key == "pmt") {
      int i;
      in >> i;
      if (i < 0 || i >= g.n_pmt)
        throw std::runtime_error("sadqun: pmt index out of range");
      in >> g.pos[3 * i] >> g.pos[3 * i + 1] >> g.pos[3 * i + 2]
         >> g.dir[3 * i] >> g.dir[3 * i + 1] >> g.dir[3 * i + 2] >> g.qe[i];
    } else {
      std::string discard;
      std::getline(in, discard);
    }
  }
  if (g.n_pmt <= 0) throw std::runtime_error("sadqun: geometry has no PMTs");
  return g;
}

// ----------------------------------------------------------------- Profile

double Profile::Interp1D(const std::vector<double>& x,
                         const std::vector<double>& y, double v) {
  if (v <= x.front()) return y.front();
  if (v >= x.back()) return y.back();
  size_t i = 1;
  while (i < x.size() && x[i] < v) ++i;
  const double f = (v - x[i - 1]) / (x[i] - x[i - 1]);
  return y[i - 1] * (1. - f) + y[i] * f;
}

Profile Profile::Load(const std::string& path) {
  TFile f(path.c_str());
  if (f.IsZombie()) throw std::runtime_error("sadqun: cannot open " + path);

  Profile p;
  // fiTQun reads the axes off the bin LOW EDGES, so the evaluation points are
  // the low edges and there is one extra edge closing the last bin.
  auto* h0 = dynamic_cast<TH3F*>(f.Get("hI3d_0"));
  if (!h0) throw std::runtime_error("sadqun: hI3d_0 missing from " + path);
  p.n_r0_ = h0->GetNbinsX();
  p.n_th0_ = h0->GetNbinsY();
  p.n_mom_ = h0->GetNbinsZ();
  for (int i = 1; i <= p.n_r0_; ++i)
    p.r0_bins_.push_back(h0->GetXaxis()->GetBinLowEdge(i));
  for (int i = 1; i <= p.n_th0_; ++i)
    p.th0_bins_.push_back(h0->GetYaxis()->GetBinLowEdge(i));
  for (int i = 1; i <= p.n_mom_; ++i)
    p.mom_bins_.push_back(h0->GetZaxis()->GetBinLowEdge(i));

  p.r0_binw_recp_ = 1. / (p.r0_bins_[1] - p.r0_bins_[0]);
  p.th0_binw_recp_ = 1. / (p.th0_bins_[1] - p.th0_bins_[0]);

  p.In_.assign(static_cast<size_t>(3) * p.n_r0_ * p.n_th0_ * p.n_mom_, 0.);
  for (int n = 0; n < 3; ++n) {
    auto* h = dynamic_cast<TH3F*>(f.Get(Form("hI3d_%d", n)));
    if (!h) throw std::runtime_error("sadqun: hI3d_n missing");
    for (int i = 0; i < p.n_r0_; ++i)
      for (int j = 0; j < p.n_th0_; ++j)
        for (int k = 0; k < p.n_mom_; ++k)
          p.In_[((static_cast<size_t>(n) * p.n_r0_ + i) * p.n_th0_ + j) * p.n_mom_ + k] =
              h->GetBinContent(i + 1, j + 1, k + 1);
  }

  for (int n = 1; n <= 2; ++n) {
    auto* h = dynamic_cast<TH1D*>(f.Get(Form("hI_iso_%d", n)));
    if (!h) throw std::runtime_error("sadqun: hI_iso_n missing");
    auto& dst = (n == 1) ? p.iso1_ : p.iso2_;
    for (int k = 1; k <= h->GetNbinsX(); ++k) dst.push_back(h->GetBinContent(k));
  }

  for (const char* name : {"gNphot", "gsthr"}) {
    auto* g = dynamic_cast<TGraph*>(f.Get(name));
    if (!g) throw std::runtime_error(std::string("sadqun: ") + name + " missing");
    auto& dst = (std::string(name) == "gNphot") ? p.nphot_ : p.smax_;
    for (int k = 0; k < g->GetN(); ++k) dst.push_back(g->GetY()[k]);
  }
  return p;
}

void Profile::SetTypemom(double mom, double* In_iso, double& nphot, double& smax) {
  // fiTQun_shared::SetTypemom, raw branch.
  for (imom_cur_ = 1; imom_cur_ < n_mom_; ++imom_cur_)
    if (mom < mom_bins_[imom_cur_]) break;
  if (imom_cur_ == n_mom_) --imom_cur_;
  --imom_cur_;

  mom_frac_ = (mom - mom_bins_[imom_cur_]) /
              (mom_bins_[imom_cur_ + 1] - mom_bins_[imom_cur_]);

  nphot = Interp1D(mom_bins_, nphot_, mom);
  smax = Interp1D(mom_bins_, smax_, mom);
  if (In_iso) {
    In_iso[0] = Interp1D(mom_bins_, iso1_, mom);
    In_iso[1] = Interp1D(mom_bins_, iso2_, mom);
  }
}

void Profile::EvalIn(double* In, double R0, double costh0) const {
  // fiTQun_shared::InterpolateIn -- trilinear, with fiTQun's edge clamping.
  int iR0 = static_cast<int>((R0 - r0_bins_[0]) * r0_binw_recp_);
  int ipR0 = iR0 + 1;
  int ith0 = static_cast<int>((costh0 - th0_bins_[0]) * th0_binw_recp_);
  int ipth0 = ith0 + 1;
  const int ipmom = imom_cur_ + 1;

  if (iR0 >= n_r0_ - 1) { ipR0 = n_r0_ - 1; iR0 = ipR0 - 1; }
  else if (iR0 < 0)     { iR0 = 0; ipR0 = 1; }
  if (ith0 >= n_th0_ - 1) { ipth0 = n_th0_ - 1; ith0 = ipth0 - 1; }
  else if (ith0 < 0)      { ith0 = 0; ipth0 = 1; }

  const double xd = (R0 - r0_bins_[iR0]) * r0_binw_recp_;
  const double yd = (costh0 - th0_bins_[ith0]) * th0_binw_recp_;
  const double zd = mom_frac_;

  for (int i = 0; i < 3; ++i) {
    const double i1 = At(i, iR0, ith0, imom_cur_) * (1. - zd) + At(i, iR0, ith0, ipmom) * zd;
    const double i2 = At(i, iR0, ipth0, imom_cur_) * (1. - zd) + At(i, iR0, ipth0, ipmom) * zd;
    const double j1 = At(i, ipR0, ith0, imom_cur_) * (1. - zd) + At(i, ipR0, ith0, ipmom) * zd;
    const double j2 = At(i, ipR0, ipth0, imom_cur_) * (1. - zd) + At(i, ipR0, ipth0, ipmom) * zd;
    const double w1 = i1 * (1. - yd) + i2 * yd;
    const double w2 = j1 * (1. - yd) + j2 * yd;
    In[i] = w1 * (1. - xd) + w2 * xd;
  }
}

// -------------------------------------------------------- AngularResponse

AngularResponse AngularResponse::Load(const std::string& path) {
  TFile f(path.c_str());
  if (f.IsZombie()) throw std::runtime_error("sadqun: cannot open " + path);
  auto* fn = dynamic_cast<TF1*>(f.Get("angResp"));
  if (!fn)
    throw std::runtime_error(
        "sadqun: no 'angResp' TF1 in " + path +
        " -- run fit_cos.C on the angRespAll histograms first");

  AngularResponse a;
  // Sample onto a table and interpolate, as fiTQun_shared::epsilon does: the
  // table is what fiTQun evaluates, so matching it matters.
  const double lo = 0., hi = 1.;
  for (int i = 0; i <= kNEpsilonBins; ++i) {
    const double x = lo + (hi - lo) * i / kNEpsilonBins;
    a.edges_.push_back(x);
    a.values_.push_back(fn->Eval(x));
  }
  return a;
}

double AngularResponse::Eval(double cos_eta) const {
  if (!(cos_eta >= 0. && cos_eta < 1.)) return 0.;
  const double binw_recp = kNEpsilonBins;  // edges span [0,1]
  const int ix = static_cast<int>(cos_eta * binw_recp);
  const double xd = (cos_eta - edges_[ix]) * binw_recp;
  return values_[ix] * (1. - xd) + values_[ix + 1] * xd;
}

// -------------------------------------------------------- ChargePredictor

ChargePredictor::ChargePredictor(Geometry geom, Profile profile,
                                 AngularResponse angresp,
                                 double water_atten_cm, double qe_eff)
    : geom_(std::move(geom)), profile_(std::move(profile)),
      angresp_(std::move(angresp)), watt_cm_(water_atten_cm), qe_eff_(qe_eff) {}

namespace {
/// Distance to the nearest wall of a cylinder; negative means outside.
double Dwall(const double* p, double R, double halfz) {
  const double r = std::hypot(p[0], p[1]);
  return std::min(R - r, halfz - std::fabs(p[2]));
}
}  // namespace

double ChargePredictor::ToWall(const double* xyz, const double* dir) const {
  // Smallest positive distance along dir to the barrel or an end cap.
  const double R = geom_.det_radius_cm, hz = geom_.det_halfz_cm;
  double best = 1e9;

  const double a = dir[0] * dir[0] + dir[1] * dir[1];
  if (a > 0.) {
    const double b = xyz[0] * dir[0] + xyz[1] * dir[1];
    const double c = xyz[0] * xyz[0] + xyz[1] * xyz[1] - R * R;
    const double disc = b * b - a * c;
    if (disc > 0.) {
      const double s = (-b + std::sqrt(disc)) / a;
      if (s > 0.) best = std::min(best, s);
    }
  }
  if (dir[2] != 0.) {
    const double s = ((dir[2] > 0. ? hz : -hz) - xyz[2]) / dir[2];
    if (s > 0.) best = std::min(best, s);
  }
  return best;
}

int ChargePredictor::Get1Rmudist(const Track& trk, std::vector<double>* mu) {
  mu->assign(geom_.n_pmt, 0.);

  double dir[3] = {std::sin(trk.theta) * std::cos(trk.phi),
                   std::sin(trk.theta) * std::sin(trk.phi),
                   std::cos(trk.theta)};
  double xyz[3] = {trk.x, trk.y, trk.z};

  double Iiso[2], nphot, smax;
  profile_.SetTypemom(trk.momentum, Iiso, nphot, smax);
  const double smaxrecp = 1. / smax;

  // Three evaluation points along the track: fiTQun samples the profile at
  // s = 0, smax/2, smax and fits a quadratic through them.
  double pos[3][3];
  for (int istp = 0; istp < 3; ++istp) {
    const double s = 0.5 * istp * smax;
    for (int i = 0; i < 3; ++i) pos[istp][i] = xyz[i] + s * dir[i];
  }

  // PC flag: fiTQun calls the track contained when 1.1*smax still fits inside.
  double Rend[3];
  for (int i = 0; i < 3; ++i) Rend[i] = xyz[i] + 1.1 * smax * dir[i];
  const int PCflg = (Dwall(Rend, geom_.det_radius_cm, geom_.det_halfz_cm) < 0.) ? 1 : 0;

  const double fact = nphot * qe_eff_;
  const double r_pmt = geom_.pmt_radius_cm;
  const double watt_recp = 1. / watt_cm_;

  for (int icab = 0; icab < geom_.n_pmt; ++icab) {
    const double* ppos = &geom_.pos[3 * icab];
    const double* pdir = &geom_.dir[3 * icab];

    double costh[3], R[3], Jarr[3];
    for (int istp = 0; istp < 3; ++istp) {
      double Rvct[3], coseta = 0.;
      costh[istp] = 0.;
      R[istp] = 0.;
      for (int i = 0; i < 3; ++i) {
        Rvct[i] = pos[istp][i] - ppos[i];
        R[istp] += Rvct[i] * Rvct[i];
        costh[istp] -= Rvct[i] * dir[i];
        coseta += Rvct[i] * pdir[i];
      }
      R[istp] = std::sqrt(R[istp]);
      const double Rrecp = 1. / R[istp];
      costh[istp] *= Rrecp;
      coseta *= Rrecp;

      // J = Omega(R) * T(R) * epsilon(coseta), each as fiTQun defines it.
      const double omega = 0.5 * r_pmt * r_pmt / (R[istp] * R[istp] + r_pmt * r_pmt);
      const double trans = std::exp(-R[istp] * watt_recp);
      Jarr[istp] = omega * trans * angresp_.Eval(coseta);
    }

    double Iarr[3];
    profile_.EvalIn(Iarr, R[0], costh[0]);

    double jcoeff[3];
    jcoeff[0] = Jarr[0];
    jcoeff[1] = (-3. * Jarr[0] + 4. * Jarr[1] - Jarr[2]) * smaxrecp;
    jcoeff[2] = (Jarr[0] - 2. * Jarr[1] + Jarr[2]) * 2. * smaxrecp * smaxrecp;

    double mutmp = geom_.qe[icab] * fact *
                   (jcoeff[0] * Iarr[0] + jcoeff[1] * Iarr[1] + jcoeff[2] * Iarr[2]);
    if (!(mutmp > 0.)) mutmp = 0.;  // fiTQun: predicted charge is never negative
    (*mu)[icab] = mutmp;
  }
  return PCflg;
}

}  // namespace sadqun
