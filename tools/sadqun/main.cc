// sadqun — predict fiTQun's mu at every PMT for a set of tracks.
//
//   sadqun --geometry geom.txt --profile CProf_13_WCSim.root
//          --angresp angResp.root --tracks tracks.txt --out mu.txt
//
// tracks.txt: one track per line, "x y z t theta phi momentum" in cm/ns/rad/MeV.
// mu.txt:     one line per track, "PCflg mu_0 mu_1 ... mu_(n-1)".
//
// The defaults for --watt and --qeeff are makehistWCSim.cc's SetWAttL(6800.)
// and SetQEEff(0.1), which that stage sets deliberately to pin the
// normalisation to constants that are not themselves being tuned.
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "sadqun.h"

namespace {

void Usage() {
  std::cerr <<
      "usage: sadqun --geometry FILE --profile FILE --angresp FILE\n"
      "              --tracks FILE --out FILE [--watt CM] [--qeeff F]\n";
}

const char* Arg(int argc, char** argv, const char* key) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  const char* geom_path = Arg(argc, argv, "--geometry");
  const char* prof_path = Arg(argc, argv, "--profile");
  const char* ang_path = Arg(argc, argv, "--angresp");
  const char* trk_path = Arg(argc, argv, "--tracks");
  const char* out_path = Arg(argc, argv, "--out");
  if (!geom_path || !prof_path || !ang_path || !trk_path || !out_path) {
    Usage();
    return 2;
  }
  const double watt = Arg(argc, argv, "--watt") ? std::atof(Arg(argc, argv, "--watt")) : 6800.;
  const double qeeff = Arg(argc, argv, "--qeeff") ? std::atof(Arg(argc, argv, "--qeeff")) : 0.1;

  try {
    auto geom = sadqun::Geometry::Load(geom_path);
    auto prof = sadqun::Profile::Load(prof_path);
    auto ang = sadqun::AngularResponse::Load(ang_path);
    sadqun::ChargePredictor pred(geom, prof, ang, watt, qeeff);

    std::ifstream in(trk_path);
    if (!in) throw std::runtime_error("sadqun: cannot open tracks");
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("sadqun: cannot write output");
    out.precision(9);

    sadqun::ChargePredictor::Track trk;
    std::vector<double> mu;
    long n = 0;
    while (in >> trk.x >> trk.y >> trk.z >> trk.t >> trk.theta >> trk.phi >>
           trk.momentum) {
      const int pc = pred.Get1Rmudist(trk, &mu);
      out << pc;
      for (double m : mu) out << ' ' << m;
      out << '\n';
      ++n;
    }
    std::cerr << "sadqun: " << n << " tracks x " << geom.n_pmt << " PMTs -> "
              << out_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
