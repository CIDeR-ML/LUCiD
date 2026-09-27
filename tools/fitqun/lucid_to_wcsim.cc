// Write LUCiD events as a WCSim-format ROOT file, so fiTQun can reconstruct them.
//
// fiTQun reads its input through WCSimWrap, which opens exactly three things:
// the `wcsimT` tree's `wcsimrootevent` branch, the `wcsimGeoT` tree's
// `wcsimrootgeom` branch (entry 0), and an optional `Settings` tree whose
// rotation branches it falls back to the identity without. This writes those
// and nothing else -- it is a format adapter, not a WCSim emulation.
//
// The geometry comes from tools/sadqun/export_geometry.py's text form, which
// already carries everything fiTQun reads off a WCSimRootGeom. Everything here
// is in cm, as WCSim stores it.
//
// Hit times are shifted into the trigger frame a WCSim file is expected to carry.
//
// The offset is NOT free to choose. makehistWCSim.cc:262 removes a hardcoded
// 950 ns (`aSubToffs = 950 - trigOffset`, and trigOffset is the trigger header
// Date, which is 0 here) when it computes the corrected time tc. Shipping any
// other offset leaves tc displaced by the difference, and since the tc axis is
// only +-100 ns, a 100 ns displacement moves the time PDF off its own range.
// 1050 maximises the hits landing inside fiTQun's 900-1400 ns window (65% vs
// 58%), but that gain is not worth a 100 ns systematic, so 950 it is. To change
// it, set the trigger Date to (950 - offset) so aSubToffs follows.
// LUCiD times are relative to the event (median ~0-200 ns), while fiTQun fits a
// default window of 900-1400 ns (fiTQun.DefaultTimeWindow{Start,End}) and silently
// DISCARDS every hit outside it -- which leaves a sparse late tail and a fit with
// no vertex or direction information. SK's global trigger sits near 800-900 ns
// (skdetsim DS-GLTTIM 800), so --time-offset-ns defaults to 950.
//
//   lucid_to_wcsim --geometry geom.txt --sensor wc_sensor_0000.h5 -o out.root
//                  [--truth FILE] [-n N] [--time-offset-ns 950]
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <map>
#include <cstdlib>

#include <hdf5.h>

#include <TFile.h>
#include <TTree.h>

#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"

namespace {

/// WCSim's trigger frame, WCSimWCTrigger.hh:107-115 and .cc:330-454. A WCSim
/// digit time is `t_hit + offset - triggertime`, with `triggertime` the
/// threshold-th digit of the gate snapped down to `kStep`. That self-centring
/// is why fiTQun's default 900-1400 ns window sits where it does and why
/// makehistWCSim can hardcode `aSubToffs = 950`.
///
/// LUCiD already runs the same NDigits trigger (200 ns / 25 hits) and stores
/// its gates in labl/event_NNN/per_window; what it does not do is reference
/// times to the trigger or split gates into subevents. Both are done here.
constexpr int    kNDigitsThreshold = 25;
constexpr int    kStep             = 5;     ///< ns, trigger-time granularity


/// One truth primary, as tools/../truth.py writes it (cm, MeV/c, ns).
struct Track {
  int event = 0, pdg = 0;
  double x = 0., y = 0., z = 0., t = 0.;
  double dx = 0., dy = 0., dz = 0., p = 0.;
};

struct Geometry {
  int n_pmt = 0;
  double pmt_radius_cm = 0., det_radius_cm = 0., det_halfz_cm = 0.;
  std::vector<double> pos, dir;   // 3*n_pmt
};

Geometry LoadGeometry(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open geometry: " + path);
  Geometry g;
  std::string key;
  while (in >> key) {
    if (key == "n_pmt") {
      in >> g.n_pmt;
      g.pos.resize(3 * g.n_pmt);
      g.dir.resize(3 * g.n_pmt);
    } else if (key == "pmt_radius_cm") {
      in >> g.pmt_radius_cm;
    } else if (key == "det_radius_cm") {
      in >> g.det_radius_cm;
    } else if (key == "det_halfz_cm") {
      in >> g.det_halfz_cm;
    } else if (key == "pmt") {
      int i; double qe;
      in >> i >> g.pos[3*i] >> g.pos[3*i+1] >> g.pos[3*i+2]
           >> g.dir[3*i] >> g.dir[3*i+1] >> g.dir[3*i+2] >> qe;
    } else {
      std::string skip;
      std::getline(in, skip);
    }
  }
  if (g.n_pmt <= 0) throw std::runtime_error("geometry has no PMTs");
  return g;
}

/// Truth tracks keyed by event index. Absent file -> no tracks, which is fine
/// for a reconstruction run but not for the time-PDF stage, which reads them.
std::map<int, Track> LoadTracks(const std::string& path) {
  std::map<int, Track> out;
  if (path.empty()) return out;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open truth: " + path);
  std::string key;
  while (in >> key) {
    if (key != "track") { std::string skip; std::getline(in, skip); continue; }
    Track t;
    in >> t.event >> t.pdg >> t.x >> t.y >> t.z >> t.t >> t.dx >> t.dy >> t.dz >> t.p;
    out[t.event] = t;
  }
  return out;
}

/// Mass for the PDG codes a tune covers; AddTrack wants m and E alongside p.
double MassMeV(int pdg) {
  switch (std::abs(pdg)) {
    case 11: return 0.51099895;
    case 13: return 105.6583755;
    case 211: return 139.57039;
    case 22: return 0.0;
    case 2212: return 938.27208816;
    case 321: return 493.677;
    default: return 0.0;
  }
}

/// Read one 1-D dataset into a vector of the requested native type.
template <typename T>
std::vector<T> ReadVec(hid_t file, const std::string& name, hid_t mem_type) {
  hid_t ds = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (ds < 0) throw std::runtime_error("missing dataset " + name);
  hid_t space = H5Dget_space(ds);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  std::vector<T> out(dims[0]);
  if (dims[0] > 0)
    H5Dread(ds, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
  H5Sclose(space);
  H5Dclose(ds);
  return out;
}

/// The event group names, in file order then sorted, so "event_10" cannot
/// overtake "event_9" the way a printf-formatted guess would.
std::vector<std::string> EventGroups(hid_t file) {
  H5G_info_t info;
  H5Gget_info(file, &info);
  std::vector<std::string> names;
  for (hsize_t i = 0; i < info.nlinks; ++i) {
    ssize_t len = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                                     nullptr, 0, H5P_DEFAULT);
    std::string name(len, '\0');
    H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(),
                       len + 1, H5P_DEFAULT);
    if (name.rfind("event_", 0) == 0) names.push_back(name);
  }
  std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
    return a.size() != b.size() ? a.size() < b.size() : a < b;
  });
  return names;
}

const char* Arg(int argc, char** argv, const char* key) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  const char* geom_path = Arg(argc, argv, "--geometry");
  const char* sensor_path = Arg(argc, argv, "--sensor");
  const char* out_path = Arg(argc, argv, "-o");
  const char* truth_path = Arg(argc, argv, "--truth");
  if (!geom_path || !sensor_path || !out_path) {
    std::cerr << "usage: lucid_to_wcsim --geometry FILE --sensor FILE -o FILE"
                 " [--truth FILE] [-n N]\n";
    return 2;
  }
  const long n_max = Arg(argc, argv, "-n") ? std::atol(Arg(argc, argv, "-n")) : -1;
  const double t_offset = Arg(argc, argv, "--time-offset-ns")
                              ? std::atof(Arg(argc, argv, "--time-offset-ns")) : 950.0;

  try {
    Geometry g = LoadGeometry(geom_path);
    std::map<int, Track> tracks = LoadTracks(truth_path ? truth_path : "");

    TFile out(out_path, "recreate");

    WCSimRootGeom geom;
    geom.SetWCNumPMT(g.n_pmt);
    geom.SetWCCylRadius(g.det_radius_cm);
    geom.SetWCCylLength(2. * g.det_halfz_cm);
    geom.SetWCPMTRadius(g.pmt_radius_cm);
    geom.SetWCOffset(0., 0., 0.);
    for (int i = 0; i < g.n_pmt; ++i) {
      double rot[3] = {g.dir[3*i], g.dir[3*i+1], g.dir[3*i+2]};
      double pos[3] = {g.pos[3*i], g.pos[3*i+1], g.pos[3*i+2]};
      // cylLoc by orientation, the same |dir_z| cut fiTQun uses to tell a cap
      // from the barrel. fiTQun copies it into fPMTloc and never reads it back,
      // so this only has to be self-consistent.
      const int cyl_loc = rot[2] > 0.8 ? 2 : (rot[2] < -0.8 ? 0 : 1);
      // tubeNo is 1-based in WCSim; LUCiD's sensor_idx is the 0-based row of
      // this same table, so tubeNo = idx + 1 keeps the labellings aligned.
      geom.SetPMT(i, i + 1, 0, 0, cyl_loc, rot, pos, true, false);
    }
    TTree tgeo("wcsimGeoT", "geometry");
    WCSimRootGeom* geom_ptr = &geom;
    tgeo.Branch("wcsimrootgeom", "WCSimRootGeom", &geom_ptr);
    tgeo.Fill();
    tgeo.Write();

    const char* gates_path = Arg(argc, argv, "--gates");
    hid_t gates_h5 = gates_path
        ? H5Fopen(gates_path, H5F_ACC_RDONLY, H5P_DEFAULT) : (hid_t)-1;
    hid_t h5 = H5Fopen(sensor_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5 < 0) throw std::runtime_error(std::string("cannot open ") + sensor_path);
    std::vector<std::string> groups = EventGroups(h5);
    if (n_max > 0 && (long)groups.size() > n_max) groups.resize(n_max);

    WCSimRootEvent event;
    event.Initialize();
    WCSimRootEvent* event_ptr = &event;
    TTree tev("wcsimT", "events");
    tev.Branch("wcsimrootevent", "WCSimRootEvent", &event_ptr, 64000, 0);

    long total_digits = 0, n_tracks = 0;
    long n_hits_total = 0, n_hits_in_window = 0;
    long n_untriggered = 0, n_gates_thin = 0, n_subevents = 0;
    std::vector<size_t> first_order;
    for (size_t iev = 0; iev < groups.size(); ++iev) {
      auto pe = ReadVec<float>(h5, groups[iev] + "/PE", H5T_NATIVE_FLOAT);
      auto t = ReadVec<double>(h5, groups[iev] + "/T", H5T_NATIVE_DOUBLE);
      auto idx = ReadVec<unsigned short>(h5, groups[iev] + "/sensor_idx",
                                        H5T_NATIVE_USHORT);

      // LUCiD's own trigger gates. digit_offsets is CSR-style over this event's
      // digit arrays, so gate g owns [digit_offsets[g], digit_offsets[g+1]).
      std::vector<int> gate_off;
      if (gates_h5 >= 0)
        gate_off = ReadVec<int>(gates_h5, groups[iev] + "/per_window/digit_offsets",
                                H5T_NATIVE_INT);
      if (gate_off.size() < 2) gate_off = {0, (int)t.size()};   // one gate, all digits
      const size_t n_gates = gate_off.size() - 1;

      event.ReInitialize();
      std::vector<int> no_photons;
      int isub = 0;
      for (size_t g = 0; g < n_gates; ++g) {
        // Digits of this gate, time-ordered.
        std::vector<size_t> order;
        for (int i = gate_off[g]; i < gate_off[g + 1] && i < (int)t.size(); ++i)
          order.push_back((size_t)i);
        std::sort(order.begin(), order.end(),
                  [&t](size_t a, size_t b) { return t[a] < t[b]; });
        if ((int)order.size() <= kNDigitsThreshold) { ++n_gates_thin; continue; }

        // Trigger time the way WCSim defines it, within this gate.
        double trig_time = t[order[kNDigitsThreshold]];
        trig_time -= std::fmod(trig_time, (double)kStep);

        if (isub > 0) event.AddSubEvent();
        WCSimRootTrigger* tg = event.GetTrigger(isub);
        tg->SetHeader((int)iev, 0, 0, isub + 1);
        tg->SetMode(0);
        for (size_t j : order) {
          const double th = t[j] + t_offset - trig_time;
          ++n_hits_total;
          if (th >= 900. && th <= 1400.) ++n_hits_in_window;
          tg->AddCherenkovDigiHit(pe[j], th, (int)idx[j] + 1, 0, 0, no_photons);
        }
        tg->SetNumDigitizedTubes((int)order.size());
        total_digits += (long)order.size();
        if (isub == 0) first_order.swap(order);
        ++isub;
      }
      if (isub == 0) { ++n_untriggered; continue; }
      WCSimRootTrigger* trig = event.GetTrigger(0);
      std::vector<size_t>& order = first_order;
      // Truth primary. makehistWCSim reads Ipnu/P/Dir/Start off this, so a file
      // written without it can carry a reconstruction but not a time-PDF tune.
      //
      // It reads them off track *2*: WCSim reserves tracks 0 and 1 for the beam
      // (flag -1) and the target (flag -2), so the primaries start at index 2
      // (WCSimEventAction.cc:728, "First two tracks are special"). Writing only
      // the primary segfaults the tuning chain on At(2). For a particle gun the
      // generator has no target, so track 1 is the empty one WCSim writes when
      // GetTargetPDG() is 0.
      auto it = tracks.find((int)iev);
      if (it != tracks.end()) {
        const Track& tk = it->second;
        const double m = MassMeV(tk.pdg);
        const double E = std::sqrt(tk.p * tk.p + m * m);
        double dir[3] = {tk.dx, tk.dy, tk.dz};
        double pdir[3] = {tk.p * tk.dx, tk.p * tk.dy, tk.p * tk.dz};
        double start[3] = {tk.x, tk.y, tk.z};
        double stop[3] = {tk.x, tk.y, tk.z};
        // The trigger's own vertex, which is a SEPARATE field from the tracks.
        // makehistWCSim.cc:197 builds the predicted charge from trigger->GetVtx(),
        // not from the track, so leaving it at its default puts every predicted
        // track at the detector centre: mu comes out ~100x too small and the
        // TOF-subtracted time residual is meaningless. Reconstruction never
        // reads it, so this is invisible until the tuning chain runs.
        for (int k = 0; k < 3; ++k) trig->SetVtx(k, start[k]);
        trig->SetVtxvol(0);
        // track 0: beam. WCSim gives it the gun's own pdg/direction, zero mass,
        // and a start 100 m back along the direction for the event display.
        double beam_start[3] = {tk.x - 10000.0 * tk.dx, tk.y - 10000.0 * tk.dy,
                                tk.z - 10000.0 * tk.dz};
        double beam_pdir[3] = {E * tk.dx, E * tk.dy, E * tk.dz};
        trig->AddTrack(tk.pdg, -1, 0.0, E, E, 0, -1, dir, beam_pdir, stop,
                       beam_start, 0, (ProcessType_t)0, tk.t, 0, 0,
                       std::vector<std::vector<float>>(), std::vector<float>(),
                       std::vector<double>(), std::vector<int>());
        // track 1: target. Empty for a gun run.
        double zero[3] = {0.0, 0.0, 0.0};
        trig->AddTrack(0, -2, 0.0, 0.0, 0.0, 0, -1, zero, zero, stop, stop,
                       0, (ProcessType_t)0, tk.t, 0, 0,
                       std::vector<std::vector<float>>(), std::vector<float>(),
                       std::vector<double>(), std::vector<int>());
        // track 2: the primary the tuning chain actually reads.
        trig->AddTrack(tk.pdg, 0, m, tk.p, E, 0, 0, dir, pdir, stop, start,
                       0, (ProcessType_t)0, tk.t, 1, 0,
                       std::vector<std::vector<float>>(), std::vector<float>(),
                       std::vector<double>(), std::vector<int>());
        ++n_tracks;
      }
      n_subevents += isub;

      // Contract check, per event. Every field below is read by a tool that
      // does NOT complain when it is missing -- it takes a default and produces
      // quietly wrong numbers. Four separate bugs in this file were found only
      // by running a later stage and noticing the output was nonsense, so the
      // requirements are asserted here, where the objects are still in memory.
      if (trig->GetTracks()->GetEntries() < 3)
        throw std::runtime_error("tracks < 3: makehistWCSim.cc:34 reads the "
                                 "primary at index 2 (WCSim reserves 0,1)");
      {
        auto* prim = (WCSimRootTrack*)trig->GetTracks()->At(2);
        if (!prim || prim->GetParenttype() != 0)
          throw std::runtime_error("track 2 is not a parentless primary");
        const double vtx[3] = {trig->GetVtx(0), trig->GetVtx(1), trig->GetVtx(2)};
        if (vtx[0] == 0. && vtx[1] == 0. && vtx[2] == 0.)
          throw std::runtime_error("trigger vertex unset: makehistWCSim.cc:197 "
                                   "predicts charge from it, not from the track");
        for (int k = 0; k < 3; ++k)
          if (std::fabs(vtx[k] - prim->GetStart(k)) > 1e-3)
            throw std::runtime_error("trigger vertex disagrees with track 2");
      }
      tev.Fill();
    }
    H5Fclose(h5);
    if (gates_h5 >= 0) H5Fclose(gates_h5);

    tev.Write();
    out.Close();
    std::cout << out_path << ": " << groups.size() << " events, " << total_digits
              << " digits, " << n_tracks << " truth tracks, " << g.n_pmt
              << " PMTs, " << n_subevents << " subevents (one per LUCiD trigger "
              << "gate), times referenced to each gate's trigger + " << t_offset
              << " ns" << std::endl;
    const double frac = n_hits_total ? (double)n_hits_in_window / n_hits_total : 0.;
    std::cout << "  " << n_hits_in_window << "/" << n_hits_total << " hits ("
              << (int)(100 * frac + 0.5) << "%) inside fiTQun's 900-1400 ns window";
    if (n_untriggered) std::cout << "; " << n_untriggered << " events had no gate";
    if (n_gates_thin)  std::cout << "; " << n_gates_thin << " gates under threshold";
    std::cout << std::endl;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
