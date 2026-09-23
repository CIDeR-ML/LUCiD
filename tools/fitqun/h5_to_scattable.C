// Convert LUCiD's HDF5 scattering tables into the TScatTable ROOT file fiTQun
// reads (fiTQun_scattablesF_<config>.root).
//
// TScatTable is a user-defined class with its own dictionary, so the file must
// be written by a program that has that dictionary — which is why this step is
// here rather than in lucid/production/fitqun/scattable.py. Run it from the
// fiTQun Utilities checkout, where TScatTable.cc and its LinkDef already live:
//
//   root -l -b -q 'h5_to_scattable.C("scattables.h5", "fiTQun_scattablesF_SK_WAND.root")'
//
// Requires ROOT built with HDF5 headers available to the interpreter; if that
// is not the case, dump the datasets to text with h5dump and adapt ReadFlat.

#include <hdf5.h>

#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"

#include "TScatTableF.h"

namespace {

// Read one group written by lucid.production.fitqun.scattable.write_hdf5.
bool ReadGroup(hid_t file, const std::string& surface,
               std::vector<int>& nbins, std::vector<double>& bounds,
               std::vector<double>& flat) {
  hid_t group = H5Gopen2(file, surface.c_str(), H5P_DEFAULT);
  if (group < 0) {
    std::cerr << "missing group: " << surface << std::endl;
    return false;
  }

  nbins.assign(6, 0);
  hid_t attr = H5Aopen(group, "nbins", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, nbins.data());
  H5Aclose(attr);

  bounds.assign(12, 0.0);
  attr = H5Aopen(group, "bounds", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_DOUBLE, bounds.data());
  H5Aclose(attr);

  hid_t dset = H5Dopen2(group, "table", H5P_DEFAULT);
  hid_t space = H5Dget_space(dset);
  hsize_t n = 0;
  H5Sget_simple_extent_dims(space, &n, nullptr);
  flat.assign(n, 0.0);
  H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
  H5Sclose(space);
  H5Dclose(dset);
  H5Gclose(group);
  return true;
}

}  // namespace

void h5_to_scattable(const char* h5path, const char* outpath) {
  hid_t file = H5Fopen(h5path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "cannot open " << h5path << std::endl;
    return;
  }

  TFile out(outpath, "RECREATE");

  // fiTQun looks these up by name in GetScatRatio; keep the spelling.
  const char* surfaces[3] = {"side", "top", "bot"};
  for (const char* surface : surfaces) {
    std::vector<int> nbins;
    std::vector<double> bounds, flat;
    if (!ReadGroup(file, surface, nbins, bounds, flat)) continue;

    TScatTableF table(
        surface, surface,
        nbins[0], bounds[0], bounds[1],
        nbins[1], bounds[2], bounds[3],
        nbins[2], bounds[4], bounds[5],
        nbins[3], bounds[6], bounds[7],
        nbins[4], bounds[8], bounds[9],
        nbins[5], bounds[10], bounds[11]);

    // Element order matches: LUCiD writes dimension 0 fastest, which is what
    // TScatTable::GetIndex computes.
    for (size_t i = 0; i < flat.size(); ++i) {
      table.SetElement(static_cast<int>(i), flat[i]);
    }
    out.cd();
    table.Write(surface);
    std::cout << "wrote " << surface << " (" << flat.size() << " elements)" << std::endl;
  }

  out.Close();
  H5Fclose(file);
}
