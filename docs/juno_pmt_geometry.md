# JUNO 20-inch PMT geometry baseline (P1a)

## Scope

P1a replaces each 254 mm-radius spherical receiver with the
detector-facing front head of the corresponding JUNO 20-inch PMT. It uses
the installed CopyNo, position, orientation, and PMT technology. The PMT
equator remains at the position supplied by the official placement table.

This stage is geometry only. A photon reaching the outer glass surface is
still treated by LUCiD as an ideal sensor with QE = 1. Glass propagation,
thin-film ARC/photocathode response, vacuum propagation, and secondary
internal paths are P1b work.

## JUNOSW J26.3.2 inputs

The implementation follows these release files:

- `data/Detector/Geometry/PMTPos_CD_LPMT.csv`
- `data/Detector/Geometry/PMTType_CD_LPMT.csv`
- `junosw/Simulation/DetSimV2/PMTSim/src/`
  `Hamamatsu_R12860_PMTSolid.cc`
- `junosw/Simulation/DetSimV2/PMTSim/src/`
  `NNVT_MCPPMT_PMTSolid.cc`
- `junosw/Simulation/DetSimV2/PMTSim/src/`
  `HamamatsuR12860PMTManager.cc`
- `junosw/Simulation/DetSimV2/PMTSim/src/`
  `NNVTMCPPMTManager.cc`

The release root used for verification is
`/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.3.2`.
The committed type table is a whitespace-normalized copy of the release
table; `diff -w -B` reports no semantic difference.

## Geometry

| PMT technology | Radial semi-axis | Front axial semi-axis |
|---|---:|---:|
| Hamamatsu R12860 | 254 mm | 190 mm |
| NNVT / High-QE NNVT | 254 mm | 184 mm |

The glass inner surface uses JUNOSW's `inner_delta = -5 mm`, giving
249 mm radial and 185/179 mm axial semi-axes. P1a stores this surface for
the next optical stage but intersects photons only with the outer surface.

For PMT centre `c`, inward axis `a`, and point `x`, define

```text
z = (x - c) dot a
rho = (x - c) - z a
(rho dot rho) / R^2 + z^2 / H^2 = 1,  z >= 0
```

The first positive ray root satisfying `z >= 0` is the water-to-glass
entry point. The geometry result retains both normal conventions:

- `water_normals` points from glass into detector water;
- `detector_outward_normals` points from the detector interior into the PMT.

The transport/reflection API consumes `detector_outward_normals`. Therefore
both specular and diffuse PMT reflections return to the detector interior.

## Configuration

`config/JUNO_laser_geom_config.json` enables
`pmt_geometry = "juno_20inch_front"` and supplies the position and type
tables. The command-line option `--pmt-geometry on|off|config` changes only
this intersection model. When explicit geometry is enabled, the existing
`cos(theta)` factor is bypassed because the curved surface supplies the
angular projection geometrically. When it is disabled, the legacy spherical
receiver retains that factor, reproducing the pre-P1a angular baseline.

The transport baseline is fixed in `config/JUNO_laser_config.json`:
`K = 64`, Acrylic absorption active, and LS--Acrylic plus
Acrylic--Water Snell/Fresnel/TIR optics active. No P0 on/off switches remain.
