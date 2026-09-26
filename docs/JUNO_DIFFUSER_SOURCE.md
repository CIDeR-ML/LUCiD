# JUNO Green Diffuser Source

## Scope

The clean LUCiD baseline separates two independently switchable components:

- `diffuser_emission` reproduces the effective LS-phase emission implemented
  by JUNOSW J26.5.1 `GtOpScintTool`;
- `diffuser_hardware` lets already-emitted or returning photons collide with
  the PTFE ball, PEEK upper tube, and PE fibre.

The hardware never generates the theta-intensity distribution and does not
simulate microscopic transport inside the diffuser.  This avoids applying the
effective internal-scattering response twice.

## Emission model

The source centre is the position supplied by `--source`.  A proposal surface
normal is sampled uniformly on a sphere of radius 0.010 m.  Its cosine relative
to the fixed assembly axis is denoted by `c`.  The photon direction is then
uniform in solid angle over the local outward hemisphere, so
`mu = dot(direction, normal)` is uniform on `[0, 1]`.

The source time is Gaussian with a standard deviation of 0.23 ns.  The
wavelength remains fixed at 532 nm.

## Angular intensity

The normalized density in `c` is a positive cubic Bernstein mixture:

```text
x = (c + 1) / 2
p(c) = 2 * sum_j w_j * B_j,3(x)
w = softmax(eta)
```

The nominal J26.5.1 LSPhase weights are:

```text
[0.2818246381, 0.2170696741, 0.3539173588, 0.1471883290]
```

They exactly represent the JUNOSW cubic after its `cost = -GetRandom()` sign
convention.  The softmax removes the common scale and leaves three independent
shape degrees of freedom.

LUCiD samples `c` from a uniform proposal and applies the differentiable
importance factor `p(c)/(1/2)`.  The finite-sample weights are renormalized to
the configured total intensity.  This keeps source yield separate from angular
shape and preserves common random paths while angular coefficients vary.

## Runtime selection

`config/JUNO_laser_config.json` selects `juno_diffuser_ball` by default.

Set `SOURCE_MODEL=juno_diffuser_ball` for the new baseline or
`SOURCE_MODEL=isotropic` for the legacy mathematical point-source ablation in
the group's existing shared-account submission command.  This document does
not introduce a new submission wrapper.

Set `DIFFUSER_HARDWARE=config`, `on`, or `off` in the existing local run
script.  The checked-in geometry selects `juno_ls_diffuser_hardware`, so
`config` and `on` enable the component.  `off` preserves the finite emission
model but removes all three collision objects.  The assembly centre always
follows `--source`; its local axis follows the configured diffuser axis.

## Hardware geometry and optical seeds

The effective solids reproduce the main dimensions in the J26.5.1
`CalibUnit/share/laserdiffuser.gdml` LS configuration:

| Part | Effective LUCiD geometry | Axial placement |
|---|---|---|
| PTFE ball | sphere, radius 10 mm | centre at source position |
| PEEK upper tube | capped cylinder, radius 2 mm, length 8 mm | centre at +10 mm |
| PE fibre | capped cylinder, radius 0.5 mm, length 200 mm | centre at +110 mm |

The PEEK and fibre reflectances are the GDML values 0.5 and 0.1.  Their
`finish=3` surfaces are represented by the first-version diffuse branch.  The
PTFE surface uses a 0.95 polished/specular seed, matching the fallback optical
surface used by JUNO calibration-source GDML files; the production JUNOSW
surface obtains its wavelength table from `Material.TeflonMirror.REFLECTIVITY`.
All three surfaces are opaque in this effective model, so non-reflected
photons are absorbed.

These values are explicit baseline parameters, not precision material claims.
The current implementation omits refraction and volume absorption inside the
three solids, fine connectors, the source weight, and cable details.

## Hardware diagnostics

`results_summary.json` records `hits`, `absorbed`, `reflected`, `specular`,
`diffuse`, and `transmitted` counts for the combined `diffuser_hardware`
component and separately for `diffuser_ptfe`, `diffuser_peek`, and
`diffuser_fiber`.  The Parquet metadata records the selected geometry mode and
the complete seven-surface R/T/specular arrays.

When trajectory recording is enabled, the HDF5 event group also stores
`initial_directions` and `source_weights`.  Initial positions and emission
times remain the zeroth entries of `positions` and `times`.

## Validation contract

The deterministic tests check:

- 10 mm emission radius;
- outward-hemisphere support and mean local cosine 0.5;
- nominal weighted moments `E[n_z] = -0.05341225` and
  `E[d_z] = -0.02670612`;
- 0.23 ns source-time width;
- exact total source intensity;
- fixed-key reproducibility;
- finite non-zero gradients with respect to angular logits;
- zero emission times for legacy three-array sources.

Hardware tests additionally check the GDML dimensions, translated and rotated
source poses, finite-cylinder caps, absence of an immediate outward PTFE
self-hit, nearest-surface competition, independent on/off loading, and the
three optical parameter rows.
