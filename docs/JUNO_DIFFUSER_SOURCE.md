# JUNO Green Diffuser Source

## Scope

The clean LUCiD baseline uses the effective LS-phase emission implemented by
JUNOSW J26.5.1 `GtOpScintTool`.  It does not simulate transport inside the
optical fibre or the PTFE diffuser.  External PTFE, PEEK, and fibre collision
geometry remains a separate `diffuser_hardware` component.

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
