# JUNO chimney and Acrylic-bowl geometry

## Objective and data source

This implementation transfers the principal engineering dimensions from
JUNOSW J26.3.2 into LUCiD for component-wise ablation and optical-component
analysis. It does not reproduce the complete CAD model, and a reduction in
the three-way discrepancy after a single change is not an acceptance
criterion. The geometry source is:

~~~text
/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.3.2/
data/Detector/Geometry/share/geometry_acrylic.gdml
~~~

The extraction script \`scripts/extract_juno_acrylic_bowls.py\` generates the
portable \`config/JUNO_acrylic_bowls.csv\` file from this GDML input.
Production simulation therefore does not require CVMFS to be mounted on the
worker node.

## Chimney component

The first version retains the part of the lower chimney that overlaps the
central detector and clips it at the PMT boundary:

| Segment | z range [m] | Inner radius [m] | Outer radius [m] |
|---|---:|---:|---:|
| Acrylic | 17.824--18.424 | 0.400 | 0.524 |
| Steel | 18.424--19.434 | 0.400 | 0.405 |

The chimney bore has a radius of 0.400 m. If a spherical LS--Acrylic or
Acrylic--Water intersection lies inside the upper central aperture, that
spherical intersection is skipped. The photon remains in the LS medium until
it hits a chimney wall, a PMT, or another structure. The chimney is therefore
a genuine geometric opening rather than a decorative cylinder superimposed
on an intact spherical shell.

The chimney walls are represented by coaxial finite annular cylinders. The
inner wall, outer wall, and end caps are intersected explicitly. The current
model does not propagate refracted paths through the physical Acrylic wall;
instead, it applies one effective absorption/reflection/transmission (A/R/T)
surface interaction.

## Acrylic-bowl component

JUNOSW places 590 Acrylic additions, named
\`AdditionAcrylicConstruction\`, at support-joint positions on the
17.824 m-radius shell. Each addition is approximated by a radial frustum:

- shell-side base radius: 0.450 m;
- outward tip radius: 0.200 m;
- radial length: 0.140 m;
- orientation: the radial unit vector at the placement point.

The intersection includes the frustum side and both caps, while omitting the
central opening, eight bolt holes, and curved details created by Boolean
subtractions. A ray already inside the effective solid does not interact with
a second exit surface. This is equivalent to a single thin-surface response
and avoids applying A/R/T twice without a complete Acrylic-volume medium
state machine. To avoid numerical coincidence between the spherical
interface and the bowl base, the effective bowl surface is shifted outward
by 0.5 mm. This offset is only a numerical separation, not an additional
engineering dimension.

## Effective surface optics

Engineering surfaces satisfy

\[
P_{\rm abs}+P_{\rm refl}+P_{\rm trans}=1,
\]

where \(P_{\rm refl}\) is the total reflection probability,
\(P_{\rm trans}\) is the straight-through probability, and the remainder is
the effective absorption probability. Conditional on reflection,
\(f_{\rm spec}\) selects specular reflection and \(1-f_{\rm spec}\) selects a
cosine-weighted diffuse direction.

The initial Acrylic values in the current configuration are

\[
(P_{\rm refl},P_{\rm trans},P_{\rm abs})=(0.04,0.94,0.02),
\]

and the initial values for the steel segment are \((0.40,0,0.60)\). These are
interpretable starting values, not a claim of JUNOSW-level optical precision.
They should later be constrained with 532 nm data from multiple source
positions.

Stable surface codes are 0 for support rods, 1 for chimney Acrylic, 2 for
chimney steel, and 3 for Acrylic bowls. For each surface class,
\`results_summary.json\` records the following per-event quantities:

~~~text
hits, absorbed, reflected, specular, diffuse, transmitted
~~~

## Independent ablation

Two command-line options control the new components independently:

~~~text
--chimney-geometry config|on|off
--acrylic-bowl-geometry config|on|off
~~~

The corresponding batch-script environment variables are
\`CHIMNEY_GEOMETRY\` and \`ACRYLIC_BOWL_GEOMETRY\`. The support-rod switch,
\`SUPPORT_GEOMETRY\`, remains independent. With the same random seed, compare
at least:

1. the PMT + support baseline;
2. baseline + chimney;
3. baseline + Acrylic bowls;
4. baseline + chimney + Acrylic bowls.

Acceptance quantities are the additional hit, occlusion, reflection, and
transmission components and their prompt/late timing contributions. LUCiD is
not required to move closer to JUNOSW or data after every component is added.

## Current scope and upgrade criteria

The current implementation satisfies the near-term requirements in
\`diffsim.tex\` and \`AGENTS.md\` for principal positions, dimensions and
shapes, independent ablation, and diagnosable optical components. The
following are explicitly not implemented:

- explicit refraction, path length, and absorption inside the chimney
  Acrylic wall;
- the upper chimney beyond the PMT sphere, Tyvek, and external structures;
- the physical bowl curvature, holes, fasteners, and secondary propagation
  inside the Acrylic volume;
- angle- or wavelength-dependent engineering-surface parameters.

Additional geometry is justified only if ablation studies reveal stable,
identifiable residuals in these local regions that cannot be described by the
effective parameters. CAD complexity should not be added pre-emptively.
