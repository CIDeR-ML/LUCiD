# JUNO chimney and Acrylic-bowl geometry

## Objective and data source

This implementation transfers the principal engineering dimensions from
JUNOSW into LUCiD for component-wise ablation and optical-component
analysis. It does not reproduce the complete CAD model, and a reduction in
the three-way discrepancy after a single change is not an acceptance
criterion. The Acrylic-bowl placement source is:

~~~text
/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.5.1/
data/Detector/Geometry/share/geometry_acrylic.gdml
~~~

The extraction script \`scripts/extract_juno_acrylic_bowls.py\` generates the
portable \`config/JUNO_acrylic_bowls.csv\` file from this GDML input.
Production simulation therefore does not require CVMFS to be mounted on the
worker node. The bottom-chimney dimensions are additionally traced to:

~~~text
/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.5.1/junosw/
Simulation/DetSimV2/CentralDetector/src/DetSim1Construction.cc
Simulation/DetSimV2/CentralDetector/src/RoundBottomFlaskSolidMaker.cc
~~~

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
inner wall, outer wall, and end caps are intersected explicitly. The Acrylic
segment is a closed transparent volume. At entry and exit, the target medium
is determined from the crossed face and Snell/Fresnel/TIR transport is
applied. While the photon is in the wall, its time, scattering and bulk
attenuation use the Acrylic material parameters. The steel segment remains
an opaque effective surface.

## Bottom-chimney component

JUNOSW J26.5.1 constructs the south-pole opening in
`DetSim1Construction.cc` with `RoundBottomFlaskSolidMaker`: the LS sphere,
Acrylic sphere and coaxial tubes are Boolean unions. The steel enclosure is
then built separately from a four-plane polycone, a lower-half ellipsoid and
a fine lower pipe. LUCiD keeps the principal optical envelope:

| Part | z range [m] | Inner radius [m] | Outer radius [m] |
|---|---:|---:|---:|
| Acrylic tube wall | -18.700 to -17.700 | 0.500 | 0.580 |
| Acrylic flange ring | -18.700 to -18.600 | 0.580 | 0.675 |
| Steel neck | -18.765 to -18.700 | 0.500 | 0.506 |
| Steel flange ring | -18.750 to -18.700 | 0.506 | 0.675 |

The steel bowl below the flange is a lower-half ellipsoidal shell centred at
`z=-18.765 m`. Its inner radial/axial semiaxes are `(0.500, 0.250) m`, and
its outer semiaxes are `(0.502, 0.255) m`. The two Acrylic pieces are stored
as a non-overlapping union. Shared faces inside that union are skipped and
therefore do not become optical material boundaries.

The LS--Acrylic sphere is opened for south-going intersections with
`rho < 0.500 m`; the Acrylic--Water sphere is opened for `rho < 0.580 m`.
The bore therefore remains LS and reaches the steel enclosure instead of
crossing an intact spherical boundary. The Acrylic wall is transparent:
entry and exit use the same Snell/Fresnel/TIR and Acrylic bulk-absorption
state machine as the top chimney and Acrylic bowls. The steel neck and bowl
remain an opaque effective surface with configurable absorption, specular
reflection and diffuse reflection.

This clean model intentionally omits the 62 mm-bore fine lower pipe, bolts,
flexible coupling and edge fillets. These parts lie below the principal
enclosure and can be added only if a component ablation demonstrates an
identifiable optical contribution.

The component has an independent switch:

~~~text
--bottom-chimney-geometry config|on|off
BOTTOM_CHIMNEY_GEOMETRY=config|on|off
~~~

Surface codes 7 and 8 denote bottom-chimney Acrylic and steel. Their
per-event hit, reflection, transmission, material entry/exit, bulk
absorption and Acrylic path-length diagnostics are written separately in
`results_summary.json`. Reproducible PDF and PNG figures are generated with:

~~~sh
JAX_PLATFORMS=cpu /opt/gentoo/usr/bin/python \
  scripts/render_juno_bottom_chimney.py \
  --output-dir geometry_render/bottom_chimney
~~~

## Acrylic-bowl component

JUNOSW places 590 Acrylic additions, named
\`AdditionAcrylicConstruction\`, at support-joint positions on the
17.824 m-radius shell. LUCiD retains the nominal dimensions:

- shell-side base radius: 0.450 m;
- outward tip radius: 0.200 m;
- radial length: 0.140 m;
- orientation: the radial unit vector at the placement point.

The former straight frustum side has been replaced by the dormant JUNOSW
\`yumiao\` profile. In local outward axial coordinate \(s\), the radius is

\[
r(s)=0.69764-
\sqrt{0.776^2-(0.73543-s)^2},
\qquad 0\leq s\leq0.140\ {\rm m}.
\]

The side is one smooth surface in production. Two roots of the old straight
frustum seed six bounded Newton updates. The updates evaluate the equivalent
torus quartic with scalar Horner operations; they do not rebuild a 3D point or
evaluate square roots at every iteration. The final physical-profile residual
is checked before a hit is accepted. Surface normals use the analytic
derivative of the circle, not a segmented-polycone approximation. A 33-point
sampling of the same formula is used only for visualisation, never photon
propagation.

The browser viewer receives this sampled profile from
`scripts/view_juno_geometry.py` and constructs one rotational surface with
Three.js `LatheGeometry`. Reproducible static overview figures use the same
production radius function:

~~~sh
JAX_PLATFORMS=cpu /opt/gentoo/opt/PyMC/bin/python \
  scripts/render_juno_known_structures.py \
  --output-dir geometry_render
~~~

The angular lookup retains twelve nearby bowls per cell. Reference-sphere and
selected-surface lookups are both evaluated, giving at most twenty-four
candidates per ray. A trial reduction to eight candidates per cell was
rejected: although the centre-source bowl-hit loss was only 0.47%, the losses
at source z=16.1 m and z=17.1 m were 16.8% and 12.6%, respectively. The
24-candidate definition is therefore part of the clean geometry baseline.

The intersection includes the circular-arc side and both flat caps. These
surfaces now close one transparent Acrylic volume: a ray can enter through
the curved wall or attachment cap, propagate with Acrylic optical parameters,
and leave through the next boundary. The flat base is the current effective
shell-connection inner wall; it is not yet the exact sphere-subtracted CAD
root. To avoid numerical coincidence between the spherical interface and the
bowl base, the effective bowl is shifted outward by 0.5 mm. This offset is
only a numerical separation, not an additional engineering dimension.

JUNOSW J26.5.1 still selects its three-plane \`simple\` profile by default;
the C++ \`yumiao\` branch is present but dormant. LUCiD intentionally selects
that branch's 776 mm circular arc to represent the known overall bowl
curvature. This does not claim to reproduce the manufactured edge fillets.

## Transparent Acrylic optics

The chimney Acrylic and Acrylic bowls no longer use a fixed A/R/T surface.
For each boundary crossing, LUCiD evaluates Snell's law, Fresnel reflection
and total internal reflection from the incident and target refractive
indices. A transmitted photon changes its material state. Its survival over
an Acrylic path of length \(d\) is

\[
P_{\rm survive}=\exp(-d/L_{\rm abs}^{\rm Acrylic}),
\]

with the 532 nm baseline
\(L_{\rm abs}^{\rm Acrylic}=8.908\,\mathrm{m}\). Acrylic scattering remains
disabled through the material file's effectively infinite scattering length.

The legacy Acrylic surface values are consequently set to

\[
(P_{\rm refl},P_{\rm trans},P_{\rm abs})=(0,1,0).
\]

They are inert compatibility entries rather than physical probabilities;
Fresnel reflection and bulk absorption are applied by the material-interface
path. This prevents the old 2% surface loss from being counted a second time.
The steel segment retains its effective values \((0.40,0,0.60)\), with
specular/diffuse splitting controlled by \(f_{\rm spec}\).

Stable surface codes are 0 for support rods, 1 for chimney Acrylic, 2 for
chimney steel, 3 for Acrylic bowls, 7 for bottom-chimney Acrylic and 8 for
bottom-chimney steel. For each surface class,
\`results_summary.json\` records the following per-event quantities:

~~~text
hits, surface_absorbed, reflected, specular, diffuse, transmitted,
entered, exited, bulk_absorbed, acrylic_path_m
~~~

The summary separately records `chimney_acrylic` and `chimney_steel`; their
sum remains available as `chimney`. This separation is needed because only
the Acrylic segment has material entry/exit and bulk-path diagnostics.
The backward-compatible `absorbed` field is the sum of `surface_absorbed`
and `bulk_absorbed`.

## Independent ablation

Three command-line options control the components independently:

~~~text
--chimney-geometry config|on|off
--bottom-chimney-geometry config|on|off
--acrylic-bowl-geometry config|on|off
~~~

The corresponding batch-script environment variables are
\`CHIMNEY_GEOMETRY\`, \`BOTTOM_CHIMNEY_GEOMETRY\` and
\`ACRYLIC_BOWL_GEOMETRY\`. The support-rod switch,
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
shapes, transparent Acrylic path transport, independent ablation, and
diagnosable optical components. The following are explicitly not implemented:

- the upper chimney beyond the PMT sphere, Tyvek, and external structures;
- the short upper/lower edge fillets, exact sphere-subtracted attachment
  root, holes, and fasteners;
- closed-form quartic root selection; bounded Newton updates of the exact
  torus polynomial are used instead;
- angle- or wavelength-dependent engineering-surface parameters.

Additional geometry is justified only if ablation studies reveal stable,
identifiable residuals in these local regions that cannot be described by the
effective parameters. CAD complexity should not be added pre-emptively.
