# JUNO effective support-rod geometry

## Objective and scope

In the full-detector LUCiD forward model, this component answers two
questions: whether a ray hits a support rod before a PMT or medium boundary,
and whether that hit is effectively absorbed or continues in a specular or
diffuse reflection direction. It does not reproduce every CAD detail.

The first version retains only the two families of principal \`StrutSteel\`
tubes from the JUNOSW J26.3.2 \`geometry_acrylic.gdml\` file:

| Family | Count | Radius | Full length |
|---|---:|---:|---:|
| \`main_85mm\` | 370 | 42.5 mm | 1548.054 mm |
| \`secondary_60mm\` | 220 | 30.0 mm | 1548.077 mm |

There are 590 rods in total. Their centre coordinates are read directly from
the official GDML. The local axes of both tube families point radially toward
the detector centre, so the unit axis is derived from the centre coordinate at
runtime. This retains the physical orientation without exposing the GDML
Euler-angle convention in the simulation interface.

Ball heads, connectors, bolts, and Boolean/CAD details at support nodes are
not included. Acrylic bowls and the chimney are independent components. Any
of these details should be considered for a second version only when
supported by diagnostic evidence.

## Finite-cylinder intersection

For a rod with centre \(\mathbf c\), unit axis \(\mathbf a\), half-length
\(h\), and radius \(r\), and a ray

\[
\mathbf x(t)=\mathbf o+t\mathbf d,\qquad t>0,
\]

decompose \(\mathbf d\) and \(\mathbf o-\mathbf c\) into components parallel
and perpendicular to \(\mathbf a\). The cylindrical side is defined by

\[
\left\|\mathbf o_\perp+t\mathbf d_\perp\right\|^2=r^2,
\qquad
\left|o_\parallel+t d_\parallel\right|\le h,
\]

and the two end caps are defined by

\[
o_\parallel+t d_\parallel=\pm h,
\qquad \|\mathbf x_\perp(t)\|\le r.
\]

The smallest valid positive root is compared with the current LS--Acrylic,
Acrylic--Water, PMT, or outer-boundary intersection. A support-rod hit
replaces the selected intersection only when it is closer. The component is
therefore part of the transport geometry rather than a post-processing
correction to the final PE count.

To avoid constructing an \`n_photon x 590\` array, a spherical angular grid
stores 12 neighbouring rods per direction. Each ray queries both the grid
cell at the mean-rod-radius intersection and the cell at the currently
selected boundary intersection, for at most 24 cylinder candidates. In a
fixed-seed validation of 4000 random rays covering the full radial range, the
candidate method recovered all 40 true support-rod hits found by exhaustive
intersection with all 590 rods. The nearest rod indices and distances agreed.

## Surface process and diagnostics

The J26.3.2 \`StrutSteel\` material/surface reference gives a constant
reflectivity of 0.535 over the range that includes 532 nm. The effective
parameters are therefore:

- \`support_reflection_rate = 0.535\`: total reflection probability;
- \`support_fspec = 0.5\`: effective specular fraction conditional on
  reflection;
- \`1 - support_reflection_rate\`: effective absorption probability.

\`support_fspec\` is not a claim about the exact JUNOSW surface
microstructure. It is a low-dimensional, configuration-driven parameter
intended for later differentiable calibration. The per-event
\`results_summary.json\` records \`support_hits\`, \`support_blocked\`,
\`support_absorbed\`, \`support_reflected\`, \`support_specular\`,
\`support_diffuse\`, and \`support_transmitted\`.

The current support transmission probability is zero; the transmitted field
is retained so all engineering surfaces share the same diagnostic contract.
\`blocked\` means that the rod geometry interrupted the previously selected
transport path and is numerically equal to \`hits\`. Furthermore,
\`absorbed + reflected = hits\` and
\`specular + diffuse = reflected\`.

## Ablation and acceptance

The command-line option \`--support-geometry config|on|off\`, corresponding
to the batch environment variable \`SUPPORT_GEOMETRY\`, controls this
component independently. With the same random seed, photon count, \(K\), and
all other configuration values, compare

\[
\Delta G_{\rm support}(t)=
G_{\rm PMT+support}(t)-G_{\rm PMT}(t).
\]

The first comparison uses \(z=0\) and \(z=16.1\,\mathrm m\), followed by
\(z=17.1\,\mathrm m\) as an additional validation position. The acceptance
goal is to identify the prompt, late, occlusion, and reflection components
changed by the support rods. LUCiD is not required to move numerically closer
to JUNOSW or data.

## Criteria for an upgraded version

The current implementation satisfies the \`diffsim.tex\` and \`AGENTS.md\`
requirements for component-wise deployment of known structures: positions,
directions, and principal dimensions have an official source; the shape is
interpretable; the component can be ablated independently; intersections and
reflection processes remain differentiable; and diagnostics are traceable.

An upgraded version should be defined only if at least one of the following
conditions is observed:

1. Ablation maps show stable residuals concentrated at support nodes that
   cannot be explained by the principal rods.
2. Per-photon diagnostics show substantial leakage near rod ends, such that
   ball heads or connectors would change component attribution.
3. Validation against exhaustive intersection with all 590 rods reveals
   misses in the candidate grid.
4. Later differentiable calibration drives the rod radius or surface
   parameters to boundary solutions, indicating an insufficient effective
   shape.

Even then, a second version should add only the ball heads or connection nodes
supported by this evidence, rather than copying the complete CAD model.
