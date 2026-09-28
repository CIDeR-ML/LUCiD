# SK 20-inch PMT geometry oracle

LUCiD includes a standalone hard-intersection oracle for the Super-Kamiokande
20-inch PMT in `lucid.propagation.sk_pmt`. It is a reference implementation,
not part of the production propagator.

The oracle follows the geometry constants and local coordinate convention in
SKDetSim's `SKDONUTS`. The PMT's inward-looking axis is local (+z), and its
position defines the local origin. All quantities below have been converted
from SKDetSim's centimetres to metres.

The front of the bulb is a sphere of radius (0.315\,\mathrm{m}), centred at
((0,0,-0.127\,\mathrm{m})):

\[
  x^2+y^2+(z+0.127)^2=0.315^2.
\]

Below (z=0.1154545\,\mathrm{m}), the lower transition uses the toroidal
implicit surface

\[
  (\sqrt{x^2+y^2}-0.104)^2+z^2=0.150^2.
\]

Only the front half with (z\geq0) participates. For barrel PMTs, SKDetSim
marks the lowest (0.020\,\mathrm{m}) as an inactive absorbing band. The
oracle reports both `intersects` and `active` so this physical intersection is
not confused with an active-photocathode hit.

For the spherical portion, the oracle solves the ray/sphere quadratic. For the
toroidal portion, it solves the ray/torus quartic and checks every real root
against the original unsquared radial equation. The extra check is required
because the SK surface is a spindle torus and squaring the radial equation can
introduce roots on a second algebraic branch.

SKDetSim searches the toroidal transition in 1 mm steps after entering its
simplified GEANT PMT envelope. The oracle solves the same implicit surface
directly. It therefore represents the intended SK shape without the sub-mm
position quantization from that search loop.

The returned normal is the local curved-surface normal transformed back to
world coordinates. The local incidence cosine is consequently

\[
  \cos\theta_{\mathrm{inc}}=\max(0,-\hat d\cdot\hat n),
\]

where (hat d) points along photon travel and (hat n) points from the PMT
solid toward the water. This is the geometric angle at the actual hit point;
it differs from the angle to the overall PMT axis.

The implementation is intentionally NumPy-only. Its exact Boolean root and
region selection makes it useful as a forward-physics oracle, but unsuitable
for gradient-based fitting. A later implementation can be tested against this
oracle while replacing those discontinuous decisions with differentiable
ones.

## Differentiable surface roots

`lucid.propagation.sk_pmt_jax` evaluates the same surface using JAX. It avoids
constructing an arbitrary local transverse basis. Given the normalized inward
PMT axis \(\hat a\) and a point relative to the PMT origin, it uses

\[
  z=p\mathbin{\cdot}\hat a,\qquad
  p_\perp=p-z\hat a,\qquad
  \rho=\lVert p_\perp\rVert.
\]

The spherical intersection is an analytic quadratic root. The toroidal
intersection starts at the entry into the enclosing 25.4 cm sphere and applies
16 fixed Newton iterations to the unsquared torus function. Differentiating
those iterations gives the implicit surface-root derivative after convergence.
The world-space intersection point and curved normal consequently retain
gradients with respect to the ray and PMT geometry.

This stage remains piecewise differentiable. Selecting sphere versus torus,
rejecting a miss, and applying the inactive-band cut are Boolean operations.
Away from those boundaries, the selected surface has the same forward value
as the hard oracle and a finite physical gradient. Smooth geometric coverage
will subsequently replace the discontinuous hit/miss boundary; it is kept
separate so surface-root errors and acceptance-model errors can be tested
independently.

## Smooth Gaussian geometric coverage

`lucid.propagation.sk_pmt_coverage` provides a differentiable reference for
the expected fraction of a Gaussian bundle of parallel rays collected by the
curved photocathode. For a nominal ray with direction \(\hat d\), it evaluates

\[
  P_{\rm hit}=
  \int_{S_{\rm active}}
  \frac{\exp[-d_\perp(x)^2/(2\sigma^2)]}{2\pi\sigma^2}
  \max(0,-\hat d\mathbin{\cdot}\hat n(x))\,dA.
\]

Here (d_\perp(x)) is the distance from surface point (x) to the nominal
ray, and the final factor converts curved surface area into area projected
onto the plane perpendicular to the ray. Curvature and incidence-angle
acceptance therefore come directly from the PMT shape. No fitted
`cos(theta)` power is applied.

The integral uses Gauss-Legendre nodes on the spherical and toroidal profile
angles and uniform midpoint nodes in azimuth. SKDONUTS' rounded constants put
the torus about 1 mm behind the sphere at their piecewise switch. A narrow
cylindrical seam closes that outer boundary in the surface integral. Its
projection is zero at normal incidence, while including it is necessary to
reproduce the hard oracle's continuous oblique silhouette.

The resulting cap-PMT projected area illustrates why a flat cosine is too
strong at large angles:

| Axis angle | Curved area / normal area | Flat-disc cosine |
|---:|---:|---:|
| 0 deg | 1.000 | 1.000 |
| 30 deg | 0.907 | 0.866 |
| 60 deg | 0.656 | 0.500 |
| 75 deg | 0.508 | 0.259 |
| 90 deg | 0.369 | 0.000 |

The nonzero edge-on area is the side projection of the curved bulb. These
numbers describe geometry only; acrylic transmission and photocathode optical
response remain separate effects.

This smooth coverage value is used directly. Its reverse-mode derivative is
the derivative of the same value seen in the forward pass; there is no hard
forward value with a substituted straight-through gradient. Tests establish:

- normal-incidence projected area equals the analytic aperture area;
- projected area from 0 to 90 degrees agrees with a hard-ray raster;
- normal-incidence Gaussian coverage agrees with the exact noncentral
  chi-square probability for a circular aperture;
- oblique Gaussian coverage agrees with a Gaussian convolution of the hard
  PMT silhouette; and
- autodiff gradients agree with finite differences of forward values.

The direct surface quadrature is a validation reference rather than the
production algorithm. A 48 by 96 configuration evaluates 13,824 surface nodes
per ray/PMT pair.

## Production propagator integration

`lucid.propagation.sk_pmt_lookup` reduces the reference integral to a compact
three-dimensional table. Its coordinates are the incidence cosine, the signed
ray offset parallel to the projected PMT axis, and the absolute perpendicular
offset. Separate cap and barrel tables preserve the barrel's inactive 2 cm
band. Each table is made by antialiased rasterization of the exact JAX
silhouette followed by the same Gaussian convolution represented by the
surface integral. Runtime evaluation uses differentiable trilinear
interpolation.

The incidence cosine is used directly instead of evaluating `acos`. This
avoids the singular derivative of `acos` at normal incidence. A stable
world-space basis converts each ray/PMT pair to the two offset coordinates;
rotating the complete geometry leaves those coordinates and the coverage
unchanged.

Enable the geometry through the normal simulator entry point:

```python
sim = setup_event_simulator(
    "config/SK_geom_config.json",
    sensor_shape="sk20inch",
    temperature=0.2,
    # other simulation options...
)
```

For finite `temperature`, the lookup's Gaussian width is
`temperature * sensor_radius`. Tables are cached under
`spatial_overlap_integrals`; the standard SK table is about 0.6 MB. Passing
`temperature=None` bypasses the lookup and uses the exact hard photocathode
decision. The legacy default remains `sensor_shape="sphere"`.

The exact central ray still supplies the physical intersection position,
travel distance, and curved normal. The PMT oracle's normal points from the
glass into water, so the propagator reverses it to match LUCiD's established
"out of the water volume" transport convention. This sign does not change a
specular reflection, but it is required for diffuse reflection and the
post-surface displacement.

The curved coverage already contains projected-area angular acceptance.
Combining `sensor_shape="sk20inch"` with the older diagnostic
`sensor_acceptance_model="cosine"` is rejected because that would apply an
extra angular response. Acrylic transmission and photocathode optical response
are still separate physics and are not included here.

## Local-incidence detection response

The optional `pmt_detection_model` adds the photocathode conversion
probability after geometric interception. It accepts a portable `.npz` table
of relative efficiency versus the local incidence cosine:

```python
sim = setup_event_simulator(
    "config/SK_geom_config.json",
    sensor_shape="sk20inch",
    pmt_detection_model="sk4_pmt_absorptance_405nm.npz",
    # other simulation options...
)
```

For the SK comparison table, `scripts/convert_sk_pmt_absorptance.py` reads
SKDetSim's `reflect.dec12.dat`, reproduces the wavelength-bin selection in
`RFPMSG`, averages the s- and p-polarized absorptances, and divides by the
normal-incidence absorptance. The converter stores the source SHA-256 and
selected table wavelength in the artifact. At a requested 405 nm, SKDetSim's
integer lookup selects the table block labelled 405.5 nm.

For a hit with incoming direction \(\hat d\), the production propagator's
curved normal \(\hat n_{out}\) points out of the detector water volume. The
response stage evaluates

\[
  \mu_{local}=\max(0,\hat d\mathbin{\cdot}\hat n_{out}),\qquad
  q=w_{geometry}\,QE_0\,q_i\,R(\mu_{local}).
\]

No extra cosine appears in this expression: `w_geometry` already includes
the curved projected area. The relative response `R` may exceed one. Expected
value modes preserve that enhancement; stochastic modes clip only the final
product `QE * R` to a valid Bernoulli probability.

The linear table interpolation, curved-surface root, hit position, and local
normal all use JAX. Forward-mode and reverse-mode derivatives agree with a
finite difference away from the physical piece boundaries. As with the PMT
geometry itself, sphere/torus selection, active-band selection, and exact
hit/miss decisions remain piecewise differentiable.

For finite-temperature Gaussian coverage, the response currently uses the
exact central ray's local normal. If the central ray misses while its Gaussian
bundle overlaps the PMT, it uses the PMT-axis normal as a stable fallback. A
future response-weighted coverage table should integrate `R(mu_local)` over
the curved surface together with the Gaussian footprint. That gives the most
faithful smooth forward value and its exact gradient. A SIREN is useful later
as a learned residual for effects such as bulb position, azimuth, and magnetic
field; it should not replace the measured one-dimensional optical table.
