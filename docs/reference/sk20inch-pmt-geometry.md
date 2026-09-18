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
