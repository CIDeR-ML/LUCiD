# What we fixed in the SKDetSim–LUCiD laser comparison

**Audience:** LUCiD collaboration and Super-K calibration group

**Configuration:** SK-IV injector-1, 405 nm, 8,090 photons per pulse

**Updated:** 1 October 2026
**Code branch:** `feature/sk4-laser-validation` (geometry state `a268c31`)

## Executive summary

We did not tune water parameters until two different simulations happened to
look similar. We built a sequence of controlled tests in which one physical or
detector effect was enabled at a time. Whenever a difference appeared, we
recorded progressively earlier checkpoints in the two programs and replayed
the same rays through both detector models. This separated source generation,
water propagation, detector-wall routing, PMT geometry, photocathode response,
timing response, and electronics.

The main problems and fixes were:

| Layer | Problem found | Fix or diagnostic | Main result |
|---|---|---|---|
| SK water configuration | A zero Mie multiplier still left an additive asymmetric-scattering term | In the isolated SK comparison build, return an exactly zero asymmetric coefficient when Mie is disabled | Removed the false far-angle halo completely |
| PMT-to-PMT response | Early LUCiD runs gave every PMT the same efficiency | Applied mean-normalized SK `qetable3_0.dat` cable factors | Reduced accepted-PE per-PMT TV by 58.8% in the baseline study |
| Comparison observable | LUCiD expected PE was being compared with SK digitized charge | Made SK accepted PE before charge electronics the primary quantity | Separated optical modeling from a roughly 0.4% electronics loss and its spatial pattern |
| Monte Carlo precision | Low-statistics residual maps looked structured | Increased SK from 1,000 to 5,000 pulses and LUCiD from 2M to 8M integration rays; plotted pulls and disjoint-sample stability | Baseline residual RMS pull reached 1.04, consistent with statistical fluctuations |
| PMT timing | SK `MCPHOTON` times already included a non-Gaussian SK-IV transit-time mixture | Implemented the same six-component mixture in LUCiD | Closed the prompt-bin and late-tail discrepancy |
| Photon speed | LUCiD used the phase-velocity approximation, while SK uses group velocity | Added a monochromatic group-speed override, 0.217289524 m/ns at 405 nm | Reduced top-cap conditional TOF TV by 43.6% at that stage |
| Detector layout | It was unclear whether coordinate differences came from different PMT tables | Audited all 11,146 cable IDs, positions, axes, detector sections, and exact SK-IV dumps | PMT placement was ruled out as the main top-cap problem |
| PMT shape | Legacy LUCiD treated a 20-inch PMT as a full sphere | Implemented the SK `SKDONUTS` sphere-plus-torus bulb and barrel inactive band | Correctly reduced high-angle projected acceptance |
| Photocathode angle response | LUCiD used only scalar QE | Converted the 405 nm SK absorptance table and evaluated it using the local curved-surface normal | Added a differentiable, physically interpretable local-angle response |
| Candidate routing | LUCiD could test several nearby curved PMTs for one wall crossing; SK routes through one GEANT wall cell | Added wall-assigned candidate selection before the curved-surface test | Reduced the identical-ray accepted top-gap by 76.1% |
| PMT root numerics | The correct torus surface was evaluated by subtracting roughly 36 m float32 quantities to recover a centimetre-scale hit | Rebased the solve near the PMT, used stable sphere bounds, bracketed the physical torus branch, and retained the local hit position | Exact direct replay changed from 0.88275 to 1.00077 times the SK hit count |
| Rayleigh rotation | An older LUCiD version used the transpose of the required local-to-world convention | Verified the upstream `frame.T @ local_direction` fix and added an oblique-ray regression test | Current sample has no measurable upward-z preference |

After all current corrections, the Rayleigh-only, total-charge-normalized TOF
histogram TV is **0.002833**, compared with **0.014924** before the stable PMT
intersection correction. This is an **81.0% reduction**. The top cap, all five
barrel bands, and bottom cap now overlap closely, and both detector-wide prompts
are 147 ns before display alignment.

In plain language: the broad timing and detector-region mismatch was not a
wrong Rayleigh length. Most of it came from how rays were connected to PMTs and
from a numerical loss of valid hits near the PMT skirt. Once the same rays were
sent to the same PMT surface with stable arithmetic, SKDetSim and LUCiD nearly
agreed.

One important issue remains. The current detailed LUCiD model predicts
**728.43 PE/pulse**, while SKDetSim predicts **668.20 PE/pulse** in the
Rayleigh-only sample. The corresponding direct yields are 718.99 and 660.02.
The Rayleigh/direct ratios agree extremely well—1.01313 in LUCiD and 1.01238
in SKDetSim—so the remaining difference behaves like a common source or
normalization issue, not extra Rayleigh light. Replaying the exact SK-generated
direct rays through LUCiD gives a hit-count ratio of 1.00077. The next isolated
test should therefore compare the independently generated laser-ray banks in
detail before changing any water coefficient.

## 1. Controlled comparison and terminology

The common source configuration is:

| Quantity | Value |
|---|---:|
| Source position | $(-0.707,-7.777,18.027)$ m |
| Nominal direction | normalized $(0.01123,0.02418,-0.9768)$ |
| Wavelength | 405 nm |
| Photons per pulse | 8,090 |
| Historical beam parameter | `sigbm=3300` |
| Detector | SK-IV inner detector, 11,146 PMTs |

The value 3,300 is a parameter of the historical SK laser generator. It is not
a 3,300-degree opening angle. The standalone LUCiD source reproduces the
historical `RNGAUSYOO` transform rather than interpreting the number as the
standard deviation of an ordinary Gaussian beam.

LUCiD uses an average-response estimator in these comparisons. Its integration
rays estimate the expected response of one 8,090-photon pulse. Eight million
rays do not represent eight million physical photons in one event. SKDetSim,
by contrast, generates stochastic pulses and individual photons. We average
5,000 SK pulses and compare them with eight independent one-million-ray LUCiD
batches.

The primary charge quantity is SKDetSim **accepted PE** from `MCPHOTON`. It is
recorded after PMT acceptance but before the stochastic charge electronics.
LUCiD's average response is also an expected PE quantity. SK digitized charge
is retained as a separate electronics diagnostic.

For normalized patterns we use

$$
\mathrm{TV}=\frac{1}{2}\sum_i|f_i^{\rm SK}-f_i^{\rm LUCiD}|.
$$

A TV of 0.01 means that 1% of the normalized response would have to be moved
between bins to make the distributions identical. A pull is a residual divided
by the combined SK and LUCiD Monte Carlo standard error. An RMS pull near one
means that the observed scatter is consistent with the estimated statistics.

In plain language: we compare the same stage of the detector chain, normalize
only when asking about shape, and use error bars to avoid interpreting random
red and blue pixels as physics.

## 2. Establishing a truly scattering-free baseline

### 2.1 The hidden SK asymmetric-scattering term

The first nominally ballistic SK run still produced light beyond 20 degrees,
while LUCiD did not. The cause was in the SK-IV asymmetric-scattering
coefficient, evaluated in `WASYSCSG`. Its default form contains

$$
\alpha_{\rm asym}(\lambda)=\texttt{asyfit}\left[1+
\texttt{miefac}\frac{(\lambda-\texttt{miesfac})^2}{\lambda^4}\right].
$$

Setting `miefac=0` removes the wavelength-dependent term but leaves
`asyfit`. The surviving coefficient was approximately
$1.002079\times10^{-4}\ \mathrm{m}^{-1}$. `WTRSG` included it in the total
interaction rate, `SGABSC` selected the process, and `SGMIES` generated the
asymmetric direction.

For the isolated comparison executable, we added an early guard so that a
requested zero Mie factor returns an exactly zero asymmetric coefficient. This
was a diagnostic-build change; it did not alter the collaboration production
executable.

Before the fix, SKDetSim produced 1.306 accepted PE/pulse beyond 20 degrees.
After the fix, both accepted PE and digitized charge beyond 20 degrees became
exactly zero.

In plain language: the control knob multiplied only part of the formula. It
looked like scattering was switched off, but a small built-in floor remained.
We made “off” mean exactly off for the controlled test.

### 2.2 Absorption, Rayleigh scattering, reflection, and dark noise

For the pure baseline, the SK comparison build explicitly disabled bulk-water
absorption, symmetric Rayleigh scattering, asymmetric scattering, secondary
surface reflection, and dark noise. LUCiD used finite lengths of $10^{20}$ m
rather than infinity, because infinity created intermediate `inf/inf` values.
Over the detector this differs from zero interaction probability by less than
$10^{-18}$.

We later restored absorption and Rayleigh scattering separately. Reflections,
Mie scattering, acrylic transport in LUCiD, and dark noise remain disabled in
the present Rayleigh comparison.

In plain language: we first tested a photon flying in a straight line through
perfectly transparent water. We only added realistic water processes after
that simple case was understood.

## 3. Per-PMT efficiency and electronics

### 3.1 Relative cable-dependent QE

The early LUCiD baseline assigned every PMT the same efficiency. SKDetSim uses
the cable-dependent factors in `qetable3_0.dat`. We imported the same table,
ordered it by cable ID, divided it by its detector-wide mean, and multiplied
LUCiD's scalar base QE by the corresponding relative factor $q_i$.

This is not a fit to the laser pattern. It transfers a detector calibration
already used by SKDetSim. In the 5,000-event accepted-PE comparison, it reduced
per-PMT TV from 0.02354 to 0.00971 and angular TV from 0.00675 to 0.00221.

In plain language: two PMTs in the real detector are not equally efficient.
Once LUCiD used the same measured relative efficiencies, much of the spotty
PMT-by-PMT disagreement disappeared.

### 3.2 Accepted PE versus digitized charge

At 5,000 pulses, the ballistic SK sample contained

- 660.0224 +/- 0.3529 accepted PE/pulse;
- 657.4028 +/- 0.5123 digitized PE-equivalent charge/pulse; and
- a digitized/accepted ratio of 0.996031.

The charge electronics therefore reduce the mean by about 0.397% and add a
small PMT-dependent pattern. We did not fold this effect into water or PMT
geometry parameters. The main physics comparison uses accepted PE, while the
digitized curve is shown only when discussing electronics.

In plain language: a photoelectron being created and the readout reporting its
charge are different steps. Comparing LUCiD directly to final digitized charge
made a small electronics effect look like an optical error.

### 3.3 Statistical convergence and residual maps

Increasing SKDetSim from 1,000 to 5,000 pulses reduced the digitized per-PMT
RMS residual by 31.7%. Increasing LUCiD from two million to eight million rays
reduced the remaining residual by another 33.9%. In the final legacy-sphere
ballistic comparison, the per-PMT RMS pull was 1.04.

We therefore changed the standard spatial diagnostic to show the full detector
in the Super-K event-display layout: top cap, unwrapped barrel, and bottom cap.
It places symmetric percent difference beside statistical pull, uses total-
charge-normalized PMT fractions, and does not hide low-charge PMTs with a
threshold. Randomly mixed red and blue pulls are consistent with statistics;
a detector-wide region with a common sign reveals a systematic effect.

We also compared disjoint LUCiD samples and two-million- versus eight-million-
ray residual maps. A pattern that changes when the random sample changes is
statistical. A pattern that persists with the same sign is a candidate model
difference.

In plain language: brighter percentage colors near the edge of a faint spot do
not automatically mean bad diffusion. Dividing by the expected random error
shows whether a difference is actually surprising.

## 4. Timing fixes

### 4.1 SK-IV PMT timing mixture

The accepted-PE timestamp in SKDetSim is written after the SK-IV timing mixture
in `sgpmt.F`. It is pre-charge-electronics truth, but it is not the bare optical
surface-arrival time. The mixture is:

| Probability | Offset distribution [ns] |
|---:|---:|
| 1.400% | $N(111.5,7.0)$ |
| 1.300% | $N(39.0,5.0)$ |
| 2.375% | $N(80.0,25.0)$ |
| 0.025% | $N(-47.0,8.0)$ |
| 1.000% | $N(15.0,4.5)$ |
| 93.900% | zero offset |

We implemented this in `lucid/simulation/pmt_timing.py` and apply it after
optical propagation but before hit aggregation. It changes timestamps only.
Charge and PMT assignment stay unchanged. A dedicated JAX random substream
ensures that enabling timing does not reshuffle later QE or charge draws.

In the ballistic TOF test, the unmodified LUCiD prompt fraction from 1490 to
1510 ns was 0.83738, versus 0.78790 in SK accepted PE. After the SK timing
mixture, it became 0.78824. The fraction at or after 1520 ns became 0.05421,
versus 0.05407 in SKDetSim.

In plain language: SK PMTs sometimes report a photoelectron early or tens to
hundreds of nanoseconds late. LUCiD originally put nearly every direct photon
in the sharp central peak. Adding the same PMT delay distribution moved the
right amount of light out of that peak.

### 4.2 Group velocity at 405 nm

LUCiD initially used the phase-velocity approximation
$c/1.33=0.22541$ m/ns. SKDetSim uses the wavelength-dependent group index

$$
n_g(\lambda)=n(\lambda)-\lambda\frac{dn}{d\lambda},
$$

implemented through `EFFNSG`. At 405 nm the corresponding speed is
0.217289524 m/ns. The difference matters most for long scattered paths. Prompt
alignment can hide it for direct bottom light while leaving the top-cap tail
compressed.

We added a monochromatic propagation-speed override and used the SK group
speed. At that stage the top-cap width increased from 79.41 to 82.22 ns, its
conditional TOF TV fell from 0.05551 to 0.03131, and whole-detector canonical
TOF TV fell from 0.02062 to 0.01787.

In plain language: the direct photons all travel nearly the same distance, so
shifting the peak can hide a slightly wrong speed. Scattered photons take many
different path lengths, so the wrong speed stretches or compresses the tail.

## 5. Adding absorption by itself

LUCiD's 405 nm absorption model gave

$$
\alpha_{\rm abs}=0.002719894\ \mathrm{m}^{-1},\qquad
L_{\rm abs}=367.661\ \mathrm{m}.
$$

SKDetSim's native SK-IV parameterization gave

$$
\alpha_{\rm abs}=0.002708049\ \mathrm{m}^{-1},\qquad
L_{\rm abs}=369.270\ \mathrm{m}.
$$

We deliberately retained the two native values instead of forcing them equal.
In LUCiD average-response mode, a path of length $d$ is weighted by
$\exp(-d/L_{\rm abs})$. Sampled mode makes the corresponding survive/absorb
decision.

The absorption-only results were:

| Quantity | SKDetSim | LUCiD |
|---|---:|---:|
| Absorption-off yield | 660.022 | 660.572 |
| Absorption-on yield | 597.905 +/- 0.330 | 598.631 +/- 0.293 |
| On/off survival | 0.905885 | 0.906231 |

The survival fractions differ by only 0.000346. Their inferred effective path
lengths are 36.50 and 36.20 m.

In plain language: when absorption was the only water process restored, the
two programs removed the same fraction of light. This gave us confidence that
absorption was not responsible for later Rayleigh pattern differences.

## 6. Rayleigh scattering audit

### 6.1 Free path and first interaction

LUCiD's 405 nm Rayleigh length in this comparison is 186.516 m. A diagnostic
recorded every first interaction in both programs. For an equivalent averaged
Rayleigh phase function:

| Quantity | SKDetSim | LUCiD | Binned TV |
|---|---:|---:|---:|
| Probability of a first scatter | 0.176869 | 0.177311 | -- |
| Mean distance from laser | 17.5696 m | 17.5448 m | 0.00997 |
| Mean scatter-position z | 0.5755 m | 0.6001 m | 0.00930 |
| Mean scattering angle | 89.932 deg | 90.001 deg | 0.01015 |

The zero-scatter survival of direct charge was also close: 81.90% in SKDetSim
and 82.34% in LUCiD. The top-cap scatter-order composition agreed: about 93% of
top hits had one scatter, about 6.4% had two, and fewer than 0.5% had three.
Increasing LUCiD propagation depth from four to eight steps did not change the
result materially.

In plain language: the photon scattered at the same rate, in the same places,
and through the same angles. The disagreement arose after the scatter, when
the ray reached the detector wall and PMTs.

### 6.2 Rayleigh phase function and polarization

The LUCiD averaged Rayleigh sampler uses

$$
p(\mu)=\frac{3}{8}(1+\mu^2),\qquad \mu=\cos\theta,
$$

with inverse CDF obtained from

$$
\mu^3+3\mu-(8u-4)=0.
$$

Normal SKDetSim transports photon polarization and uses it in `SGABSC`. A
diagnostic SK build forced the polarization-averaged branch. That moved the SK
top fraction from 3.8160% to 3.8590%, only 0.0430 percentage points and about
7.8% of the then-observed diagnostic gap. Polarization is physically relevant,
but it was not the dominant cause of the top-cap excess.

### 6.3 Historical local-frame transpose bug

`create_local_frame(direction)` returns basis vectors as rows. A local vector
must therefore be mapped to world coordinates with

```python
world_direction = frame.T @ local_direction
```

An older LUCiD version used `frame @ local_direction`. That transformation is
the inverse rotation and builds the cone around the wrong axis. Because the
frame construction chooses a reference axis using a Cartesian component, the
error creates a detector-fixed azimuthal anisotropy. Upstream commit `cbad78a`
corrected every Rayleigh and Mie scattering site.

All samples used in the present comparison have source hashes matching the
fixed implementation. We nevertheless added an oblique-ray regression test
that compares the requested $\mu$ with the dot product of the incident and
scattered world directions. A test using an exactly $+z$ incident ray would
not reliably expose the transpose error.

The saved current sample contains 88,722 upward and 88,589 downward first-
scatter rays:

$$
P(d_z>0)=0.500375\pm0.001187,
$$

and

$$
\langle d_z\rangle=-0.000033\pm0.001498.
$$

The mirrored up/down test gives $\chi^2/\mathrm{ndf}=12.81/20$,
$p=0.886$. The LUCiD-versus-SK signed-z KS distance is 0.00201 with
$p=0.745$.

In plain language: a real direction-rotation bug existed historically, but it
is fixed in the code and was not present in our Rayleigh comparison. The
current photon directions show no statistically significant preference for
upward z.

## 7. Detector coordinates and PMT geometry

### 7.1 Cable IDs, PMT positions, and axes

We audited `ConnectionTable_SK5.root` directly. LUCiD contains 11,146 unique
cable IDs: 7,650 barrel, 1,748 top-cap, and 1,748 bottom-cap PMTs. Cable ordering
and table rows agree exactly after the documented detector-section reorder.

LUCiD's `SuperK` geometry projects measured centers onto an ideal cylinder by
default: barrel centers go to radius 16.9 m and cap centers to
$z=\pm18.1$ m. The mean shift from raw connection-table positions is 1.51 mm,
and the maximum is 6.29 mm.

We also dumped all SK-IV `GXYZPM` positions and `GDXYZPM` axes from the running
SKDetSim. Across the diagnostic sample, SK5-based LUCiD and SK-IV axes agree
within 0.021 degrees, while center differences have median 4.45 mm and maximum
12.35 mm. An exact SK-IV LUCiD geometry reproduced the SK positions to better
than 0.002 mm but did not improve the top-cap residual. PMT placement was
therefore ruled out as the dominant effect.

In plain language: we really were comparing the same cables in the same parts
of the detector. Moving the LUCiD PMTs to the exact SK-IV coordinates did not
solve the discrepancy.

### 7.2 The SK 20-inch sphere-plus-torus bulb

SKDetSim's `SKDONUTS` surface is piecewise. In PMT-local coordinates, with the
mounting point at the origin and the inward PMT axis along $+z$:

Sphere:

$$
\rho^2+(z+0.127)^2=(0.315)^2,
$$

Torus:

$$
(\rho-0.104)^2+z^2=(0.150)^2.
$$

The sphere/torus switch is at $z=0.11545$ m. The outer mounting-plane radius
is 0.254 m, and barrel PMTs have an inactive 0.020 m base band. The older LUCiD
model used a complete 0.254 m sphere centered at the mounting point. Its apex
sat 6.6 cm farther into the water than the SK bulb.

We implemented the SK constants and equations twice:

- `lucid.propagation.sk_pmt.intersect_sk20inch_pmt_hard` is the NumPy hard
  reference oracle;
- `lucid.propagation.sk_pmt_jax.intersect_sk20inch_pmt_jax` is the JAX forward
  implementation with differentiable roots, positions, normals, and incidence
  cosine within a fixed surface branch.

The new surface reduces high-angle projected area. Relative to the legacy full
sphere, the cap-PMT projected area is 0.907 at 30 degrees, 0.656 at 60 degrees,
0.508 at 75 degrees, and 0.369 at 90 degrees. The inactive barrel band reduces
the corresponding values further.

In plain language: the old LUCiD PMT looked too round from the side. A real SK
bulb presents a smaller target to a grazing photon. That matters strongly for
Rayleigh light reaching the top and barrel at large angles.

### 7.3 Local photocathode angle response

For a hit on the curved bulb, LUCiD now computes

$$
\mu_{\rm local}=\mathrm{clip}(\mathbf d\cdot\mathbf n_{\rm outward},0,1).
$$

The converter `scripts/convert_sk_pmt_absorptance.py` reads SKDetSim's
`reflect.dec12.dat` using the wavelength lookup in `RFPMSG`. At 405 nm, the
relative unpolarized response is

$$
R(\theta)=\frac{A_s(\theta)+A_p(\theta)}{2A(0)}.
$$

The expected LUCiD contribution is

$$
w_{\rm PE}=w_{\rm geometry}\,QE_{\rm base}\,q_i\,R(\mu_{\rm local}).
$$

There is no extra cosine factor: projected geometric acceptance is already in
$w_{\rm geometry}$. `TabulatedPmtDetectionResponse.factor` uses linear JAX
interpolation and permits a relative factor above one when the SK thin-film
table does so.

In SKDetSim, the relevant path is `SGPMT` -> `SKDONUTS` -> `SGREFPMT` ->
`RFPMSG`. `SGREFPMT` obtains the local angle from the incident and reflected
directions. LUCiD obtains it directly from the analytic local surface normal.

This response improved absolute acceptance but did not explain the remaining
top-cap excess. On a mismodeled incidence distribution, a correct angular
response can amplify the wrong photons. That observation motivated the same-ray
geometry tests instead of retuning the response curve.

In plain language: first we ask whether a photon physically hits the curved
bulb. Then we ask how likely that angle is to make a photoelectron. Those are
separate effects and should not be hidden in one fitted scale.

### 7.4 Acrylic and PMT SIREN status

A transparent-acrylic plus flat-photocathode SK diagnostic changed absolute
yield from 660.02 to 608.33 PE/pulse but did not improve normalized shape:
accepted-PE per-PMT TV changed from 0.00559 to 0.00574. In the later matched
flat-response Rayleigh test, the acrylic choice moved the top fraction by only
about 0.3% relative. Acrylic remains a real missing LUCiD component, but it was
not the source of the large top-cap mismatch.

A PMT-response SIREN interface was found and ported from the `refactor-v2`
branch, but the trained weight artifact and its input normalization metadata
were absent. No result here uses that SIREN. The known one-dimensional angular
response is better represented explicitly by the measured table. A SIREN may
later be useful for a residual dependence on bulb position, azimuth, magnetic
field, or other variables after a validated training artifact is available.

In plain language: we did not use a neural network to cover up a known optics
curve, and we did not load weights without knowing how their inputs were
normalized.

## 8. Wall routing: deciding which PMT a ray is allowed to test

The largest top-cap effect was not the bulb equation. It was candidate routing.
SKDetSim's GEANT geometry sends a wall-crossing ray through one local cell and
then calls `SKDONUTS` for that PMT. Earlier LUCiD logic could test several
nearby curved PMTs from a wall-grid candidate list. Near boundaries, it could
accept a neighboring bulb even when SKDetSim never called that PMT.

The identical first-scatter-ray diagnostic showed that cap-heavy LUCiD hits
existed for rays with no corresponding SK PMT-volume call. Exact SK4 PMT
positions did not remove them. Selecting only the earliest valid candidate
helped, but assigning one PMT from the detector-wall crossing was decisive.

The new `sensor_candidate_selection="wall"` mode:

1. intersects the ray with the nominal detector wall;
2. considers the PMTs associated with that wall grid cell;
3. selects the center nearest the wall-crossing point; and
4. applies the unchanged sphere-plus-torus and inactive-band tests only to that
   PMT in the hard forward pass.

For the identical 317,664 first-scatter ray bank, the accepted top-fraction gap
fell from 0.6478 to 0.1549 percentage points, a 76.1% contraction. In the full
eight-million-ray comparison, detector-wide TOF TV fell from 0.017866 to
0.014924, and top-cap conditional timing TV fell from 0.031306 to 0.016265.

In plain language: a ray crossing the wall between two PMTs should not get
several independent chances to hit whichever curved bulb is convenient. We
now choose the wall cell first, as SKDetSim does, and then ask whether that
cell's PMT is hit.

## 9. Stable long-baseline PMT intersections

After wall routing, LUCiD still appeared to have too much scattered light
relative to direct light. With local-angle response, its scattered/direct ratio
was 0.25890, compared with 0.22820 in SKDetSim. The regional distribution of
the scattered component by itself already agreed within 0.32 percentage point,
so the problem was the normalization of direct versus scattered light.

We built a direct-ray diagnostic in `LSRGEN` that records the retained SK laser
rays before water transport and the later active `SKDONUTS` intersection. It
recorded 358,929 source rays and 146,102 active SK PMT hits. Replaying these
exact origins and directions removed the source, water, QE, and response from
the test.

The old LUCiD solver accepted 128,971 rays, only 0.882746 times the SK count.
Every ray accepted by both programs was assigned to the same PMT, and exact
SK-IV PMT placement did not recover the missing hits. The missing population
was concentrated on the torus skirt.

The mathematical surface was correct. The numerical sequence was not. The
old float32 calculation formed a small discriminant and hit position using
terms set by a roughly 36 m origin and then tried to resolve a 25 cm PMT. Near
a grazing root, subtraction discarded the centimetre-scale information.

The stable solver now:

1. computes the 25.4 cm GEANT bounding-sphere roots using the closest approach
   and half-chord, avoiding the cancellation in the ordinary quadratic formula;
2. shifts the torus solve to the nearby bounding-sphere entry;
3. restricts it to the physical axial branch $0\le z<z_{\rm join}$ so it cannot
   select a root from the unwanted branch of the spindle torus;
4. samples a small fixed bracket, bisects the first outside-to-inside crossing,
   and refines it with Newton iterations; and
5. carries the locally computed torus hit position into the residual, normal,
   and output calculations instead of rebuilding it from the far-away origin.

The corrected JAX solver recovers all 146,102 SK-accepted rays when evaluated
on their assigned PMTs. The complete wall-routing replay predicts 146,214
hits, a LUCiD/SK ratio of 1.00077. The one-million-ray scattered/direct ratio
becomes 0.23049, compared with 0.22820 in SKDetSim.

The final eight-million-ray canonical TOF TV becomes 0.002833. Section
contributions are:

| Section | TV contribution |
|---|---:|
| Top | 0.000421 |
| B1 | 0.000191 |
| B2 | 0.000250 |
| B3 | 0.000185 |
| B4 | 0.000232 |
| B5 | 0.000156 |
| Bottom | 0.001398 |

In plain language: imagine locating a millimetre mark by subtracting two
36-metre tape measurements stored with limited precision. The mark can vanish
numerically even though the geometry is right. We moved the calculation next
to the PMT before solving for the small feature.

## 10. Differentiability

The physical hard decisions and the continuous geometry must be described
separately.

Within a selected sphere or torus branch, JAX differentiates the root distance,
hit position, local normal, incidence cosine, and tabulated response. The stable
rebase does not detach values from the graph: the nearby origin and local root
are both JAX expressions. Tests cover gradients with respect to ray origin,
ray direction, PMT position, PMT axis, and response inputs.

At a hard hit/miss edge, sphere/torus switch, inactive barrel band, grid-cell
boundary, or stochastic-history change, the physical function is discrete and
has no ordinary derivative. For parameter fitting, finite `temperature` uses a
Gaussian-smoothed projected-coverage lookup with JAX trilinear interpolation.
Forward-mode JVP and reverse-mode VJP agree, and reverse mode agrees with finite
differences away from physical boundaries.

Wall routing uses a hard single-PMT value in the forward validation path. At
finite smoothing temperature, its backward derivative uses the smooth
multi-candidate coverage surrogate so a small PMT/source displacement retains
a useful edge gradient. Both forward- and reverse-mode tests pass.

The current validation suite contains 93 relevant passing tests, including:

- hard-oracle versus JAX forward intersections;
- long-baseline torus roots;
- finite sphere and torus derivatives;
- full ray/PMT geometry Jacobians;
- PMT detection-table forward and reverse autodiff;
- wall-mode forward and reverse edge gradients; and
- the oblique Rayleigh-frame regression.

In plain language: the exact simulation still makes yes/no choices, as any
detector simulation must. Between those boundaries, the geometry and response
are differentiable. When optimization needs a gradient across an edge, LUCiD
uses a controlled smooth version of that edge.

## 11. Final quantitative picture

### 11.1 What agrees now

For the current Rayleigh-only sample:

| Quantity | SKDetSim | LUCiD |
|---|---:|---:|
| Total PE/pulse | 668.196 | 728.428 |
| Top fraction | 3.91394% | 3.92321% |
| Barrel fraction | 11.13907% | 11.17955% |
| Bottom fraction | 84.94698% | 84.89724% |
| Detector prompt | 147 ns | 147 ns |

The regional differences are +0.0093 percentage points on top, +0.0405 on the
barrel, and -0.0497 on the bottom. Detector-wide normalized TOF TV is 0.002833.

The current direct and Rayleigh ratios are:

| Ratio | SKDetSim | LUCiD |
|---|---:|---:|
| Rayleigh yield / direct yield | 1.012383 | 1.013128 |
| Difference relative to SK | -- | +0.0736% |

This is strong evidence that adding Rayleigh scattering changes the total
response by the correct relative amount after the PMT-intersection correction.

### 11.2 Why the current absolute yield is still high

The 9.0% absolute difference should not be described as a solved calibration.
It is also distinct from the earlier legacy-sphere ballistic result, where
660.572 LUCiD PE/pulse agreed with 660.022 SK PE/pulse to 0.083%. That earlier
closure used a simpler effective spherical sensor whose overall normalization
had already matched this direct configuration. The detailed bulb, local-angle
response, wall routing, and stable edge acceptance now form a more physical
model and expose a common absolute normalization difference in both direct and
Rayleigh runs.

The exact SK-ray replay is the critical clue: using SK's generated rays, LUCiD
and SK agree in active-hit count to 0.08%. Using independently generated LUCiD
rays, both direct and Rayleigh yields are about 9% high. Therefore the leading
remaining questions are:

1. Are the full SK and LUCiD `RNGAUSYOO` laser direction distributions exactly
   the same near the 20-inch PMT skirt, including random-number transforms and
   any rejection or normalization?
2. Does the standalone LUCiD source attach exactly 8,090-photon intensity to
   the same retained directional distribution as `LSRGEN`?
3. Are there source-position or launch-boundary conventions that cancel in the
   old spherical model but matter for the detailed skirt?

Water absorption or Rayleigh length should not be adjusted to absorb this
common direct-and-Rayleigh scale difference.

In plain language: the water now changes the light by the right relative
amount, and the same-ray PMT test works. The remaining extra light appears when
LUCiD makes its own laser rays. The clean next step is to compare those source
rays, not to retune the water.

## 12. Effects tested and ruled out as the dominant cause

The following effects are real but did not explain the large Rayleigh top-cap
discrepancy:

| Test | Result |
|---|---|
| More LUCiD propagation steps | Four to eight steps produced negligible change |
| Multiple-scatter composition | SK and LUCiD top-cap scatter orders agreed closely |
| Rayleigh free path | First-scatter probabilities and distance distributions agreed |
| Polarization-aware versus averaged Rayleigh | Explained only about 7.8% of the diagnostic top gap |
| Exact SK-IV PMT placement | Did not improve the gap relative to connection-table placement |
| Acrylic choice in the matched flat-response test | Changed top fraction by only about 0.3% relative |
| Local photocathode response | Reweighted the discrepancy but did not create the boundary excess |
| Average-response versus sampled transport | Did not remove the top-cap timing-shape difference |
| More Monte Carlo alone | Reduced noise but left the reproducible routing/root effect |

## 13. Code and reproducibility

The main LUCiD feature history is:

| Commit | Change |
|---|---|
| `eb638ea` | SK-IV PMT timing response |
| `36aa3eb` | Empirical Gaussian calibration beam |
| `9656f5c` | Measured Super-K detector geometry |
| `6bbccc9` | Hard SK 20-inch PMT oracle |
| `329a23b` | Differentiable SK sphere/torus roots |
| `abb0ce7` | Smooth SK PMT geometric coverage |
| `c931b07` | Integrated SK PMT geometry |
| `5623e47` | Geometry-gradient audit |
| `840cb6a` | Differentiable local-angle PMT response |
| `92288d9` | Monochromatic group-velocity override |
| `3c08a2d` | Local-incidence diagnostics |
| `6360c36` | Photon-transport diagnostics |
| `a9672cb` | Wall-assigned PMT routing |
| `a268c31` | Stable long-baseline PMT intersections |

Important outputs are:

- `results_rayleigh_stable_local_5000_lucid8m/canonical_tof.png`: current
  seven-section TOF comparison;
- `results_rayleigh_symmetry_audit/rayleigh_z_symmetry.png`: Rayleigh signed-z
  audit;
- `results_sk20inch_geometry_diagnostics/`: PMT cross-sections, projected
  silhouettes, detector coordinates, and gradient diagnostics;
- `results_first_scatter_diagnosis/`: first-scatter and identical-ray routing
  diagnostics;
- `sk_direct_source_geometry_100.npz`: compressed exact SK direct-ray bank;
- `sk_direct_lucid_replay_wall_stable_local.npz`: final exact direct replay;
- `lucid_sk20inch_angle_rayleigh_groupspeed_wall_stable_local_timing_8m.npz`:
  current eight-million-ray LUCiD sample; and
- `TOP_CAP_EXCESS_DIAGNOSIS.md`: detailed chronological investigation.

The raw 73 MB direct diagnostic text files and failed intermediate root-solver
outputs were removed after compression and validation. The retained artifacts
contain the source rays, accepted SK intersections, before/after LUCiD replays,
plots, numerical summaries, configuration, and executable hashes needed for
review.

## Suggested two-minute explanation

We started by making the two programs simulate an intentionally simple laser:
same source, transparent water, no scattering, no reflections, and no dark
noise. That exposed a hidden SK Mie floor, unmatched PMT efficiencies, and a
timing response already present in SK's accepted-photoelectron timestamps.
After correcting those definitions, the simple direct-light comparison was
statistically consistent.

We then added absorption and Rayleigh scattering one at a time. Absorption and
the first Rayleigh interaction agreed, but LUCiD put too much scattered light
on the top and barrel. Exact-ray replay showed that the water was not the
problem. The difference appeared when the scattered ray reached the detector:
LUCiD could try several neighboring PMTs, and its long-distance float32 torus
calculation lost valid direct hits near the bulb skirt.

We implemented the real SK sphere-plus-torus PMT, the local photocathode angle
response, one-PMT wall routing, the SK timing mixture, the 405 nm group speed,
and a numerically stable local torus solve. The final normalized TOF difference
fell by 81%, and the detector-region fractions now agree within five-hundredths
of a percentage point. The Rayleigh sampler is symmetric and the complete
geometry remains differentiable in the intended fitting mode.

The remaining 9% absolute yield difference is common to the direct and
Rayleigh samples. Exact SK-ray replay agrees, so the next clean target is the
independently generated laser source distribution and normalization. We should
resolve that before tuning any water-physics parameter.
