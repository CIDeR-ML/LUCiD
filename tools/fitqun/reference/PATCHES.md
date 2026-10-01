# Reference tuning macros, as they must be run under ROOT 6.30

The fiTQun tuning chain (`fiTQun/Utilities`) was written against an older ROOT.
Under the container's ROOT 6.30 / gcc 11, five of its steps fail — three of them
*silently or misleadingly*, which is why the fixes are recorded here rather than
rediscovered. Files in this directory are the upstream macros with the source
changes applied; the upstream clone stays the provenance.

Anything not listed below is byte-identical to upstream.

## `gen2d.cc` — unchanged source, must be **compiled**

Interpreted, it aborts with

    ERROR: Bad Phit fit. k = 3 p_k = 0.014708

which the reference's own `Tuning-the-photosensor-charge-response.md` says to
contact the author about. It is not a bad fit: the test is
`!(p_k+1e-3>=0. && p_k<=1.-1e-3)`, and 0.0147 satisfies it. Instrumenting the
loop shows all six coefficients passing and the error firing anyway — cling
mis-evaluates the expression. Compiled with ACLiC the same macro yields
identical coefficients and no error, so run it as:

    root -l -b -q 'gen2d.cc+'

## `plotChrgPDF.cc` — 3 lines

* `TH2D *htmp=hst2d_type0;` relied on ROOT injecting a global per object in an
  open file. Modern cling does not, so the macro would not parse. Replaced with
  an explicit `_file0->Get("hst2d_type0")`.
* `c1->Print(...)` assumed an implicit current canvas, which a batch macro has
  none of. Creates one from `gPad` or afresh.

## `fitpdf.cc` — 1 line added

`GetmuThresh` is defined below its first use. Cling resolves a call only against
declarations it has already seen, so a forward declaration is prepended. It also
needs fiTQun's own classes, which a static library cannot provide to the
interpreter — build a shared one first:

    g++ -shared -Wl,--allow-multiple-definition -o libfiTQun.so *.o ...

and load it with `-I$FITQUN_ROOT` plus `#define HEMI_CUDA_DISABLE 1` before
including `fQChrgPDF.h`, or the include drags in CUDA types.

## `TPolyFunc.{h,cxx}` — 18 lines

The `TF1(name, this, &memFn, xmin, xmax, npar, "className")` constructor is gone
in ROOT 6.30; the surviving template takes `ndim` in that position and requires
the member function to take `const Double_t*`. So:

* `,"TPolyFunc")` → `,1)`
* `myPoly` and `SetSubParameters` take `const Double_t*`
* `SetSubParameters`'s `gpars` becomes a `const` pointer with a separate owned
  buffer for the branch that fills it

Without these `fit_cos.C` cannot build its fitting function at all.

## `fit_cos.C` — unchanged source, load `TPolyFunc` first

The macro does `gROOT->ProcessLine(".L TPolyFunc.cxx+")` *inside* the function,
but cling needs the type when it parses the body. Run as:

    root -l -b -q -e '.L TPolyFunc.cxx+' -e '.x fit_cos.C("angResp_100.root","<config>_<pmttype>")'

Note it fits the hardcoded `angRespAll_100`, so the 100 cm shell must be
populated — which requires sources filling the volume out to the PMT faces.

## `fiTQun/config.gmk` — 1 line (not in this directory)

`AR = ar clq` fails on modern binutils, which reads `l` as `libdeps`:

    ar: libdeps specified more than once

Use `ar cq`.

## Known hazard, not yet fixed

`fitpdf.cc` fits verbosely; its log reached 230 MB for one PMT type. On a
quota'd filesystem that is a real risk. Drop the `"V"` from the `Fit` calls, or
redirect the log somewhere expendable.

## `MakecPDFparFile.cc` — output needs an adapter, not a patch

The merge writes both photosensor types into one file, keeping the type in every
object name (`hCPDFrange_type0`, `gParam_type0_Rang3_2`, `hPunhitPar_type0`).
`fQChrgPDF::LoadParams` (fQChrgPDF.cc:55-79) reads **un-suffixed** names and is
called once per type with a different file each time, so it finds nothing in that
output and segfaults on the first unguarded dereference -- indistinguishable from
the symptom of an empty `cPDFpar`.

`tools/fitqun/cpdfpar_for_type.C` copies one type out under the loader's names.
It also copies the range index `nRang` *inclusive*, because the loader's loop is
`for (iRang = 0; iRang <= nRang; ...)`.

## `timepdf/makehistWCSim.cc` — 1 line removed, 2 kept deliberately

Directly after `ReadSharedParams`, the upstream macro does

    fqshared->SetWAttL(6800.);
    fqshared->SetQEEff(0.1);
    fqshared->SetPhi0(-1.,1.);

Only the first is wrong for us. It overwrites the attenuation length
`fiTQun_shared`'s constructor has just read from the parameter override file
(`fiTQun_shared.cc:264`, via the `<key><DetName>` lookup), so every time PDF
would be built at SK's 6800 cm whatever the tune says, and no log line would
say so. **`SetWAttL` is deleted. `SetQEEff(0.1)` and `SetPhi0(-1.,1.)` must
stay.**

The reason they stay is that they are not physics, they are the **normalisation
convention of the charge axis**. The histogram is filled in
`log10(Phi0*QEEff*nphot)`, and `fiTQun.cc:466` (`ftdir`, the direct-light time
likelihood) divides the real normalisation back out and multiplies the same
literal `0.1` back in when it looks the PDF up:

    double logmu = logfunc(mu[iring][icab]*0.1
                     /(Phi0_local[PID]*fqshared->GetQEEff()*QEEffCorr))*ln10recp;

So building with the tune's own QEEff is the bug, not the fix: it shifts the
axis by `log10(QEEff_tune/0.1)` against the axis fiTQun evaluates on — for our
0.01556 that is -0.81 decades, a factor 6.4 in charge, silently.

Two consequences worth stating. `fiTQun.cc:99`'s commented-out
`SetQEEff(0.1086)` is a *different* quantity, the absolute charge scale of the
forward model, which does correctly come from the parameter file; it is not the
same override and does not argue for deleting this one. And because the fitted
QEEff is divided out at evaluation time, **a time PDF stays valid across QEEff
refits** — rebuilding it after a refit is unnecessary.

### Time origin — two terms added to `tc`

The corrected hit time the PDF is binned in must be measured from the photon's
emission, so everything between the generator's clock and the PMT has to come
off. Upstream subtracts the trigger offset and the times of flight; two terms
are missing for a LUCiD sample:

    if (!aSubToffs) aSubToffs = 950 - trigOffset;
    double tc = digiHit->GetT() - aSubToffs - TrkParam[3]
                - RmidPMT*nwtr/c0 - smid/c0;

* `aSubToffs` comes from `wc->GetTOffset()`, which is 0 when the converter
  writes a single gate per event. Falling back to `950 - trigOffset` puts the
  origin at the same place the multi-gate path does — 950 ns is the offset
  `lucid_to_wcsim` references every gate to.
* `TrkParam[3]` is the true interaction time. The reference subtracts it
  (`makehist.cc:199`) for samples not generated at t=0, and LUCiD randomises it
  per event, so leaving it in smears the PDF by the full gate width.

Without both, the time PDF is sharp but offset, and the resulting vertex
resolution is roughly 6x worse (182 cm against 31 cm on a 1 GeV mu- sample)
with nothing in the fit output indicating why.

## Building `makehistWCSim` against a standalone `libWCSimRoot`

`config.gmk:60` expects `$(WCSIMDIR)/include` to hold the WCSim headers directly
and `-L$(WCSIMDIR)` to find the library. A standalone WCSimRoot build installs
them under `include/WCSimRoot/` and `lib/`, so point a shim directory at both:

    mkdir -p $F/wcsim_shim
    ln -sfn $F/wcsim_build/include/WCSimRoot $F/wcsim_shim/include
    ln -sfn $F/wcsim_build/lib               $F/wcsim_shim/lib
    ln -sfn $F/wcsim_build/lib/libWCSimRoot.so $F/wcsim_shim/libWCSimRoot.so
    make makehistWCSim WCSIMDIR=$F/wcsim_shim   # needs FITQUN_ROOT set too

## The Cherenkov profile chain has **four** steps, not two

`makehistWCSim.cc:127` calls `ReadSharedParams(..., fFitCProf=true, ...)` with the
flag hardcoded, so the tuning chain requires `CProf_<pdg>_fit_WCSim.root` and
ignores `fiTQun.UseFitCProfile`. Per `Utilities/cprofile/README.txt` that file is
the output of steps 3 and 4:

    integcprofile <pdg>   # step 2 -> CProf_<pdg>.root
    fitcprofile   <pdg>   # step 3 -> CProf_<pdg>_fit_out.root
    root -l -b -q 'writecprof.cc(<pdg>)'   # step 4 -> CProf_<pdg>_fit.root

`fitcprofile` reads `CProf_<pdg>.root` from the cwd, so symlink the `_WCSim`
file to that name. Install the result as `CProf_<pdg>_fit_WCSim.root`, the name
`fiTQun_shared.cc:1683` builds when `WCSimConfig` is set. Stopping at step 2 is
what forces a tune to carry `UseFitCProfile = 0`, and `runfiTQunWC` never
complains because reconstruction does not read the fitted profile.

## `cprofile/writecprof.cc` — 1 declaration added

Step 4 of the Cherenkov profile chain aborts under ROOT 6.30 with

    error: use of undeclared identifier 'fconn'

five times over. `FitConPoly` does `fconn = new TF1(...)` with no declaration,
relying on CINT creating a global from a bare assignment. Modern cling does not,
the same failure as `plotChrgPDF.cc`. Declared it `TF1 *fconn`; it is used only
inside that function.

Worth knowing because the failure is *silent in the pipeline*: `fitcprofile`
(step 3) takes ~93 min and succeeds, writing a multi-GB `_fit_out` file, and
only then does step 4 fall over — so a driver that runs both in one job appears
to do 93 minutes of useful work and produce nothing.

**Run the CProf chain on EOS, not AFS.** `CProf_<pdg>_fit_out.root` is ~2.8 GB
per particle, against an AFS work quota of ~100 GB that also holds the
checkouts. The final `CProf_<pdg>_fit.root` is ~50 MB and is the only one worth
keeping; delete the intermediates once step 4 has run.

## `timepdf/fittpdf.cc` — 3 fixes

The shipped file does not compile under ROOT 6.30.

* **line 176 is a literal typo**: `<< lowRange < < " " << hiRange`. Two separate
  `<` characters where `<<` was meant. Nothing about this is ROOT-version
  specific -- the file as distributed cannot compile.
* `ntmp` is declared inside the momentum loop (`int ntmp=hmeantmp->GetNbinsX()`)
  and used after it, at line 255. Hoisted to the enclosing scope.
* `if (PID==13) nmom = 24;` hardcodes the momentum-point count for mu and pi,
  which over-runs a `hist_tpdf` built from fewer points -- which is what a
  partial grid produces. Clamped with `std::min(nmom, 24)`.

### Bad per-slice Gaussian fits are now dropped

Upstream has the cut written and commented out. It is enabled, and extended to
clear the width histogram as well as the mean:

    if (!(dtmp>tcmin && dtmp<tcmax) || chi2tmp > 20) {
      hmeantmp->SetBinContent(ibin,0.); hmeantmp->SetBinError(ibin,0.);
      hsigmtmp->SetBinContent(ibin,0.); hsigmtmp->SetBinError(ibin,0.);
    }

Each log10(mu) slice is fitted with a Gaussian; a slice with too few entries
returns a nonsense mean and width with small errors, which then pulls the
polynomial in mu. Clearing only the mean (as upstream would) leaves the bad
width in place.

### Momentum fit ignores the per-node errors by default

The signature is

    int fittpdf(int PID, bool flogfit=false, bool fNoErrorbars=true)

and `build_timepdf.sh` / `tpdf_combine.sh` both call `fittpdf.cc($PDG,0,1)`, so
the third argument is 1 and the momentum graphs are built with NULL errors even
though `arparerr[][][]` already holds each node's Gaussian-fit error. Passing 0
uses them.

Measured: on a 14-node grid it changes nothing, because that fit is singular
rather than merely ill-weighted (`TDecompChol::Decompose: matrix not positive
definite`, `TLinearFitter::Eval: Matrix inversion failed`) -- 14 points against
the 10 coefficients `ntpdfppar` hardcodes. The outputs are bit-identical with
errors on and off, and `gtcsgpar_0` (the Gaussian width at 1 pe) comes out
negative at 3 of the 14 nodes. Weighting is worth switching on, but it is not a
substitute for enough momentum nodes; see `check_tpdfpar.C`, which now refuses
both conditions.

## `fiTQun/runfiTQun.cc` — truth seeding for WCSim input

`TuningMode != 0` is supposed to pin all eight track parameters at truth, so the
charge scale cannot be absorbed into momentum. On WCSim input it did not: the
`NOSKLIBRARIES` branch was left stubbed at vertex `(0,0,0)` and momentum
`(350,350,350)`, i.e. `|p| = 606.2 MeV/c`, `theta = acos(1/sqrt3)`, `phi = pi/4`
-- the same fictitious track for **every event**. Truth seeding was only ever
wired to SKDETSIM's `vcwork_` common block. Every `QEEff` measured this way is
meaningless, and nothing warns.

Two fixes, both mirroring what `makehistWCSim.cc` already does:

* **Position and momentum** from the WCSim primary. WCSim reserves tracks 0 and
  1 for beam and target, so the primary is `GetTracks()->At(2)`
  (`makehistWCSim.cc:87,140,214`), and the vertex comes from the *trigger*, not
  the track (`makehistWCSim.cc:197`). Bails out loudly if no truth track exists.
* **Track time in the hits' frame**: `GetTime() + 950. - GetHeader()->GetDate()`,
  as `makehistWCSim.cc:276` does. WCSim splits the time frame three ways -- the
  track carries its raw time, digits carry the sub-event shift, the trigger
  header's `Date` reconciles them. Without this the truth track sits ~950 ns
  before its own hits (measured median residual +927 ns), and since
  `FitQEEffWrapper` minimises the **full** likelihood including time, the fitted
  `QEEff` is wrong rather than merely noisy.

This depends on the converter writing the trigger time into the header `Date`
field; `tools/fitqun/lucid_to_wcsim.cc` wrote `0` there until it was fixed.

## `fiTQun/fiTQun.cc` — per-PMT prediction dump

Nothing exposed fiTQun's predicted charge per PMT, only event-level aggregates,
and `mutot` (all PMTs + dark) is not comparable to `fqtotq` (hit PMTs only) --
which is how earlier diagnoses went wrong. Gated on `FITQUN_PMT_DUMP`, `FitQEEff`
now writes one row per PMT per event:

    event icab flgHit flgMask R_cm costheta0 mu muscat chrg tHit tau tres qeeff

`mu` (direct) and `muscat` (indirect) stay separate so a run can be split by
which term dominates without re-running. This is what `tools/fitqun/ablate.sh`
and `ablation_report.py` consume to test the closure identity `E[q] = mu`
differentially -- **the committed harness does not function without this patch.**

Known limitation: `mu_dark` is added at `fiTQun.cc:617` (`mutmp += mu_dark`) and
never stored in the `mu`/`muscat` arrays, so the dump understates the prediction
on near-zero-`mu` PMTs.

## What lives where, and how to re-apply it

**We have no push access to fiTQun, so these fixes are ours to carry
indefinitely.** They live as patches in `patches/`, generated against pinned
upstream commits, with `patches/apply.sh` to re-apply them to a fresh checkout:

    tools/fitqun/reference/patches/apply.sh /path/to/fitqun-tree

Patches rather than whole-file copies, for three reasons: the diff *is* the
documentation of what we changed and why; a copy silently reverts an upstream
improvement while a patch conflicts loudly; and ~160 lines of patch is reviewable
where six whole files are not.

Pinned upstream commits (`apply.sh` warns if the checkout has moved):

| repo | commit |
|---|---|
| `fiTQun/Utilities` | `c0a0916` |
| `fiTQun/WCSimFQTuner` | `bfd18d3` |
| `fiTQun/fiTQun` | `752bfb6` |

Upstream sources use CRLF and the patches are LF-normalised, so `apply.sh`
converts each target before matching -- `patch -l` alone does not reconcile
CRLF context lines.

Verified: applying all six patches to pristine upstream reproduces our working
copies byte for byte.

The `.cc` files alongside this document are the *resulting* sources, kept for
reading. `fitqun_SK_WAND.parameters.dat` is our own tune file, not a patch.
`runfiTQun.cc` and `fiTQun.cc` are deliberately **not** vendored -- at 43 KB and
191 KB they would dominate the directory, and their patches are small and
self-contained enough to read on their own.
