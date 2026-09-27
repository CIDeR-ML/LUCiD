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

## `timepdf/makehistWCSim.cc` — 2 lines removed

Directly after `ReadSharedParams`, the upstream macro does

    fqshared->SetWAttL(6800.);
    fqshared->SetQEEff(0.1);

which overwrites the two values `fiTQun_shared`'s constructor has just read from
the parameter override file (`fiTQun_shared.cc:264,291`, via the
`<key><DetName>` lookup). Left in, every time PDF is built at SK's attenuation
length and charge scale whatever the tune says — so a tune for another detector
or water model would be trained against the wrong optics and no log line would
say so. `fiTQun.cc:99` comments out the identical overwrite for this reason.

Both lines are deleted so the values come from the tune. Nothing else reads them.

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

## What lives where, and how to re-apply it

**We have no push access to fiTQun, so these fixes are ours to carry
indefinitely.** They live as patches in `patches/`, generated against pinned
upstream commits, with `patches/apply.sh` to re-apply them to a fresh checkout:

    tools/fitqun/reference/patches/apply.sh /path/to/fitqun-tree

Patches rather than whole-file copies, for three reasons: the diff *is* the
documentation of what we changed and why; a copy silently reverts an upstream
improvement while a patch conflicts loudly; and 88 lines of patch is reviewable
where four whole files are not.

Pinned upstream commits (`apply.sh` warns if the checkout has moved):

| repo | commit |
|---|---|
| `fiTQun/Utilities` | `c0a0916` |
| `fiTQun/WCSimFQTuner` | `bfd18d3` |
| `fiTQun/fiTQun` | `752bfb6` |

Upstream sources use CRLF and the patches are LF-normalised, so `apply.sh`
converts each target before matching -- `patch -l` alone does not reconcile
CRLF context lines.

Verified: applying all four patches to pristine upstream reproduces our working
copies byte for byte.

The `.cc` files alongside this document are the *resulting* sources, kept for
reading. `fitqun_SK_WAND.parameters.dat` is our own tune file, not a patch.
