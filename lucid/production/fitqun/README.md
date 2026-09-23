# Tuning fiTQun to LUCiD

fiTQun reconstructs a water-Cherenkov event by comparing it against an
analytic prediction of the charge and time at every PMT. That prediction is
driven by five tuned tables. This package produces the LUCiD-derived inputs to
each of them, so fiTQun can be tuned to LUCiD's detector response the same way
it is normally tuned to WCSim's.

## Where the boundary is

Each module writes the file the corresponding stage of `fiTQun/Utilities`
already reads. The fitting code that turns those histograms into fiTQun's
parametrised `const/` files stays where it is — it is the part the
collaboration has validated, and re-implementing it would only add a second
thing to keep in step.

| Table | What LUCiD writes | Consumed by |
|---|---|---|
| Cherenkov profile | `CProf_<pdg>_WCSim.root` | `fiTQun_shared::LoadProfiles` directly |
| Charge PDF | `<mu>_pdf.root` per mu | `chrgpdf/gen2d.cc` → `fitpdf` → `MakecPDFparFile.cc` |
| Angular response | `angRespAll_<r>` histogram | `angular/fit_cos.C` |
| Direct time PDF | `<cell>_hist.root` | `timepdf/combhists.cc` → `fittpdf.cc` |
| Indirect light | `scattables.h5` | `tools/fitqun/h5_to_scattable.C` → fiTQun |

`rootio.py` is the seam that lets a non-ROOT codebase write those files. It
covers `TH1D`/`TH2D`/`TH3F` through uproot and adds a `TGraph` writer, which
uproot does not have. The one thing it cannot do is `TScatTable`, a
user-defined class whose streamer info uproot cannot synthesise — hence the
HDF5 hand-off for the scattering tables.

## Three things to know before running any of this

**Momentum, not energy.** fiTQun's grids are momentum grids. PhotonSim takes
`/gun/momentumAmp`, provided by G4's own particle-gun messenger and the same
command WCSim's tuning macros use, so the grids transfer unchanged. No
PhotonSim change is needed for any of this.

**Never copy process *numbers* between physics lists.** The reference WCSim
macros disable processes `7` and `8` for `mu-`, which there are decay and
capture. In PhotonSim's list index 8 is **`Cerenkov`**, so the same macro
deletes the muon's own light and leaves only delta-ray light: a profile with
the wrong angle and a sixth of the yield, which looks perfectly plausible.
`macros.py` names processes instead, and a test asserts nothing it generates
switches Cerenkov off.

**Direct light comes from configuration, not a special build.** The reference
uses a WCSim fork (`/fqTune/mode killScatterRef`) to kill scattered and
reflected photons. The LUCiD equivalent is to run with scattering and
reflection off in the physics config.

## Running it

### Cherenkov profile — particle and water only, no detector

```bash
# Fan out one job per (particle, momentum) cell; run from the submit host.
python3 lucid/production/jobs/fitqun/generate_jobs.py \
    -c lucid/production/jobs/fitqun/configs/water_mu.json -s

# Then merge the cells into the table fiTQun loads.
python -m lucid.production.fitqun cprofile build \
    $OUTPUT_BASE_PATH/fitqun_cprofile/13/*/cell.npz \
    --pdg 13 -o CProf_13_WCSim.root
```

Each job deletes its raw photon list as soon as the reduction succeeds: a
high-momentum cell is hundreds of MB of photons and a few tens of kB reduced.

### Charge PDF — digitizer only, no geometry

```bash
python -m lucid.production.fitqun chargepdf scan \
    -o chrgpdf_out --n-pmt 2000 --n-events 80 --model ski
```

Runs in one process. It drives `lucid.simulation.digitizer` rather than
re-deriving the response, so a change to the detector electronics shows up in
the next tune instead of silently diverging from it.

### Angular response, time PDF, indirect light

These three need LUCiD simulation output and are reductions over it
(`angular.measure`, `timepdf.TimePdfAccumulator`, `scattable.ScatTable`),
driven from the usual detector + physics config pair. They are detector
specific; the first two are not.

## Two choices worth reviewing

**The charge-PDF discriminator.** LUCiD applies it to the integrated
photoelectron count; WCSim applies it to the smeared charge. The difference is
visible in the `P(unhit | mu)` coefficients `gen2d.cc` fits — with LUCiD's
convention `P(hit) = 1 - exp(-mu)` exactly. This is left as LUCiD has it,
because the point of the exercise is a fiTQun tuned to LUCiD's response.

**The time PDF's `log10(mu)` axis.** The reference links against fiTQun and
calls `Get1Rmudist`, which needs the Cherenkov profile and charge PDF to have
been tuned already. `timepdf` takes LUCiD's own expected charge instead, which
makes the first pass self-consistent and removes the circular dependency. For
a second iteration indexed by the tuned fiTQun's mu, pass that in as `mu` and
nothing else changes.

## Provenance

The grids are the reference tune's, kept as data files rather than
transcribed:

| File | Source |
|---|---|
| `data/cprofile_momenta.dat` | `Utilities/cprofile/CprofileMomRepList.dat` |
| `data/charge_mu_bins.txt` | `Utilities/chrgpdf/workdir/mutbl.txt` |
| `data/charge_q_bins.txt` | `Utilities/chrgpdf/workdir/qbins_sk1.txt` |
| `data/timepdf_momenta.json` | recovered from `WCSim_v1.12.19/Utilities/TuningFiles/timepdf` |

The reference tune itself (code, worked example, and the finished WCTE
`const/` files) is at
`/eos/project/n/neutrino-generators/cjesus/fitqun_inputs/`.
