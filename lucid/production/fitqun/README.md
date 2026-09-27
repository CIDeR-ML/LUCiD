# Tuning fiTQun for a LUCiD detector

fiTQun needs nine files and fifteen scalars before it will reconstruct anything,
and every one of them is detector- and water-model-specific. This directory
generates them from LUCiD output for *any* detector, so bringing up a new one is
a matter of running the pipeline rather than rediscovering the reference chain.

Nothing here is a one-off measurement. If you find yourself hard-coding a number
for a specific detector, that number belongs in the manifest instead.

## How it fits together

```
LUCiD sample (HTCondor)                  the four generators
  |                                        |
  |  sensor/hits/labl/step .h5             | names.py   -- what fiTQun calls each file
  +--> sample_propagate --> shard.npz -----+ manifest.py -- what we produced, and its status
  |                                        |
  +--> truth.py --> truth.txt              v
  |                                   tune_manifest_<detector>.json
  +--> lucid_to_wcsim --> events.root      |
                                           +--> report.py  -- tables and plots
                                           +--> install     -- copy to fiTQun's const/
                                           +--> overrides   -- the .parameters.dat scalars
```

Three modules carry the contract, and everything else is replaceable:

| module | answers |
|---|---|
| `names.py` | what fiTQun *requires*, and exactly what each file must be called |
| `manifest.py` | what we *produced*, where it is, and whether it can be trusted |
| `report.py` | renders the manifest as tables and plots; knows nothing about physics |

**Every generator records its own product in the manifest.** That is the rule
that keeps this working. Knowledge about an input exists at the moment it is
produced -- how many events, which config, what was cross-checked -- and it is
either written down then or lost. Do not add a script that scans directories
afterwards trying to guess what is there.

## Status is data, not a comment

A real tune is never simply "done": it carries borrowed files and untuned scalars
for a long time. `manifest.Artifact.status` records that, so the set of known
compromises is queryable instead of living in someone's memory:

| status | meaning |
|---|---|
| `adopted` | measured here, in use |
| `borrowed` | in use, but produced for another detector or photosensor -- replace next |
| `measured_not_adopted` | measured here, measurement sound, deliberately not used. `note` **must** say why |
| `reference_default` | taken from the reference tune, no measurement of our own |
| `missing` | fiTQun requires it and we have nothing |

`manifest.verify()` compares the manifest against `names.required_files()` and
returns exactly this breakdown. If it reports `unused`, some generator built a
filename by hand instead of through `names.py`.

## Bringing up a new detector

```bash
source lucid/production/jobs/user_paths.sh
M=$FITQUN/tune_manifest_<DET>.json

# 1. Samples. One electron bomb feeds the angular response and both scattering
#    tables; muon/electron momentum grids feed the time PDF.
python3 -m lucid.production.jobs.fitqun.generate_sample_jobs \
        -c lucid/production/jobs/fitqun/configs/sample_<det>.json -s
python3 tools/fitqun/make_timepdf_grid.py \
        --chart $FITQUN/Utilities/timepdf/chart_13.txt --pdg 13 --scale 0.085 \
        -o lucid/production/configs/GeV/tpdf

# 2. Inputs, in dependency order. Each records itself in $M.
#    cprofile: genhist -> integcprofile -> fitcprofile -> writecprof  (FOUR steps)
#    charge PDF, angular response, 6D scattering table, then the time PDF.

# 3. Check what you have before trying to reconstruct.
python3 -m lucid.production.fitqun.report $M

# 4. Install and reconstruct.
bash tools/fitqun/run_reconstruction.sh <sample-dir> <n-events> <tag>
python3 tools/fitqun/fq_resolution.py --fq <fq.root> --labl ... --step ... \
        --pdg 13 --require-fc
```

Always score with `--require-fc` and quote the event count. Sanity-check before
believing a number: a vertex resolution near the detector size, a direction
resolution near 90 deg, or a large momentum bias means the fit is broken, not
that performance is poor.

## Things that cost days to find

Each of these fails silently or misleadingly. `tools/fitqun/reference/PATCHES.md`
carries the same list for changes that live in the reference tree.

**Sample and geometry**

- `sensor_idx` indexes the sensor file's own `config/sensor_positions`, which is a
  *different permutation* from `sk_geometry.npz` -- 0 of 11096 rows in common.
  Export the geometry with `export_geometry.py --sensor` or every hit is
  attributed to an unrelated PMT. The symptom is a charge-weighted `<cos>` of
  about -0.87 where the geometry requires about +0.7.
- The primary is `track_idx == 0`; `primary_track_ids_data` points at a delta ray.
  `primary_energies_data` is **kinetic**.

**Writing the WCSim file**

- Hit times must be shifted into the trigger frame (`--time-offset-ns`, default
  950). fiTQun fits a 900-1400 ns window and silently discards everything
  outside it.
- WCSim reserves tracks 0 and 1 for the beam and target, so the primary must be
  **track 2** (`makehistWCSim.cc:34`). Writing only the primary segfaults the
  tuning chain on `At(2)`, while reconstruction -- which never reads truth --
  works fine.

**The tuning chain**

- The Cherenkov profile chain has **four** steps, not two. `makehistWCSim.cc:127`
  hardcodes `fFitCProf=true`, so it needs `CProf_<pdg>_fit_WCSim.root` from steps
  3-4 and ignores `fiTQun.UseFitCProfile`. Stopping at step 2 is what forces a
  tune to carry `UseFitCProfile = 0`.
- `makehistWCSim` shipped with `SetWAttL(6800.)` and `SetQEEff(0.1)` overwriting
  the parameter file. Removed -- otherwise the time PDF is trained at SK's optics
  whatever the tune says.
- `isct = nScatter + 1000*nReflection`. `AttenL` cuts `isct < 1000`, i.e. it
  **excludes reflected photons**, so run its sample with reflections off. The
  scattering table splits on `isct != 0`, so it **includes** them -- LUCiD's
  `indirect = scatters | reflects` matches it directly.
- `fitpdf.cc:52` requires every charge-range boundary to fall exactly on a bin
  centre, so the q axis cannot be rebinned freely.
- `QEEffCorr` is assigned only inside `if(IsItRealData())`, so it cannot correct a
  simulated sample. The MC charge scale has to go in `QEEff`.

**Cost and scheduling**

- Histogram statistics are **additive**: `mergehists.pl` sums and `combhists`
  re-bins the sum. So generate the *full momentum range* at shallow depth first
  and top up by resubmitting the same configs. Momentum gaps cannot be patched
  later; depth can.
- The reference time PDF trains on 840k mu + 860k e over 130-7000 MeV/c, which is
  about 5700 CPU-h at LUCiD's measured 12.1 s/event. The reference's own charts
  imply 12.6-15.9 s/event, so this is the intrinsic cost of the product, not
  LUCiD overhead.
- Submit on `workday`. `condor_qedit MaxRuntime` does **not** lift the limit a
  running job was matched with -- jobs die at the original cap with exit 143.
- Size jobs from the measured 12.1 s/event. A config claiming 4.833 undersizes
  them by 2.5x.
- Pass `dataprod_fanout -c` an **absolute** config path; a relative one does not
  resolve inside the container and every job dies in 25 s.

## Layout on LXPLUS

Source and logs on AFS, container image and all ROOT output on EOS. Standard
LXPLUS schedds reject `/eos` paths in a submit file's `output`/`error`/`log`, so
logs must land on AFS while `--output-dir` stays on EOS.
