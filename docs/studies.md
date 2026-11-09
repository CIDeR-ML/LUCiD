# Studies and archives

The LUCiD research record lives in versioned release assets, not in the tree.
Each asset is one tarball whose extracted root carries an `INDEX.md` (what the
material was for, the commit it ran against, how to run it, and every known
defect), a `PROVENANCE.txt` and a `MANIFEST.sha256`. Each is built from one
manifest under `tools/assets/`; see "Rebuilding an asset" below.
Every asset is under CC BY 4.0 (its `INDEX.md` says so); the code in this
repository is under the terms of `LICENSE`.

No path inside an asset may be cited from `lucid/`, `analysis/` or `tests/` —
a citation names the asset tag and this page instead. If you came here from a
code comment or a docstring, the row below is where that material went.

Every row is what `tools/make_asset.sh --row <manifest>` prints
for one manifest under `tools/assets/` (its `summary:`, `size:` and `commit:`),
so a change goes into the manifest and the row is regenerated.

| asset | what it was for | size | ran against |
|---|---|---|---|
| detector-geometry-sources | the WCSim geofile dumps behind the tracked npz | 5.5 MB | eaf2e89f |
| legacy-io-archive | read HDF5 files written before format 6 | 176 KB | eaf2e89f |
| lucid-data-v1 | the first-run bundle: the four SIREN emitters' weights + 5-event 1 GeV and 6-event 1.5 GeV GEANT4 muon slices and the 12 overlap tables the shipped code loads | 142.2 MB (141.7 MB tarball) | 7e3452a5 |
| lucid-paper-results-v1 | the paper figures' inputs: the nrays, energy and geometry tracking scans and the cached data of the other figures | 299 MB (84.2 MB tarball) | 7a913a09 |
| studies-archive-absorption-siren | the student's SIREN-as-absorption-field project and its merged-quality twin | 6.3 MB | 97c9076a |
| studies-archive-adam-optimizer | the retired 5-stage Adam recon pipeline, its 3-D displays and its composite losses | 36 KB + notebooks | eaf2e89f |
| studies-archive-calib-campaign-2026-09 | the 2026-09 calibration estimator campaign + the `schur_gn` arm it is the only user of | 195 KB | 71f00e49 |
| studies-archive-calib-closure-2026-10 | the calibration-closure figure and its plot module, as committed on paper-rerun-qe | 21 KB | 2b03c27b |
| studies-archive-geometry-v2-2026-10 | the V2/CSG research record -- the gradient ladder, the benchmarks, the NaN audit, the diaries | 4.6 MB in 520 files (1.7 MB gzipped) | 85e98bd1 |
| studies-archive-gifs-2026 | the three event/optimisation animations | 13.2 MB | eaf2e89f |
| studies-archive-icecube | the IceCube/`ppc` measurement programme + the legacy string engine + the sizing maths | 3.7 MB (plus 144 MB of run outputs on /sdf/data, not in the tarball) | 958807f6 |
| studies-archive-lartpc | the LArTPC optical package and its calibration arc | 2.9 MB | 85545f72 |
| studies-archive-method-ab-2026 | the expected-vs-sampled A/B driver | 6 KB | eaf2e89f |
| studies-archive-notebooks-2026 | the S-class half of `good_notebooks/` + the tutorials + `s3df_jobs/` | 1.5 MB | eaf2e89f |
| studies-archive-notebooks-legacy-2026 | the 24 pre-rename `notebooks/` + `wcte_bldg157.geo` | 18.3 MB | c799b404 |
| studies-archive-string-telescope-2026 | the string event producers + the standalone three.js viewer | 56 KB | eaf2e89f |
| studies-archive-tracking-diagnostics-2026-07 | the 2026-07 track-fit diagnostics and the TrackingPipeline branches they drove | 225 KB | 252f89b8 |

`lucid.fitting.schur_gn` itself is still in the library; the calibration-campaign
asset carries the campaign that was its only user.

Large run artefacts do **not** travel in an asset: 144 MB of IceCube run
outputs, the 282 MB CRB-appendix data and 1.6 MB of `pool_bench/out/` run
artefacts are kept outside the repository, and each asset's `INDEX.md` records
where.

## Rebuilding an asset

Each asset is described by one manifest under `tools/assets/`, and
`tools/make_asset.sh` builds it:

```bash
LUCID_ASSET_OUT=<somewhere with room> tools/make_asset.sh tools/assets/legacy-io-archive.manifest
```

The script refuses to build an asset whose manifest carries no `INDEX.md`
body, and refuses one whose `commit:` does not resolve — a study tarball with
no provenance is worse than a deleted directory, because it looks runnable.
The script touches no remote. `tools/publish_assets.sh <dir>` then publishes
every built asset in `<dir>`: it checks each tarball against its `.sha256` and
the two installer assets against `data/MANIFEST.sha256` and
`analysis/paper/RESULTS.sha256`, prints one `gh release create` per asset
(a dry run), and with `--go` creates the releases on `main`, skipping any that
exist. A published asset is never replaced; a changed one is a new tag.

`lucid-data-v1` and `lucid-paper-results-v1` carry material that is on no git
ref: the trained SIREN weights, the PhotonSim ROOT slices, the git-ignored
overlap tables and the paper's results. Their manifests name those parts
relative to the directory in `$LUCID_ASSET_SRC`, so no tracked file carries
anyone's absolute path.
