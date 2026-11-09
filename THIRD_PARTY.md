# Third-party inventory

The third-party files this repository vendors, the third-party libraries it loads at runtime,
and the third-party layers the published container redistributes: for each, its version, origin
and licence, why it is here, and what is still outstanding.

LUCiD's own terms are in [`LICENSE`](LICENSE) (currently: all rights reserved). Licence texts are
in [`LICENSES/`](LICENSES/) (§7). The rule this file tracks: no vendored third-party asset without
its licence text beside it, and an `integrity=` hash on every CDN `<script src>` or import-map
entry. The rows below say where that does not hold yet.

## 1. Vendored source (in the tree, redistributed on every clone)

| component | version | origin | licence | in-tree licence text? | why it is here |
|---|---|---|---|---|---|
| `viewer/colormaps.js` | matplotlib viridis / plasma / inferno, 7 hex stops each | https://matplotlib.org | **CC0-1.0** (the viridis family was public-domain-dedicated by its authors) | [`LICENSES/CC0-1.0.txt`](LICENSES/CC0-1.0.txt) | perceptually uniform colour ramps in the WebGL viewer | **no obligation**: seven stops are not a substantial reproduction, and CC0 requires nothing anyway |

The fiTQun / WCSimFQTuner reference C++, its patches and `fitqun_SK_WAND.parameters.dat`
(Super-Kamiokande collaboration material with no licence) are not in the current tree; they are
in the separate fiTQun repository. They remain in this repository's git history under
`tools/fitqun/reference/` and `lucid/production/fitqun/data/`, which is not being rewritten.

## 2. Loaded at runtime from a CDN (none carries an `integrity=` hash)

Two packages, six `<script src>` / import entries. Without subresource integrity, a compromised
CDN or a mutated published version runs in the browser of anyone who opens the viewer, with no
warning. The licences are not the problem; the missing hashes are.

| component | version | loaded by | licence | needs |
|---|---|---|---|---|
| three.js (`three.min.js`, `OrbitControls.js`, `LineSegmentsGeometry.js`, `LineMaterial.js`, `LineSegments2.js`) | **0.128.0** | `viewer/index.html` | MIT | `integrity=` + `crossorigin` on all five, or vendor with `LICENSES/MIT.txt` |
| h5wasm | **0.10.1** | `viewer/h5_worker.js` | MIT | `integrity=` not expressible on a bare `import`; same remedy |

## 3. Redistributed by the published container image

`.github/workflows/container.yml` builds and pushes `ghcr.io/cider-ml/lucid` when a `v*` tag is
pushed (tagged `:<sha>`, `:<tag>` and `:latest`) and on a manual run (tagged `:<sha>` only).
`container/Dockerfile` builds `FROM nuisancemc/tutorial:nuint2024`, and its runtime-environment
comment records what that base carries. The image therefore redistributes, in one artefact:

| layer | licence | note |
|---|---|---|
| the whole LUCiD source tree (`COPY LUCiD /opt/LUCiD` in `container/Dockerfile`) | **all rights reserved** ([`LICENSE`](LICENSE)) | an editable install of an unlicensed tree |
| PhotonSim, pinned to `v1.0.3` (`container.yml` `PHOTONSIM_REF_DEFAULT`) | see the PhotonSim repository | separate repository, separate terms |
| ROOT, GENIE 3.04 + xsec splines, Pythia6, NUISANCE, NEUT, NuWro, LHAPDF (base image) | mixed; **several are not freely redistributable**: GENIE's splines, NEUT and NuWro in particular are collaboration-restricted | the strictest terms in the project are in this layer, which has not been inventoried |
| Geant4 11.3 data sets | Geant4 licence | pulled by the Dockerfile's `G4*DATA` setup |

`container.yml` builds with `sbom: true`, so BuildKit attaches an SBOM (an inventory of those
layers) to each pushed image. Outstanding: stop pushing `:latest` from a public repository until
the base-image terms are established.

## 4. Python dependencies

Declared in `pyproject.toml`. All are permissive and none is copyleft, so none constrains the
licence choice:

| package | licence |
|---|---|
| jax, jaxlib, flax, optax | Apache-2.0 |
| numpy, scipy, h5py, uproot | BSD-3-Clause |
| matplotlib | matplotlib licence (PSF-derived, BSD-compatible) |
| tqdm | MIT |
| plotly, kaleido (`[viz]`) | MIT |
| imageio (`[viz]`) | BSD-2-Clause |
| mkdocs-material (`[docs]`) | MIT |
| pytest, pre-commit, ruff (`[dev]`) | MIT |

CI checks this table: the `licenses` job in `.github/workflows/ci.yml` installs `.[dev,viz]`
against `tests/constraints.txt` on every pull request and push to `main`, runs `pip-licenses`,
uploads the table as an artefact, and fails if a GNU GPL / LGPL / AGPL licence appears. The
`[docs]` extra is not in that install.

A GPL/AGPL dependency that is imported (rather than run as a separate process) would have to go,
or it constrains the licence choice. `sntools` and `snewpy` run out of process
(`lucid/production/dataprod/generators/supernova.py` runs `sys.executable -m sntools.genevts`
in a separate environment), so they are not linked in and do not appear above.

## 5. Fetched data

`scripts/download_data.py` installs the `lucid-data-v1` GitHub release asset: the trained SIREN
weights, slices of PhotonSim (Geant4) output and the overlap tables, all produced by the project,
with a sha256 for every file in `data/MANIFEST.sha256`. With `--paper` it installs
`lucid-paper-results-v1`, the paper figures' inputs (`analysis/paper/RESULTS.sha256`). Every
release asset is under CC BY 4.0, stated in its `INDEX.md`; the code stays under `LICENSE`.

The fitted artefacts under `data/` (the `emitter.json` files) are first-party, so no third-party
obligation attaches.

## 6. Cited, not licensed

Physics constants taken from the literature carry a citation obligation and no licence
obligation. `config/materials/{water,wbls,ice}.json` and 12 of the 18 `*_physics_config.json`
files cite their sources (arXiv:1307.0162 Eqs. 14-17 and Table 3, Pope & Fry 1997, Caravaca 2020
arXiv:2006.00173, and others), as do `lucid/simulation/digitizer.py` and `lucid/utils.py` (the
Bellamy 1994 SPE model). No single page collects them yet: a `docs/reference/citations.md` and a
`references:` block in `CITATION.cff` are where they would belong.

## 7. Licence texts (`LICENSES/`)

One file per licence, named by its SPDX identifier. The directory holds only the texts; this file
maps each component to the one that covers it. A licence text there is **not** a statement about
LUCiD's own terms, which are in [`LICENSE`](LICENSE).

| file | covers |
|---|---|
| [`LICENSES/MIT.txt`](LICENSES/MIT.txt) | three.js and h5wasm (CDN-loaded by `viewer/`) |
| [`LICENSES/CC0-1.0.txt`](LICENSES/CC0-1.0.txt) | the matplotlib colormap stops transcribed in `viewer/colormaps.js` |

MIT requires its permission text to travel with every copy, so a vendored MIT file whose header
carries only a URL is not compliant on that header alone; `MIT.txt` plus the component's
copyright line makes it compliant. No MIT file is vendored in the tree today.
