# Third-party inventory

Every third-party file this repository **vendors**, every third-party library
it **loads at runtime**, and every third-party layer the published container
**redistributes**. One row per component: what it is, which version, where it
came from, under which licence, why it is here, and what still has to happen.

LUCiD's own terms are in [`LICENSE`](LICENSE) (currently: all rights
reserved). Licence texts are in [`LICENSES/`](LICENSES/) (§7). This file is the map
between them.

The rule it serves: no vendored third-party asset without its licence text
beside it, and every CDN `<script src>` or import-map entry carries an
`integrity=` hash. The rows below say where that does not hold yet.

---

## 1. Vendored source — in the tree, redistributed on every clone

| component | version | origin | licence | in-tree licence text? | why it is here |
|---|---|---|---|---|---|
| fiTQun / WCSimFQTuner reference C++ (12 files, 2,773 lines), its 6 patches + `apply.sh`, `fitqun_SK_WAND.parameters.dat` (SK-IV fitted constants), and the reference grids transcribed from fiTQun `Utilities` | upstream, unversioned | Super-Kamiokande collaboration software, not publicly released | **NONE — and none exists** | the reference the fiTQun tuning chain is checked against | **Removed from the tip**: moved with their history to the separate repository `LUCiD-fitqun`, with the no-licence record. **Still in LUCiD's history** under `tools/fitqun/reference/` and `lucid/production/fitqun/data/`, so this stays **BLOCKING** until a history rewrite drops them; the tip alone is cosmetic |
| `viewer/colormaps.js` | matplotlib viridis / plasma / inferno, 7 hex stops each | https://matplotlib.org | **CC0-1.0** (the viridis family was public-domain-dedicated by its authors) | [`LICENSES/CC0-1.0.txt`](LICENSES/CC0-1.0.txt) | perceptually uniform colour ramps in the WebGL viewer | **no obligation.** Recorded so it is not re-litigated: seven stops are not a substantial reproduction, and CC0 requires nothing anyway |

## 2. Loaded at runtime from a CDN — **not one carries an `integrity=` hash**

Two packages, six `<script src>` / import entries. With no subresource
integrity, a CDN compromise or a mutated published version executes in the
browser of anyone who opens the viewer, with no signal. The licences are not
the problem here; the missing hashes are.

| component | version | loaded by | licence | needs |
|---|---|---|---|---|
| three.js (`three.min.js`, `OrbitControls.js`, `LineSegmentsGeometry.js`, `LineMaterial.js`, `LineSegments2.js`) | **0.128.0** | `viewer/index.html` | MIT | `integrity=` + `crossorigin` on all five, or vendor with `LICENSES/MIT.txt` |
| h5wasm | **0.10.1** | `viewer/h5_worker.js` | MIT | `integrity=` not expressible on a bare `import`; same remedy |

## 3. Redistributed by the published container image

`.github/workflows/container.yml` pushes `ghcr.io/cider-ml/lucid:latest` and
`:<sha>` on every push to `main` or `refactor-v2` that touches `container/`,
`lucid/`, `tests/`, `pyproject.toml` or the workflow, and on a manual run; a
published release also pushes `:<release tag>`. `container/Dockerfile` builds
`FROM nuisancemc/tutorial:nuint2024`, and its runtime-env comment records what
that base carries. The image therefore redistributes, in one artefact:

| layer | licence | note |
|---|---|---|
| the whole LUCiD source tree (`COPY LUCiD /opt/LUCiD` in `container/Dockerfile`) | **all rights reserved** ([`LICENSE`](LICENSE)) | an editable install of an unlicensed tree |
| PhotonSim, pinned to `v1.0.3` (`container.yml` `PHOTONSIM_REF_DEFAULT`) | see the PhotonSim repository | separate repository, separate terms |
| ROOT, GENIE 3.04 + xsec splines, Pythia6, NUISANCE, NEUT, NuWro, LHAPDF (base image) | mixed; **several are not freely redistributable** — GENIE's splines, NEUT and NuWro in particular are collaboration-restricted | the strictest terms in the whole project are in a layer nobody inventoried |
| Geant4 11.3 data sets | Geant4 licence | pulled by the Dockerfile's `G4*DATA` setup |

`container.yml` builds with `sbom: true`, so BuildKit attaches an SBOM (an
inventory of those layers) to each pushed image. One change is still needed:
stop pushing `:latest` from a public repository until the base-image terms are
established.

## 4. Python dependencies

Declared in `pyproject.toml`. All are permissive and none is copyleft, so none
of them constrains the licence choice:

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

The table is checked by CI: the `licenses` job in `.github/workflows/ci.yml`
installs `.[dev,viz]` against `tests/constraints.txt` on every pull request,
runs `pip-licenses`, uploads the table as an artefact, and fails if a GNU
GPL / LGPL / AGPL licence appears. (The `[docs]` extra is not in that install.)

Any GPL/AGPL dependency that turns out to be **imported** (rather than run as
a separate process) would have to go, or it constrains the licence choice.
`sntools` and `snewpy` are invoked out of process
(`lucid/production/dataprod/generators/supernova.py` runs `sys.executable -m
sntools.genevts` against a separate environment), so they are not linked in
and do not appear above.

## 5. Fetched data

`scripts/download_data.py` installs the `lucid-data-v1` GitHub release asset: the
trained SIREN weights and a slice of PhotonSim (Geant4) output, both produced by
the project, with a sha256 for every file in `data/MANIFEST.sha256`. The asset
carries **no licence yet**, like the repository; recorded here because "where did
this input come from, and may we redistribute it" is the same question this file
answers for code.

The fitted artefacts under `data/` (the `emitter.json` files) are first-party,
so no third-party obligation attaches.

## 6. Cited, not licensed

Physics constants taken from the literature carry a **citation** obligation
and no licence obligation: `config/materials/{water,wbls,ice}.json` and the 18
`*_physics_config.json` files cite their sources per field (arXiv:1307.0162
Tables 3 and 16-17, Pope & Fry 1997, Caravaca 2020 arXiv:2006.00173, and
others), as do `lucid/simulation/digitizer.py` and `lucid/utils.py` (the
Bellamy 1994 SPE model). That is in good shape. The gap is that no single page
collects them: a `docs/reference/citations.md` plus a `references:` block in
`CITATION.cff` is where they would belong. Neither exists yet.

## 7. Licence texts (`LICENSES/`)

One file per licence, named by its SPDX identifier; the directory holds only the texts, and this
file maps each component to the one that covers it. A licence text there is **not** a statement
about LUCiD's own terms, which are in [`LICENSE`](LICENSE).

| file | covers |
|---|---|
| [`LICENSES/MIT.txt`](LICENSES/MIT.txt) | three.js and h5wasm (CDN-loaded by `viewer/`) |
| [`LICENSES/CC0-1.0.txt`](LICENSES/CC0-1.0.txt) | the matplotlib colormap stops transcribed in `viewer/colormaps.js` |

MIT requires its permission text to travel with every copy, so a vendored MIT file whose header
carries only a URL is not compliant on the strength of that header alone. `MIT.txt` plus the
component's copyright line is what makes it compliant. No MIT file is vendored in the tree today.
