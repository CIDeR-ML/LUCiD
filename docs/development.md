# Development

**LUCiD is not accepting outside contributions yet.** The repository is public but carries no
open licence (see [`LICENSE`](https://github.com/CIDeR-ML/LUCiD/blob/main/LICENSE)), so there is
nothing for a contributor's grant to point at. Until that changes, please open an issue rather
than a pull request. Commits from project members carry a
[Developer Certificate of Origin](https://developercertificate.org/) sign-off (`git commit -s`).

## Setup

```bash
pip install -e '.[dev]' -c tests/constraints.txt   # core + pytest, pre-commit and ruff
python scripts/download_data.py                    # SIREN weights + GEANT4 events (examples, data tests)
pre-commit install                                 # run the hooks on every commit
```

`tests/constraints.txt` pins the toolchain CI runs, so a local run and CI agree.

## Tests

```bash
pytest tests/ -q          # the fast lane (the slow marker is deselected by default)
pytest tests/ -q --slow   # everything
```

The markers (`slow`, `golden`, `gpu`, `requires_data`, `requires_emitter`), the layout by
subject, the one stored reference (`tests/physics/test_forward_golden.py`, re-captured with
`LUCID_CAPTURE_GOLDEN=1` in a commit of its own that says why) and the data and GPU switches are
in [`tests/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/tests/README.md).

## The gate

A change merges when the suite stays green, the paper figure scripts still re-plot the output
already there (`--plot-results`), the production entry points still run at their smallest size,
and a reviewer has read the diff.
`bash tools/gate.sh` runs the three runnable checks in one command (exit 0 is green; 1, 2 and 3
name the first that failed, and every check always runs). A change that moves a number on
purpose says what moved and by how much in its own commit.

## Lint and hooks

`pre-commit run --all-files` runs ruff (configured in `pyproject.toml`), `nbstripout`,
`shellcheck --severity=warning` and the shell-header hook. The shell-header hook
(`.github/scripts/shell_header_check.py`) holds every shell script to two rules: `set -euo
pipefail` in the first 30 lines, and no personal or site-absolute paths (roots come from
`user_paths.sh`). CI's `lint` job is advisory until the existing findings are cleared.

## Docs

```bash
pip install -e '.[docs]' -c tests/constraints.txt
python .github/scripts/check_doc_links.py   # links, snippet includes and backtick repo paths resolve
mkdocs build --strict
```

- One source per fact: a statement lives on one page and the others link to it.
- Code on a page is included from the file it runs in (a `pymdownx.snippets` include of an
  `examples/` script or a `# --8<-- [start:name]` section of one), and CI's `examples` job runs
  that file.
- A new page goes into the `nav` of `mkdocs.yml` in the same change.
- Notebooks are committed with their outputs stripped; see
  [`notebooks/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/notebooks/README.md).
- Third-party files and CDN imports are inventoried in
  [`THIRD_PARTY.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/THIRD_PARTY.md).
