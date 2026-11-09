# Development

LUCiD does not accept outside contributions yet: the repository is public but all rights are
reserved ([`LICENSE`](https://github.com/CIDeR-ML/LUCiD/blob/main/LICENSE)). Please open an
issue rather than a pull request. Project members sign off their commits with a
[Developer Certificate of Origin](https://developercertificate.org/) (`git commit -s`).

## Setup

```bash
pip install -e '.[dev]' -c tests/constraints.txt   # LUCiD with pytest, pre-commit, ruff
python scripts/download_data.py                    # data the examples and data tests need
pre-commit install                                 # run the hooks on every commit
```

## Tests

```bash
pytest tests/ -q          # the fast set (tests marked slow are skipped)
pytest tests/ -q --slow   # everything
```

The markers (`slow`, `golden`, `gpu`, `requires_data`, `requires_emitter`), the layout and the
switches for data and GPU tests are in
[`tests/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/tests/README.md). The one stored
reference, `tests/physics/test_forward_golden.py`, is re-captured with `LUCID_CAPTURE_GOLDEN=1`
in a commit of its own that says why.

## The gate

A change merges when the test suite passes, the paper figure scripts still re-plot their
existing output (`--plot-results`), the production entry points still run at their smallest
size, and a reviewer has read the diff. `bash tools/gate.sh` runs the first three; it exits 0
when all pass, or 1, 2 or 3 naming the first that failed. A change that moves a published number
on purpose says what moved, and by how much, in its own commit.

## Lint and hooks

`pre-commit run --all-files` runs ruff (configured in `pyproject.toml`), `nbstripout`,
`shellcheck --severity=warning` and a shell-header check: every shell script sets
`set -euo pipefail` in its first 30 lines and contains no personal or site-specific absolute
paths (those come from `user_paths.sh`). The CI lint job is advisory for now.

## Docs

```bash
pip install -e '.[docs]' -c tests/constraints.txt
python .github/scripts/check_doc_links.py   # links, snippet includes and backtick repo paths resolve
mkdocs build --strict
```

- A fact lives on one page; other pages link to it.
- Code shown on a page is included from a file that runs, an `examples/` script or a
  `# --8<-- [start:name]` section of one, and CI's `examples` job runs those files.
- A new page goes into the `nav` of `mkdocs.yml` in the same change.
- Notebooks are committed with outputs stripped
  ([`notebooks/README.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/notebooks/README.md)).
- Third-party files and CDN imports are listed in
  [`THIRD_PARTY.md`](https://github.com/CIDeR-ML/LUCiD/blob/main/THIRD_PARTY.md).
