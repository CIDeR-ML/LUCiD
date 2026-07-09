"""Console-script entry point `lucid-build-dedx-table`.

Real implementation lives in `build_tables.py`. See
`docs/guides/production/siren-training-inputs.md` for the pipeline.
"""

from .build_tables import main_dedx as main

if __name__ == "__main__":
    raise SystemExit(main())
