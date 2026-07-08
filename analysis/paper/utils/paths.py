"""Default output locations for the paper figures.

Everything defaults to a repo-local ``analysis/paper/output/`` tree so a fresh
clone reproduces figures without any site-specific assumptions. The S3DF data
area (large 500-event productions) is opt-in via ``LUCID_PAPER_DATA`` or the
``--data-dir`` CLI flag.
"""
import os
from pathlib import Path

# analysis/paper/  (parents[1] of utils/paths.py)
PAPER_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get('LUCID_PAPER_OUTPUT', PAPER_ROOT / 'output'))

# Large S3DF productions (opt-in). Only used when --backend s3df or explicitly pointed here.
S3DF_DATA = Path('/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking')


def data_dir(figure: str, backend: str = 'local') -> Path:
    """Where a figure's reconstructed .h5 files live."""
    root = S3DF_DATA / 'paper' if backend == 's3df' else OUTPUT_ROOT / 'data'
    d = root / figure
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure_dir() -> Path:
    d = OUTPUT_ROOT / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d
