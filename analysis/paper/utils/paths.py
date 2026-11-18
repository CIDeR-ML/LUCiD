"""Default output locations for the paper figures.

Everything defaults to a repo-local ``analysis/paper/output/`` tree (moved by
``LUCID_PAPER_OUTPUT``) so a fresh clone reproduces figures without any
site-specific assumptions. The S3DF data area (large 500-event productions) is
opt-in: ``data_dir(figure, 's3df')``, i.e. a figure's ``--backend s3df``.
"""
import os
from pathlib import Path

# analysis/paper/  (parents[1] of utils/paths.py)
PAPER_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get('LUCID_PAPER_OUTPUT', PAPER_ROOT / 'output'))

# The S3DF site: every site-specific location the paper code knows is here, and nowhere else.
# The large productions (opt-in: --backend s3df).
S3DF_DATA = Path('/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking')
# The 500-event PhotonSim inputs of the scans, <mu-|e->/<E>MeV_500events.root (utils/generate_data.py
# made them; a figure's --root-base points elsewhere).
S3DF_ROOT_FILES = '/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water'

# What the S3DF sbatch writers run: the LUCiD container, the CUDA jaxlib user site layered over
# its CPU-only one, and the binds.
S3DF_IMAGE = '/sdf/data/neutrino/cjesus/software/images/lucid.sif'
S3DF_ENV_BASE = '/sdf/data/neutrino/cjesus/python_envs/lucid'
S3DF_BINDS = '/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs'


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
