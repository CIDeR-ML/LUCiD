"""Repository locations for the tests, so no test computes the repo root from its own depth.

``REPO`` is the checkout (``tests/__init__.py`` makes pytest import its ``lucid/``); ``CONFIG``
and ``DATA`` are its ``config/`` and gitignored ``data/``.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'config'
DATA = REPO / 'data'

# The 1 GeV muon PhotonSim events, under data/. The lucid-data release asset
# (scripts/download_data.py) ships events 0-4 of the 100-event file, cut with TTree::CopyTree and
# every branch copied verbatim, so those five are the same in both. A test that reads only them
# takes whichever is present.
MUON_1GEV_EVENTS = ('water/muon/1000MeV_5events.root', 'water/muon/1000MeV_100events.root')


def data_file(options):
    """The first of ``options`` (paths under ``data/``, or one path) that is present.

    Pair it with ``@pytest.mark.requires_data(options)``, which skips the test when none is.
    """
    options = options if isinstance(options, tuple) else (options,)
    return next(DATA / p for p in options if (DATA / p).is_file())
