"""Shared pytest fixtures for tests/io/.

Wraps the builder functions in ``_event_fixture`` as pytest fixtures so
test modules can request them by name instead of importing the helpers
directly.

Two flavours are provided for each builder:

* **Session-scoped default fixture** (``synthetic_event``,
  ``synthetic_pileup_event``) -- calls the builder with its default
  arguments once per session. Suitable for the majority of tests that
  only need a representative event_dict.

* **Factory fixture** (``build_synthetic_event_factory``,
  ``build_synthetic_pileup_event_factory``) -- returns the builder
  function itself so the caller can pass custom arguments
  (e.g. ``t0``, ``n_sensors``).
"""

import pytest

from tests.io._event_fixture import (
    build_synthetic_event,
    build_synthetic_pileup_event,
)


# ---------------------------------------------------------------------------
# Factory fixtures -- return the builder callable for custom arguments
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def build_synthetic_event_factory():
    """Return the ``build_synthetic_event`` builder function."""
    return build_synthetic_event


@pytest.fixture(scope="session")
def build_synthetic_pileup_event_factory():
    """Return the ``build_synthetic_pileup_event`` builder function."""
    return build_synthetic_pileup_event


# ---------------------------------------------------------------------------
# Default-argument fixtures -- session-scoped, built once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_event():
    """A (config_meta, event_dict, sensor_positions) tuple built with defaults.

    Uses ``build_synthetic_event()`` with default arguments:
    ``source_event_idx=0, t0=7.5, n_sensors=20``.
    """
    return build_synthetic_event()


@pytest.fixture(scope="session")
def synthetic_pileup_event():
    """A (config_meta, event_dict, sensor_positions) tuple for a 2-vertex pile-up.

    Uses ``build_synthetic_pileup_event()`` with default arguments:
    ``source_event_idx=0, n_sensors=20, t0_a=-17.0, t0_b=123.4``.
    """
    return build_synthetic_pileup_event()
