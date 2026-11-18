"""Fixtures shared by the tests in tests/production."""

# Defined beside the event-generation tests that introduced them; re-exported here so
# other modules (test_derived_views.py) get them by fixture name, without an import
# that each test argument would shadow.
from tests.production.test_event_generation import detector, fake_root  # noqa: F401
