"""String-telescope propagation.

Primary entry point: ``create_fast_string_simulator`` in ``fast.py``.

The DDA/hash/match pipeline in ``propagator.py``, ``match.py``, ``dda.py``,
``kernel.py``, and ``hash.py`` is the original reference implementation,
superseded by the brute-force + lax.scan approach in ``fast.py`` (~20× faster).
"""
from lucid.propagation.string.fast import create_fast_string_simulator

__all__ = ['create_fast_string_simulator']
