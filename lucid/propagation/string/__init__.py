"""String-telescope propagation.

The simulator's string propagator is ``create_string_propagator`` in ``string_propagator.py``
(``DetectorGeometry.from_config`` builds it for a ``string`` detector): brute-force distance to
every string, the closest few, a snap to the nearest DOMs on each, then the ray-DOM closest
approach. ``create_fast_string_simulator`` in ``fast.py``, exported here, is the IceCube line's
brute-force variant with the whole K-loop in one ``lax.scan``; the simulator does not call it.
"""
from lucid.propagation.string.fast import create_fast_string_simulator

__all__ = ['create_fast_string_simulator']
