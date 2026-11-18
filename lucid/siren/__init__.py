"""SIREN emitters: the network, the trained-emitter record and its loader.

- :class:`SIREN`: the network;
- :func:`load_emitter` / :class:`Emitter`: a trained emitter from ``data/<material>/<particle>/<kind>/``;
- :func:`density`: the one evaluator.

Training lives in ``lucid.siren.training`` (not imported here).
"""

from .net import SIREN, SineLayer
from .emitter import Emitter, density, load_emitter

__all__ = ['SIREN', 'SineLayer', 'Emitter', 'density', 'load_emitter']
