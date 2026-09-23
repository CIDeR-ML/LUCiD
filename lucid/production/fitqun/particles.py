"""Particle constants shared by the tuning generators.

Kept free of numpy (and of anything else) so the cluster fan-out, which runs
on the submit host rather than inside the container, can generate PhotonSim
macros without a scientific-Python stack.
"""
from __future__ import annotations

# fiTQun's particle hypotheses, by PDG code.
PDG_NAMES = {11: "e-", 13: "mu-", 211: "pi+"}

# Rest masses (MeV) for the momentum -> kinetic-energy conversion.
PDG_MASSES = {11: 0.51099895, 13: 105.6583755, 211: 139.57039}
