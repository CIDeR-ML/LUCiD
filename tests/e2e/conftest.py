"""Shared constants and helpers for e2e tests.

Split from the original test_e2e_wavelength.py.
"""
import os
import sys

# Ensure the LUCiD version (not diffCherenkov) is imported
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

# ---- Paths ---------------------------------------------------------------
CONFIG = os.path.join(BASE, "config")

WCTE_GEOM = os.path.join(CONFIG, "WCTE_like_geom_config.json")
WCTE_PHYS = os.path.join(CONFIG, "WCTE_like_physics_config.json")
SK_LIKE_GEOM = os.path.join(CONFIG, "SK_like_geom_config.json")
SK_LIKE_PHYS = os.path.join(CONFIG, "SK_like_physics_config.json")
SK_GEOM = os.path.join(CONFIG, "SK_geom_config.json")
SK_PHYS = os.path.join(CONFIG, "SK_physics_config.json")
JUNO_PHYS = os.path.join(CONFIG, "JUNO_physics_config.json")

OLD_ROOT = os.path.join(BASE, "data", "water", "muon",
                        "muon_gun_1050_MeV_100_events_fixed_energy.root")
NEW_ROOT = os.path.join(BASE, "..", "PhotonSim", "build",
                        "test_wavelength_5_events.root")

NPHOT = 5_000          # small for speed
K_VAL = 3              # few bounces for speed
KEY = jax.random.PRNGKey(42)

results = {}


def report(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    results[name] = (passed, detail)
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))
