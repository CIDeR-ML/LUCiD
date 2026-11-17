"""The categorizer (``lucid.dataset.labels.categorize_event``) on hand-built track lists.

Each test exercises a branch of the decision tree that picks which tracks become particles
(primary, decay electron from mu decay or capture, gamma from pi0, secondary pion), the
Cherenkov-threshold gates, and the category-parent walk.
"""
from __future__ import annotations

import numpy as np

from lucid.dataset.labels import (
    CAT_PRIMARY, CAT_DECAY_ELECTRON, CAT_SECONDARY_PION, CAT_GAMMA, CAT_NONE,
    TrackEntry, categorize_event,
)


# ---- Unit-test mini-events --------------------------------------------------

def _entry(track_id, parent_id, pdg, ke_mev=500.0, creator_process="Primary"):
    return TrackEntry(
        track_id=track_id, parent_id=parent_id, pdg=pdg,
        ke_mev=ke_mev, creator_process=creator_process)


def test_categorizer_unit_decision_tree():
    # Track 1: primary mu-. Track 2: decay e- from mu- via "Decay" process
    # (KE=10 MeV, above 1 MeV threshold). Track 3: pi0 (uncategorized).
    # Track 4: gamma from pi0 decay (kGamma). Track 5: pi+ from inelastic
    # with KE giving p > 195 MeV (kSecondaryPion).
    rows = [
        _entry(1, 0, 13, ke_mev=2000.0, creator_process="Primary"),     # mu- primary
        _entry(2, 1, 11, ke_mev=10.0, creator_process="Decay"),         # decay e-
        _entry(3, 1, 111, ke_mev=300.0, creator_process="Primary"),     # pi0 (intermediate; not categorized)
        _entry(4, 3, 22, ke_mev=150.0, creator_process="Decay"),        # gamma from pi0
        _entry(5, 1, 211, ke_mev=300.0, creator_process="pi+Inelastic"),  # secondary pi+, KE=300 → p~411 MeV
    ]
    res = categorize_event(rows)
    assert res.category_by_track_id == {
        1: CAT_PRIMARY,
        2: CAT_DECAY_ELECTRON,
        3: CAT_NONE,           # pi0 isn't in the taxonomy
        4: CAT_GAMMA,
        5: CAT_SECONDARY_PION,
    }


def test_secondary_pion_below_cherenkov_threshold_is_not_categorized():
    # KE = 50 MeV gives p ~ sqrt((50+139.57)^2 - 139.57^2) ~ 128 MeV < 195.
    rows = [
        _entry(1, 0, 211, ke_mev=2000.0, creator_process="Primary"),
        _entry(2, 1, 211, ke_mev=50.0, creator_process="pi+Inelastic"),
    ]
    res = categorize_event(rows)
    assert res.category_by_track_id[1] == CAT_PRIMARY
    assert res.category_by_track_id[2] == CAT_NONE


def test_decay_electron_below_ke_threshold_is_not_categorized():
    rows = [
        _entry(1, 0, 13, ke_mev=2000.0, creator_process="Primary"),
        _entry(2, 1, 11, ke_mev=0.5, creator_process="Decay"),  # < 1 MeV
    ]
    res = categorize_event(rows)
    assert res.category_by_track_id[2] == CAT_NONE


def test_decay_electron_from_mu_capture_at_rest():
    rows = [
        _entry(1, 0, 13, ke_mev=200.0, creator_process="Primary"),
        _entry(2, 1, 11, ke_mev=20.0, creator_process="muMinusCaptureAtRest"),
    ]
    res = categorize_event(rows)
    assert res.category_by_track_id[2] == CAT_DECAY_ELECTRON


def test_secondary_pion_from_categorized_pion_inherits_parent():
    # pi+ primary, then a deflection-spawned pi+ with creator_process containing
    # "Deflection" (or the categorized-pion-parent fallback).
    rows = [
        _entry(1, 0, 211, ke_mev=2000.0, creator_process="Primary"),
        _entry(2, 1, 211, ke_mev=300.0, creator_process="Deflection_Elastic"),
    ]
    res = categorize_event(rows)
    assert res.category_by_track_id[2] == CAT_SECONDARY_PION


def test_category_parent_walk_through_intermediate():
    """Primary pi+ (1) -> uncategorised proton (2) -> secondary pi+ (3).

    The compressed genealogy of 3 skips the proton: (1, 3). The extended
    genealogy follows the meaningful-track chain and keeps it: (1, 2, 3).
    The proton's photons belong to the primary's particle.
    """
    rows = [
        _entry(1, 0, 211, ke_mev=2000.0, creator_process="Primary"),
        _entry(2, 1, 2212, ke_mev=500.0, creator_process="pi+Inelastic"),  # intermediate proton
        _entry(3, 2, 211, ke_mev=300.0, creator_process="protonInelastic"),  # secondary pi+
    ]
    res = categorize_event(rows, meaningful_track_parent_pdg={
        1: (0, 211), 2: (1, 2212), 3: (2, 211)})
    assert res.category_by_track_id == {1: CAT_PRIMARY, 2: CAT_NONE, 3: CAT_SECONDARY_PION}
    assert res.genealogies == [(1,), (1, 3)]
    assert res.ext_genealogies == [(1,), (1, 2, 3)]
    assert res.particle_idx_by_meaningful_track == {1: 0, 2: 0, 3: 1}


M_PI = 139.57039   # charged-pion mass, MeV (PDG 2024)


def _ke_for_momentum(p):
    return float(np.sqrt(p * p + M_PI * M_PI) - M_PI)


def test_cherenkov_gates_at_their_edges():
    """Secondary pion: |p| >= 195 MeV/c (the pi Cherenkov threshold in water);
    decay electron: KE > 1 MeV. Kinetic energies computed from p by
    KE = sqrt(p^2 + m^2) - m."""
    def cat(rows, tid):
        return categorize_event(rows).category_by_track_id[tid]
    for p, want in ((194.9, CAT_NONE), (195.1, CAT_SECONDARY_PION)):
        rows = [_entry(1, 0, 211, ke_mev=2000.0),
                _entry(2, 1, 211, ke_mev=_ke_for_momentum(p), creator_process="pi+Inelastic")]
        assert cat(rows, 2) == want, p
    for ke, want in ((1.0, CAT_NONE), (1.001, CAT_DECAY_ELECTRON)):
        rows = [_entry(1, 0, 13, ke_mev=200.0),
                _entry(2, 1, 11, ke_mev=ke, creator_process="Decay")]
        assert cat(rows, 2) == want, ke


def test_primary_only_event_collapses_to_one_particle():
    rows = [
        _entry(1, 0, 13, ke_mev=500.0, creator_process="Primary"),
    ]
    res = categorize_event(rows, meaningful_track_parent_pdg={1: (0, 13)})
    assert res.genealogies == [(1,)]
    assert res.ext_genealogies == [(1,)]
