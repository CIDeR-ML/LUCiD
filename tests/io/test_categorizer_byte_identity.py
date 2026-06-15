"""Stage 4 byte-identity gate: the Python categorizer reproduces
PhotonSim's C++ ``TrackInfo_Category`` / ``Particle_GenealogyData`` /
``Particle_ExtGenealogyData`` exactly.

Two checks:

1. ``test_categorizer_unit_decision_tree``: hand-rolled mini-events
   exercising each branch of the decision tree (Primary, DecayElectron
   from μ-decay, Gamma from π0, SecondaryPion from inelastic / from a
   categorized-pion parent, plus the Cherenkov-threshold gate and the
   category-parent walk).

2. ``test_categorizer_matches_legacy_roots`` (skip-if-no-data): on a
   PhotonSim ROOT file (env var ``LUCID_PYTHON_CATEGORIZER_ROOT``)
   produced *after* the all-tracks dump landed, asserts every event's
   ``category`` array equals ``TrackInfo_Category``, every particle's
   genealogy equals ``Particle_GenealogyData`` (in order), and every
   ``Particle_ExtGenealogyData`` matches.

   Drive end-to-end via:

       LUCID_PYTHON_CATEGORIZER_ROOT=/tmp/photonsim/out.root \\
           pytest tests/test_python_categorizer_byte_identity.py

Sub-ids are NOT compared — they tie within a multi-secondary decay
where Geant4's stack (LIFO) order differs from track-id order; the
sub-id is downstream metadata only and not consumed by inst.h5 /
labl.h5.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from lucid.sources.particle_categorization import (
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
    # Primary pi+ (1) → intermediate hadron (2, NOT categorized) →
    # secondary pi+ (3, kSecondaryPion). Walk should set 3's
    # parent_id to 1 (the categorized ancestor), not 2.
    rows = [
        _entry(1, 0, 211, ke_mev=2000.0, creator_process="Primary"),
        _entry(2, 1, 2212, ke_mev=500.0, creator_process="pi+Inelastic"),  # intermediate proton
        _entry(3, 2, 211, ke_mev=300.0, creator_process="protonInelastic"),  # secondary pi+
    ]
    # Optionally provide MTrack-equivalent map (covers track 2 even though
    # it's not in TrackInfo_*). Not needed for this all-rows-in-by_id case.
    res = categorize_event(rows)
    assert res.category_by_track_id[3] == CAT_SECONDARY_PION
    # genealogy of track 3 should include only {1, 3}, not 2
    # (mt_parent map needs track 3 in it for the walker to reach it; here
    # we have no mt_parent so use a synthetic one):
    res2 = categorize_event(rows, meaningful_track_parent_pdg={3: (1, 211)})
    # After categorization, track 3's parent_id was overridden to 1.
    assert (1, 3) in res2.genealogies


def test_primary_only_event_collapses_to_one_particle():
    rows = [
        _entry(1, 0, 13, ke_mev=500.0, creator_process="Primary"),
    ]
    res = categorize_event(rows, meaningful_track_parent_pdg={1: (0, 13)})
    assert res.genealogies == [(1,)]
    assert res.ext_genealogies == [(1,)]


# ---- Byte-identity vs legacy ROOT ------------------------------------------

_ROOT_ENV = 'LUCID_PYTHON_CATEGORIZER_ROOT'


@pytest.mark.skipif(
    _ROOT_ENV not in os.environ,
    reason=f'set {_ROOT_ENV} to a PhotonSim ROOT file produced after the '
           'all-tracks-dump commit to run',
)
def test_categorizer_matches_legacy_roots():
    """Every event in the ROOT file matches the legacy C++ output."""
    import uproot

    root_path = os.environ[_ROOT_ENV]
    f = uproot.open(root_path)
    t = f['OpticalPhotons']
    arr = t.arrays([
        'TrackInfo_TrackID', 'TrackInfo_PDG', 'TrackInfo_ParentTrackID',
        'TrackInfo_Energy', 'TrackInfo_CreatorProcess', 'TrackInfo_Category',
        'MTrack_TrackID', 'MTrack_ParentID', 'MTrack_PDG', 'MTrack_NCherenkov',
        'Particle_GenealogySize', 'Particle_GenealogyData',
        'Particle_ExtGenealogySize', 'Particle_ExtGenealogyData',
    ], library='np')

    n_events = t.num_entries
    assert n_events > 0, f'no events in {root_path}'

    failures: list[str] = []
    for ev in range(n_events):
        tid = np.asarray(arr['TrackInfo_TrackID'][ev], dtype=np.int64)
        pdg = np.asarray(arr['TrackInfo_PDG'][ev], dtype=np.int64)
        par = np.asarray(arr['TrackInfo_ParentTrackID'][ev], dtype=np.int64)
        ke = np.asarray(arr['TrackInfo_Energy'][ev], dtype=np.float64)
        cp_arr = arr['TrackInfo_CreatorProcess'][ev]
        cat_root = np.asarray(arr['TrackInfo_Category'][ev], dtype=np.int64)
        rows = [
            TrackEntry(track_id=int(tid[i]), parent_id=int(par[i]),
                       pdg=int(pdg[i]), ke_mev=float(ke[i]),
                       creator_process=str(cp_arr[i]))
            for i in range(len(tid))
        ]
        mt_id = np.asarray(arr['MTrack_TrackID'][ev], dtype=np.int64)
        mt_pa = np.asarray(arr['MTrack_ParentID'][ev], dtype=np.int64)
        mt_pdg = np.asarray(arr['MTrack_PDG'][ev], dtype=np.int64)
        mt_nch = np.asarray(arr['MTrack_NCherenkov'][ev], dtype=np.int64)
        mt_parent = {int(mt_id[i]): (int(mt_pa[i]), int(mt_pdg[i]))
                     for i in range(len(mt_id))}
        mt_cher = {int(mt_id[i]): int(mt_nch[i]) for i in range(len(mt_id))}

        res = categorize_event(rows, meaningful_track_parent_pdg=mt_parent,
                               cherenkov_count_by_mt_track=mt_cher)

        cat_py = np.asarray(
            [res.category_by_track_id[int(tid[i])] for i in range(len(tid))],
            dtype=np.int64)
        if not np.array_equal(cat_py, cat_root):
            n_diff = int((cat_py != cat_root).sum())
            failures.append(
                f'event {ev}: TrackInfo_Category mismatch on {n_diff} of '
                f'{len(tid)} tracks')
            continue

        # Genealogies — order must match (lexicographic / std::map iteration).
        sizes = np.asarray(arr['Particle_GenealogySize'][ev], dtype=np.int64)
        data = np.asarray(arr['Particle_GenealogyData'][ev], dtype=np.int64)
        off = 0
        gen_root: list[tuple[int, ...]] = []
        for s in sizes:
            gen_root.append(tuple(int(x) for x in data[off:off + s]))
            off += int(s)
        if gen_root != res.genealogies:
            failures.append(
                f'event {ev}: genealogy mismatch '
                f'(ROOT n={len(gen_root)} Py n={len(res.genealogies)})')
            continue

        sizes_e = np.asarray(arr['Particle_ExtGenealogySize'][ev], dtype=np.int64)
        data_e = np.asarray(arr['Particle_ExtGenealogyData'][ev], dtype=np.int64)
        off = 0
        ext_root: list[tuple[int, ...]] = []
        for s in sizes_e:
            ext_root.append(tuple(int(x) for x in data_e[off:off + s]))
            off += int(s)
        if ext_root != res.ext_genealogies:
            failures.append(f'event {ev}: ext-genealogy mismatch')

    assert not failures, '\n'.join(failures)
