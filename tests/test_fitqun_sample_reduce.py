"""The sparse shard accumulator must agree with the dense table it replaces.

Shards are kept sparse only to survive the merge; what fiTQun eventually reads
is the densified sum. So the one property that matters is that densifying loses
nothing -- including the out-of-range rule, which drops an entry outside any
axis rather than clamping it.
"""
import numpy as np
import pytest

from lucid.production.fitqun import scattable
from lucid.production.fitqun.sample_reduce import SampleShard, SparseCounts


NB = (7, 5, 5, 4, 3, 3)
BD = ((-100.0, 100.0), (0.0, 80.0), (-90.0, 90.0),
      (-np.pi, np.pi), (-1.0, 1.0), (-np.pi, np.pi))


def _coords(rng, n):
    return [rng.uniform(lo, hi, n) for lo, hi in BD]


def test_sparse_matches_dense_bin_for_bin():
    rng = np.random.default_rng(0)
    vals = _coords(rng, 5000)

    dense = scattable.ScatTable("sidescattable", NB, BD)
    dense.fill(*vals)
    sparse = SparseCounts.from_values("sidescattable", NB, BD, vals)

    np.testing.assert_array_equal(sparse.to_dense().table, dense.table)
    assert sparse.count.sum() == dense.table.sum()


def test_out_of_range_entries_are_dropped_not_clamped():
    """An entry outside one axis must not pile up on that axis's edge bin."""
    inside = [np.array([0.0]), np.array([40.0]), np.array([0.0]),
              np.array([0.0]), np.array([0.0]), np.array([0.0])]
    outside = [v.copy() for v in inside]
    outside[1] = np.array([1e6])          # far beyond rs

    vals = [np.concatenate([a, b]) for a, b in zip(inside, outside)]
    dense = scattable.ScatTable("sidescattable", NB, BD)
    dense.fill(*vals)
    sparse = SparseCounts.from_values("sidescattable", NB, BD, vals)

    assert dense.table.sum() == 1          # only the in-range entry survived
    np.testing.assert_array_equal(sparse.to_dense().table, dense.table)


def test_sparse_addition_matches_dense_addition():
    rng = np.random.default_rng(1)
    a, b = _coords(rng, 800), _coords(rng, 900)

    da = scattable.ScatTable("topscattable", NB, BD); da.fill(*a)
    db = scattable.ScatTable("topscattable", NB, BD); db.fill(*b)
    sa = SparseCounts.from_values("topscattable", NB, BD, a)
    sb = SparseCounts.from_values("topscattable", NB, BD, b)

    np.testing.assert_array_equal((sa + sb).to_dense().table, (da + db).table)


def test_addition_refuses_mismatched_binning():
    other = (8, 5, 5, 4, 3, 3)
    a = SparseCounts("sidescattable", NB, BD)
    b = SparseCounts("sidescattable", other, BD)
    with pytest.raises(ValueError, match="different binning"):
        a + b


def test_shard_round_trip_through_npz(tmp_path):
    rng = np.random.default_rng(2)
    shard = SampleShard(
        scattered={"sidescattable": SparseCounts.from_values(
            "sidescattable", NB, BD, _coords(rng, 400))},
        direct={"sidescattable": SparseCounts.from_values(
            "sidescattable", NB[:4] + (1, 1), BD, _coords(rng, 300))},
        angular_counts={100.0: rng.random(25)},
        angular_sumw2={100.0: rng.random(25)},
        n_photons=1000, n_detected=700, n_indirect=200)

    back = SampleShard.load(shard.save(tmp_path / "shard.npz"))
    np.testing.assert_array_equal(
        back.scattered["sidescattable"].to_dense().table,
        shard.scattered["sidescattable"].to_dense().table)
    np.testing.assert_allclose(back.angular_counts[100.0], shard.angular_counts[100.0])
    assert (back.n_photons, back.n_detected, back.n_indirect) == (1000, 700, 200)

    # Summing shards is how the merge works, so it must survive the round trip.
    total = back + shard
    assert total.n_detected == 1400
    np.testing.assert_allclose(total.angular_counts[100.0],
                               2 * shard.angular_counts[100.0])
