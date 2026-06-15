"""Unit tests for the dataprod fan-out partition round-robin
(`lucid.production.cluster_common.dataprod_fanout`)."""
from lucid.production.cluster_common.dataprod_fanout import (
    _parse_partition_spec,
    _WeightedRoundRobin,
)


class TestParsePartitionSpec:
    def test_single_partition(self):
        assert _parse_partition_spec("roma") == [("roma", 1)]

    def test_single_flavour_unweighted(self):
        # HTCondor flavour / NERSC qos — must pass through untouched.
        assert _parse_partition_spec("workday") == [("workday", 1)]
        assert _parse_partition_spec("cpu") == [("cpu", 1)]

    def test_equal_round_robin(self):
        assert _parse_partition_spec("roma,milano") == [("roma", 1), ("milano", 1)]

    def test_weighted(self):
        assert _parse_partition_spec("roma:130,milano:272") == [
            ("roma", 130), ("milano", 272)]

    def test_empty_spec(self):
        assert _parse_partition_spec("") == []
        assert _parse_partition_spec("  ") == []

    def test_whitespace_and_trailing_comma(self):
        assert _parse_partition_spec(" roma , milano ,") == [
            ("roma", 1), ("milano", 1)]

    def test_non_integer_suffix_is_part_of_name(self):
        # A ':x' that isn't a positive int stays in the name (no weight split).
        assert _parse_partition_spec("foo:bar") == [("foo:bar", 1)]
        assert _parse_partition_spec("roma:0") == [("roma:0", 1)]
        assert _parse_partition_spec("roma:-5") == [("roma:-5", 1)]


class TestWeightedRoundRobin:
    def test_single_always_same(self):
        wrr = _WeightedRoundRobin([("roma", 1)])
        picks = [wrr.next() for _ in range(5)]
        assert picks == ["roma"] * 5
        assert wrr.counts == {"roma": 5}

    def test_equal_alternates(self):
        wrr = _WeightedRoundRobin([("a", 1), ("b", 1)])
        picks = [wrr.next() for _ in range(4)]
        assert picks == ["a", "b", "a", "b"]
        assert wrr.counts == {"a": 2, "b": 2}

    def test_weighted_exact_over_one_period(self):
        # Over sum(weights) picks, a smooth WRR yields exactly the weights.
        wrr = _WeightedRoundRobin([("roma", 130), ("milano", 272)])
        for _ in range(130 + 272):
            wrr.next()
        assert wrr.counts == {"roma": 130, "milano": 272}

    def test_weighted_is_interleaved_not_blocked(self):
        # Heavier target appears first and the two are mixed, not segregated.
        wrr = _WeightedRoundRobin([("roma", 130), ("milano", 272)])
        first20 = [wrr.next() for _ in range(20)]
        assert first20[0] == "milano"          # heavier weight leads
        assert "roma" in first20[:6]            # lighter target shows up early
        assert 0 < first20.count("roma") < 20   # genuinely mixed

    def test_proportional_for_partial_wave(self):
        # 398 jobs split ~ in proportion to 130:272.
        wrr = _WeightedRoundRobin([("roma", 130), ("milano", 272)])
        for _ in range(398):
            wrr.next()
        assert wrr.counts["roma"] + wrr.counts["milano"] == 398
        # within one of the exact proportional target
        assert abs(wrr.counts["roma"] - round(398 * 130 / 402)) <= 1
