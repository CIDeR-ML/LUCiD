"""The overlap-table cache survives parallel jobs sharing it.

Jobs that miss a cold cache all build the same table and all write it, so a reader can meet a
file mid-write. The write goes through a temporary file and a rename, and a file that cannot be
read is treated as a miss.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from lucid import overlap


@pytest.fixture
def cache_dir(tmp_path):
    overlap.set_cache_dir(str(tmp_path))
    yield tmp_path / overlap._CACHE_SUBDIR
    overlap.set_cache_dir(None)


def test_round_trip_leaves_no_temporary_file(cache_dir):
    d, f = jnp.linspace(0.0, 1.0, 5), jnp.linspace(1.0, 0.0, 5)
    overlap.save_overlap_values(0.1, 0.02, d, f)
    got = overlap.load_overlap_values(0.1, 0.02)
    assert got is not None
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(d))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(f))
    assert [p.name for p in cache_dir.iterdir()] == [overlap.get_cache_filename(0.1, 0.02)]


@pytest.mark.parametrize('content', ['{"r": 0.1, "d_val', '', '{"r": 0.1}'])
def test_a_partial_or_foreign_file_is_a_miss(cache_dir, content):
    os.makedirs(cache_dir, exist_ok=True)
    (cache_dir / overlap.get_cache_filename(0.1, 0.02)).write_text(content)
    assert overlap.load_overlap_values(0.1, 0.02) is None
