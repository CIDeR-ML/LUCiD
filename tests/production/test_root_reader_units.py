"""The one mm-to-m conversion: ``root_reader`` turns PhotonSim's float32 millimetres into float32 metres.

Every photon position in the library is metres from here on (the data-mode simulator takes them as
given). The oracle is the correctly rounded quotient: PhotonSim stores ``PhotonPos*`` as float32 mm,
so the expected value is ``float32(float64(x_mm) / 1000)``. IEEE division of a float32 by 1000 (exact
in float32) is correctly rounded too, so the reader must agree bit for bit; a cm or mm slip, or an
extra rounding through another unit, fails it.
"""
import numpy as np

from lucid.sources.root_reader import _read_photons_for_event


class _Branch:
    def __init__(self, a):
        self.a = a

    def array(self, library):
        return self.a


class _RawTree:
    """OpticalPhotonsRaw with two chunks of event 0 stored out of order, and one chunk of event 1."""

    def __init__(self, chunks):
        self.chunks = chunks          # list of (event_id, chunk_start, {branch: array})

    def __getitem__(self, name):
        key = {'EventID': 0, 'ChunkStartID': 1}[name]
        return _Branch(np.array([c[key] for c in self.chunks]))

    def arrays(self, names, entry_start, entry_stop, library):
        rows = self.chunks[entry_start:entry_stop]
        return {n: np.array([r[2][n] for r in rows], dtype=object) for n in names}


def _chunk(rng, n):
    pos = rng.uniform(-20_000, 20_000, (3, n)).astype(np.float32)        # mm, a 40 m detector
    d = {f'PhotonPos{c}': pos[i] for i, c in enumerate('XYZ')}
    d.update({f'PhotonDir{c}': np.zeros(n, np.float32) for c in 'XYZ'})
    d.update(PhotonTime=np.zeros(n, np.float32), PhotonWavelength=np.full(n, 400, np.float32))
    return d


def test_positions_are_the_correctly_rounded_metres_in_chunk_order():
    rng = np.random.default_rng(3)
    late, early, other = _chunk(rng, 7), _chunk(rng, 5), _chunk(rng, 4)
    tree = _RawTree([(0, 5, late), (0, 0, early), (1, 0, other)])

    pos, _, _, _ = _read_photons_for_event(tree, 0)

    mm = np.stack([np.concatenate([early[f'PhotonPos{c}'], late[f'PhotonPos{c}']]) for c in 'XYZ'], 1)
    assert pos.dtype == np.float32 and pos.shape == (12, 3)
    np.testing.assert_array_equal(pos, (mm.astype(np.float64) / 1000.0).astype(np.float32))
