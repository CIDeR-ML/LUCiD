"""The dataset format's constants: version, layout, dtypes and the column encodings.

A dataset is a set of batches; batch ``k`` is four HDF5 files,
``<root>/<sub>/wc_<sub>_<k:04d>.h5`` for ``sub`` in :data:`SUBDIRS`, each holding a ``config``
group and one ``event_NNN`` group per event. The full schema is
``docs/reference/dataset-schema.md``; the version below is the one this package writes.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "FORMAT_VERSION",
    "SUBDIRS",
    "TIME_DTYPE",
    "GZIP_OPTS",
    "SOURCE_TYPE_PARTICLES",
    "SOURCE_TYPE_GENIE",
    "SOURCE_TYPE_SUPERNOVA",
    "source_type_code",
    "EMISSION_PROCESS_CHERENKOV",
    "EMISSION_PROCESS_SCINTILLATION",
    "EMISSION_PROCESS_DARK",
]

FORMAT_VERSION = 6
SUBDIRS = ('sensor', 'hits', 'step', 'labl')
GZIP_OPTS = dict(compression='gzip', compression_opts=4)
# All absolute times are stored float64: a supernova burst spans seconds
# (~1e9-1e10 ns) while the light timing is sub-ns, which float32 cannot hold
# jointly. The extra width compresses away (the low mantissa bits are ~zero).
TIME_DTYPE = np.float64

# per_interaction/source_type encoding
SOURCE_TYPE_PARTICLES = 0
SOURCE_TYPE_GENIE     = 1
SOURCE_TYPE_SUPERNOVA = 2

# emission_process column encoding on hits.h5 + step.h5/sensor_hits/.
# Per-row tag identifying which physical process produced the contribution.
# Cherenkov-only datasets emit the column with all-zeros (Cherenkov), so
# downstream consumers can always group/filter by it without a presence check.
EMISSION_PROCESS_CHERENKOV     = 0
EMISSION_PROCESS_SCINTILLATION = 1
EMISSION_PROCESS_DARK          = 2   # electronic dark-noise contribution (particle_idx = -1)


def source_type_code(primary_source):
    """Map a config's ``primary_source`` string to the ``per_interaction/source_type`` code."""
    if primary_source == 'genie':
        return SOURCE_TYPE_GENIE
    if primary_source == 'supernova':
        return SOURCE_TYPE_SUPERNOVA
    return SOURCE_TYPE_PARTICLES
