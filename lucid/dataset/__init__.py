"""The dataset format: the four-file HDF5 batches production writes and every reader reads.

``schema`` (version, layout, encodings), ``writer`` (``write_batch``, ``dataset_meta``),
``reader`` (per-event readers, ``iter_events``), ``layout`` (``batch_paths``, ``list_batches``,
``check_batch``) and ``labels`` (categorization, segment grouping, ownership, containment).
See ``docs/reference/dataset-schema.md``.
"""
