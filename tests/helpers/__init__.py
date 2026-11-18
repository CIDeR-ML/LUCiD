"""Test inputs that are shared between test modules, as an importable package.

One import path per helper, spelled in full, e.g.::

    from tests.helpers.calib_toy import charge, dp_true, sources, FIELDS
    from tests.helpers.recon_analytic_model import AnalyticModel
    from tests.helpers.event_fixture import build_synthetic_event

This ``__init__`` imports nothing on purpose. A helper pulls in jax, numpy or a lucid module, and
re-exporting them here would make ``import tests.helpers`` -- which pytest performs while
collecting any sibling -- pay for all of them.

The modules carry no leading underscore: nothing here is named ``test_*``, so pytest does not
collect this package, and the underscore the old ``tests/_calib_toy.py`` needed was only ever a
collection guard.

Modules
-------
calib_toy
    A synthetic calibration problem: per-sensor charge log-linear in three optical fields.
recon_analytic_model
    An ill-conditioned analytic nine-parameter least-squares problem for the fit_track loop.
event_fixture
    Synthetic dataset events (single, pile-up, digit) for the IO round-trip tests, and
    ``write_batch``, which writes one through the four writers.
sensor_map
    Brute-force views of the propagator's forward and inverse sensor maps.
stats
    ``ks_distance``: the exact Kolmogorov-Smirnov distance to a closed-form CDF.
params
    ``lab_dp`` and ``calib_dp``: the two DetectorParams presets the optics tests share, and
    leaf get/replace/central-difference helpers for the gradient tests.
"""
