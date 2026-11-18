"""A PhotonSim event made into detector data: placed photons, the photon buffer, and the observed
``(charge, time)``.

The ``source='photons'`` simulator transports a recorded photon list instead of sampling one from a
track, so its track argument is never read. :func:`simulate_data_event` is the one call that
turns an event's photons into what a detector records; every reconstruction client goes through
it.
"""
import jax
import jax.numpy as jnp
import numpy as np


# What a padding slot holds: the simulator masks every slot past the true count N, so it carries
# no light; its direction is +z so it stays a unit vector, and it belongs to no segment.
_PAD_FILL = {'photon_directions': (0.0, 0.0, 1.0), 'photon_segment_index': -1}


def pad_photons(photons, nbuf):
    """Each per-photon array of ``photons`` (a dict) zero-filled to ``nbuf`` rows, dtype kept;
    directions are filled with +z and segment indices with -1."""
    padded = {}
    for name, a in photons.items():
        a = np.asarray(a)
        out = np.empty((nbuf,) + a.shape[1:], a.dtype)
        out[:] = _PAD_FILL.get(name, 0)
        out[:len(a)] = a
        padded[name] = out
    return padded


def pad_photon_data(raw, nbuf):
    """Pad a PhotonSim event (from :func:`read_photon_data_from_photonsim`) to a fixed ``nbuf``
    photon buffer for the data-mode simulator.

    The slots past the event's ``N`` photons are zero-filled (:func:`pad_photons`, as production
    pads its buckets); the simulator masks them, so they contribute nothing. Returns the
    ``photon_data`` dict the ``source='photons'`` simulator consumes and ``N``.
    """
    n = int(np.asarray(raw['photon_origins']).shape[0])
    if n > nbuf:
        raise ValueError(f'the event has {n} photons, more than the buffer nbuf={nbuf}; raise nbuf')
    names = ('photon_origins', 'photon_directions', 'photon_times') + (
        ('wavelengths',) if 'wavelengths' in raw else ())
    pd = {k: jnp.asarray(v) for k, v in pad_photons({k: raw[k] for k in names}, nbuf).items()}
    pd['N'] = jnp.asarray(n)
    return pd, n


def place_photons(photons, *, primary_vertex, rotation, vertex):
    """The event's photons after the placement ``x -> R (x - primary_vertex) + vertex``.

    ``photons`` as :func:`lucid.sources.root_reader.read_photon_data_from_photonsim` returns them;
    ``primary_vertex`` and ``vertex`` as :func:`lucid.sources.root_reader.read_primary` and
    :func:`lucid.sources.placement.draw_placement` give them. All in metres, float64 out.
    """
    from lucid.sources.placement import rigid_motion
    origins, directions = rigid_motion(
        np.asarray(photons['photon_origins'], np.float64),
        np.asarray(photons['photon_directions'], np.float64),
        pivot=np.asarray(primary_vertex, np.float64), rotation=np.asarray(rotation, np.float64),
        vertex=np.asarray(vertex, np.float64))
    return {**photons, 'photon_origins': origins, 'photon_directions': directions}


def place_event(root, event, photons, *, region, event_seed, accept, max_tries):
    """Event ``event`` of the PhotonSim file ``root`` placed by the library convention
    (:mod:`lucid.sources.placement`): a uniform rotation about the recorded primary vertex, the
    vertex uniform in ``region``, redrawn until ``accept(vertex, direction)`` holds, at most
    ``max_tries`` draws.

    ``photons`` is the event as :func:`lucid.sources.root_reader.read_photon_data_from_photonsim`
    returns it. The random streams are the event's, ``derive_event_keys(event_seed, 0, event)``
    (:func:`lucid.sources.seed_utils.derive_event_keys`). Returns ``(placed photons, vertex,
    direction, keys)``; ``keys`` carries the event's other streams (``sim_key``, ``seed_key``,
    ``t0_seed``, ...).
    """
    from lucid.sources.placement import draw_placement
    from lucid.sources.root_reader import read_primary
    from lucid.sources.seed_utils import derive_event_keys
    primary_vertex, primary_direction = read_primary(root, event)
    keys = derive_event_keys(event_seed, 0, event)
    rotation, vertex, direction = draw_placement(
        primary_direction, region=region, rotation_rng=np.random.default_rng(keys['rot_seed']),
        vertex_rng=np.random.default_rng(keys['vertex_seed']), accept=accept, max_tries=max_tries)
    placed = place_photons(photons, primary_vertex=primary_vertex, rotation=rotation, vertex=vertex)
    return placed, vertex, direction, keys


def simulate_data_event(data_sim, photons, key, *, nbuf, t0):
    """Observed per-sensor ``(charge, time)`` of one event, as numpy arrays.

    ``data_sim`` is a ``source='photons'`` simulator built for a buffer of ``nbuf`` photons;
    ``photons`` is the event as the reader returns it (origins in metres, times from the primary's
    start). The event starts at ``t0`` (ns): it is added to the recorded hit times, after the
    simulation, as production does. ``time`` is zero on a sensor with no charge.
    """
    pd, _ = pad_photon_data(photons, nbuf)
    c, t = jax.lax.stop_gradient(data_sim(key, pd))
    charge = np.asarray(c)
    return charge, np.where(charge > 0, np.asarray(t) + t0, 0.)
