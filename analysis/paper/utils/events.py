"""Shared event preparation for the paper figures.

Thin wrappers so the paper figures load photons and build the data-mode photon dict through
the same library code as the tracking pipeline (the reader returns origins in metres; the photon
dict comes from lucid.sources.data_event.pad_photon_data). Display poses use the library's
placement rule, a rotation about the recorded primary vertex (lucid.sources.placement).
"""


def load_event(root, entry):
    """Load a PhotonSim event via the pipeline's reader (origins in metres, as the data simulator
    takes them). Returns ``(raw, energy_MeV, primary_vertex_m, primary_direction)``, the
    primary as Geant4 recorded it."""
    from lucid.sources.root_reader import read_photon_data_from_photonsim
    from lucid.sources.root_reader import read_primary
    raw = read_photon_data_from_photonsim(str(root), entry)
    v0, d0 = read_primary(str(root), entry)
    return raw, float(raw['energy']), v0, d0


def pad_event(raw, n_photons, *, primary_vertex, rotation, vertex):
    """Data-mode photon dict (:func:`lucid.sources.data_event.pad_photon_data`) of the event
    placed by the library rule: ``rotation`` about the recorded primary vertex, which moves to
    ``vertex`` (metres)."""
    from lucid.sources.data_event import pad_photon_data, place_photons
    placed = place_photons(raw, primary_vertex=primary_vertex, rotation=rotation, vertex=vertex)
    pd, _ = pad_photon_data(placed, n_photons)
    return pd


def build_track(vertex, direction, energy_MeV, t0=0.0):
    """(vec9, track) for the given vertex/direction/energy."""
    import jax.numpy as jnp
    from lucid.fitting import track_from_vec9, vec9_from_track
    th9 = vec9_from_track(energy_MeV, vertex, direction, t0)
    return th9, track_from_vec9(jnp.asarray(th9))
