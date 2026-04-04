"""Detector geometry registry — decorator-based dispatch replacing if/elif chains.

Usage::

    from lucid.geometry.registry import register_detector, get_detector_class

    @register_detector('cylinder')
    class Cylinder(Detector):
        ...

    cls = get_detector_class('cylinder')   # returns Cylinder
    cls = get_detector_class('Cylinder')   # also returns Cylinder (case-insensitive)
"""

_REGISTRY = {}


def register_detector(name):
    """Class decorator that registers a Detector subclass under ``name``.

    The name is stored in lowercase. Lookup via ``get_detector_class``
    is case-insensitive.
    """
    key = name.lower()

    def decorator(cls):
        if key in _REGISTRY:
            raise ValueError(
                f"Detector '{key}' already registered as {_REGISTRY[key].__name__}, "
                f"cannot re-register as {cls.__name__}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_detector_class(name):
    """Look up a registered detector class by name (case-insensitive).

    Parameters
    ----------
    name : str
        Detector type name (e.g. 'cylinder', 'Cylinder', 'CYLINDER').

    Returns
    -------
    type
        The registered Detector subclass.

    Raises
    ------
    ValueError
        If no detector is registered under this name.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown detector type '{name}'. Available: {available}")
    return _REGISTRY[key]


def list_detector_types():
    """Return sorted list of registered detector type names."""
    return sorted(_REGISTRY.keys())
