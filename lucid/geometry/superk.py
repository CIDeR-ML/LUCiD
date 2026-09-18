"""Super-Kamiokande geometry loaded from an SK ConnectionTable ROOT file."""

import warnings

import numpy as np

from .base import Detector
from .cylinder import Cylinder
from .registry import register_detector


_OPTIONAL_BRANCHES = (
    "pmtflag", "supserial", "modserial", "hutnum", "group",
    "hvcrate", "hvmodadd", "hvch", "oldhv",
    "prodyear_sk4", "prodyear_sk5",
)


@register_detector("superk")
class SuperK(Cylinder):
    """A cylindrical detector using measured Super-K PMT positions.

    This subclasses :class:`Cylinder`, so it uses the same grid,
    ray-intersection, bounds-check, normal, and propagation implementation.
    Only PMT placement differs: positions and metadata are read from an SK
    ``ConnectionTable`` TTree.

    ROOT positions are expected in centimetres and are converted to metres.
    Sensors are reordered to the standard LUCiD order (barrel, top, bottom).
    """

    CM_TO_M = 0.01
    DEFAULT_Z_BOUNDARY = 18.0
    geometry_type = "cylinder"

    def __init__(
        self,
        connection_table_path,
        radius,
        height,
        n_sensors,
        sensor_radius,
        z_boundary=None,
        tree_name="ConnectionTable",
        snap_to_wall=True,
    ):
        self.connection_table_path = str(connection_table_path)
        self.r = float(radius)
        self.H = float(height)
        self.z_boundary = (
            self.DEFAULT_Z_BOUNDARY if z_boundary is None else float(z_boundary)
        )
        self.tree_name = tree_name
        self.snap_to_wall = bool(snap_to_wall)
        self._n_cap = None
        self._n_angular = None
        self._n_height = None

        self._load_connection_table()
        actual_n = len(self._pmtx)
        if actual_n != int(n_sensors):
            warnings.warn(
                f"SuperK config declares {n_sensors} sensors, but "
                f"'{self.connection_table_path}' contains {actual_n}; "
                "using the ConnectionTable count.",
                UserWarning,
                stacklevel=2,
            )

        Detector.__init__(self, actual_n, float(sensor_radius))
        self.place_photosensors()

    def _load_connection_table(self):
        """Read required and optional ConnectionTable branches."""
        try:
            import uproot
        except ImportError as exc:
            raise ImportError(
                "SuperK geometry requires uproot. Install the project "
                "dependencies or run `pip install uproot`."
            ) from exc

        with uproot.open(self.connection_table_path) as root_file:
            tree = root_file[self.tree_name]
            required = ("cableid", "pmtx", "pmty", "pmtz")
            missing = [name for name in required if name not in tree]
            if missing:
                raise KeyError(
                    f"ConnectionTable '{self.connection_table_path}' is "
                    f"missing required branches: {missing}"
                )

            arrays = {
                name: tree[name].array(library="np") for name in required
            }
            for name in _OPTIONAL_BRANCHES:
                if name in tree:
                    arrays[name] = tree[name].array(library="np")

        n_pmts = len(arrays["cableid"])
        bad_lengths = {
            name: len(values)
            for name, values in arrays.items()
            if len(values) != n_pmts
        }
        if bad_lengths:
            raise ValueError(
                "ConnectionTable branches do not have a consistent length: "
                f"expected {n_pmts}, got {bad_lengths}"
            )

        self._cableid = np.asarray(arrays.pop("cableid"))
        self._pmtx = np.asarray(arrays.pop("pmtx"), dtype=float) * self.CM_TO_M
        self._pmty = np.asarray(arrays.pop("pmty"), dtype=float) * self.CM_TO_M
        self._pmtz = np.asarray(arrays.pop("pmtz"), dtype=float) * self.CM_TO_M
        self._connection_metadata = {
            name: np.asarray(values) for name, values in arrays.items()
        }

    def place_photosensors(self):
        """Populate Cylinder-compatible arrays using measured PMT positions."""
        top = self._pmtz > self.z_boundary
        bottom = self._pmtz < -self.z_boundary
        barrel = ~(top | bottom)
        order = np.concatenate(
            [np.flatnonzero(barrel), np.flatnonzero(top), np.flatnonzero(bottom)]
        )
        self._reorder_indices = order

        surfaces = np.empty(len(order), dtype="<U6")
        n_barrel = int(np.sum(barrel))
        n_top = int(np.sum(top))
        surfaces[:n_barrel] = "barrel"
        surfaces[n_barrel:n_barrel + n_top] = "top"
        surfaces[n_barrel + n_top:] = "bottom"
        self.surfaces = surfaces

        positions = np.column_stack((self._pmtx, self._pmty, self._pmtz))[order]
        self.raw_positions = positions.copy()
        if self.snap_to_wall:
            positions = self._snap_to_wall(positions, surfaces)

        self.all_points = positions
        self.barr_points = positions[:n_barrel]
        self.tcap_points = positions[n_barrel:n_barrel + n_top]
        self.bcap_points = positions[n_barrel + n_top:]

        self.ID_to_position = {
            i: self.all_points[i] for i in range(self.n_sensors)
        }
        cases = np.empty(self.n_sensors, dtype=int)
        cases[:n_barrel] = 0
        cases[n_barrel:n_barrel + n_top] = 1
        cases[n_barrel + n_top:] = 2
        self.ID_to_case = {
            i: int(cases[i]) for i in range(self.n_sensors)
        }

        self.pmt_id = self._cableid[order]
        self.ID_to_cableid = {
            i: int(cable_id) for i, cable_id in enumerate(self.pmt_id)
        }
        self.cableid_to_ID = {
            cable_id: idx for idx, cable_id in self.ID_to_cableid.items()
        }
        if len(self.cableid_to_ID) != self.n_sensors:
            raise ValueError("ConnectionTable cableid values must be unique")
        self.pmt_id_to_idx = dict(self.cableid_to_ID)

        for name, values in self._connection_metadata.items():
            ordered = values[order]
            setattr(self, name, ordered)
            setattr(self, f"_{name}", ordered)

        # Derive inward viewing directions from the cylindrical surface.
        directions = np.zeros_like(self.all_points)
        xy = self.all_points[:n_barrel, :2]
        xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
        directions[:n_barrel, :2] = -xy / np.where(xy_norm > 0, xy_norm, 1.0)
        directions[n_barrel:n_barrel + n_top, 2] = -1.0
        directions[n_barrel + n_top:, 2] = 1.0
        self.pmt_directions = directions

    def get_pmt_info(self, sequential_id):
        """Return position, surface, cable ID, and available metadata."""
        idx = int(sequential_id)
        if idx < 0 or idx >= self.n_sensors:
            raise IndexError(f"PMT index {idx} is outside [0, {self.n_sensors})")

        info = {
            "cable_id": int(self.pmt_id[idx]),
            "position": self.all_points[idx].copy(),
            "raw_position": self.raw_positions[idx].copy(),
            "surface": self.surfaces[idx],
        }
        for name in _OPTIONAL_BRANCHES:
            if hasattr(self, name):
                value = getattr(self, name)[idx]
                info[name] = value.item() if hasattr(value, "item") else value
        return info

