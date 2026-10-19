"""Position- and angle-dependent PMT detection efficiency from a SIREN.

The model contract follows the PMT prototype trained on TitouanGr/LUCiD's
``refactor-v2`` branch: the three physical inputs are the hit position's local
zenith and azimuth on the PMT bulb, followed by the photon incidence cosine.
The trained output is a multiplicative efficiency in ``[0, 1]``; quantum
efficiency is applied separately by :mod:`lucid.simulation.sensor_response`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from lucid.siren.core import SIREN


_PARAM_PREFIX = "param__"


def _nested_set(tree: dict, path: list[str], value: Any) -> None:
    node = tree
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def _flatten_params(tree: Mapping[str, Any], prefix: tuple[str, ...] = ()):
    for name, value in tree.items():
        path = prefix + (name,)
        if isinstance(value, Mapping):
            yield from _flatten_params(value, path)
        else:
            yield path, np.asarray(value)


def _params_only(value: Any) -> Mapping[str, Any]:
    """Accept either a Flax params tree or ``{"params": tree}``."""
    if hasattr(value, "unfreeze"):
        value = value.unfreeze()
    if not isinstance(value, Mapping):
        raise TypeError("PMT SIREN weights must be a mapping of Flax parameters")
    if set(value) == {"params"}:
        value = value["params"]
    return value


@dataclass(frozen=True)
class PmtSirenResponse:
    """Loaded PMT-efficiency model and its input normalization."""

    model: SIREN
    params: Mapping[str, Any]
    input_mean: jnp.ndarray
    input_scale: jnp.ndarray
    output_min: float = 0.0
    output_max: float = 1.0

    def efficiency(self, physical_inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate ``(local_zenith, local_azimuth, cos_incidence)`` rows."""
        normalized = (physical_inputs - self.input_mean) / self.input_scale
        output = self.model.apply({"params": self.params}, normalized)
        if isinstance(output, tuple):
            output = output[0]
        # Titouan's PMT network was trained with a Softplus output.  Keep that
        # response-specific transform here instead of changing the emission SIREN.
        output = jax.nn.softplus(jnp.asarray(output).squeeze(-1))
        return jnp.clip(output, self.output_min, self.output_max)


def save_pmt_siren_response(
    path: str | Path,
    params: Mapping[str, Any],
    *,
    input_mean,
    input_scale,
    first_omega_0: float = 30.0,
    hidden_omega_0: float = 15.0,
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> None:
    """Write a portable, pickle-free ``.npz`` PMT SIREN artifact.

    Network width and depth are inferred from the parameter shapes/names.  The
    normalization must come from the training axes; weights alone are not
    sufficient to reproduce the trained function.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("PMT SIREN artifacts must use the .npz extension")

    params = _params_only(params)
    sine_names = sorted(k for k in params if k.startswith("SineLayer_"))
    if not sine_names or "Dense_0" not in params:
        raise ValueError("Unrecognized PMT SIREN parameter tree")
    first_kernel = np.asarray(params["SineLayer_0"]["Dense_0"]["kernel"])
    hidden_features = int(first_kernel.shape[1])
    hidden_layers = len(sine_names) - 1

    arrays: dict[str, Any] = {
        "format_version": np.asarray(1, dtype=np.int32),
        "hidden_features": np.asarray(hidden_features, dtype=np.int32),
        "hidden_layers": np.asarray(hidden_layers, dtype=np.int32),
        "first_omega_0": np.asarray(first_omega_0, dtype=np.float32),
        "hidden_omega_0": np.asarray(hidden_omega_0, dtype=np.float32),
        "input_mean": np.asarray(input_mean, dtype=np.float32),
        "input_scale": np.asarray(input_scale, dtype=np.float32),
        "output_min": np.asarray(output_min, dtype=np.float32),
        "output_max": np.asarray(output_max, dtype=np.float32),
    }
    for names, value in _flatten_params(params):
        arrays[_PARAM_PREFIX + "__".join(names)] = value
    np.savez(path, **arrays)


def _load_npz(path: Path) -> PmtSirenResponse:
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "format_version", "hidden_features", "hidden_layers",
            "first_omega_0", "hidden_omega_0", "input_mean", "input_scale",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(
                f"PMT SIREN artifact {path} is missing {sorted(missing)}")
        params: dict[str, Any] = {}
        for key in artifact.files:
            if key.startswith(_PARAM_PREFIX):
                names = key[len(_PARAM_PREFIX):].split("__")
                _nested_set(params, names, jnp.asarray(artifact[key]))
        if not params:
            raise ValueError(f"PMT SIREN artifact {path} contains no parameters")
        input_mean = jnp.asarray(artifact["input_mean"], dtype=jnp.float32)
        input_scale = jnp.asarray(artifact["input_scale"], dtype=jnp.float32)
        if input_mean.shape != (3,) or input_scale.shape != (3,):
            raise ValueError("PMT SIREN input_mean and input_scale must have shape (3,)")
        if np.any(np.asarray(input_scale) == 0):
            raise ValueError("PMT SIREN input_scale entries must be nonzero")
        model = SIREN(
            hidden_features=int(artifact["hidden_features"]),
            hidden_layers=int(artifact["hidden_layers"]),
            out_features=1,
            outermost_linear=True,
            first_omega_0=float(artifact["first_omega_0"]),
            hidden_omega_0=float(artifact["hidden_omega_0"]),
            square_output=False,
        )
        output_min = float(artifact["output_min"]) if "output_min" in artifact else 0.0
        output_max = float(artifact["output_max"]) if "output_max" in artifact else 1.0
    return PmtSirenResponse(
        model=model,
        params=params,
        input_mean=input_mean,
        input_scale=input_scale,
        output_min=output_min,
        output_max=output_max,
    )


def _load_legacy_npy(path: Path) -> PmtSirenResponse:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Legacy PMT SIREN weights {path} need {metadata_path.name} with "
            "input_mean and input_scale. Titouan's branch did not commit this "
            "normalization or the trained weights, so it cannot be inferred safely.")
    with metadata_path.open() as stream:
        metadata = json.load(stream)
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    params = _params_only(raw)
    sine_names = sorted(k for k in params if k.startswith("SineLayer_"))
    model = SIREN(
        hidden_features=int(np.asarray(params["SineLayer_0"]["Dense_0"]["kernel"]).shape[1]),
        hidden_layers=len(sine_names) - 1,
        out_features=1,
        outermost_linear=True,
        first_omega_0=float(metadata.get("first_omega_0", 30.0)),
        hidden_omega_0=float(metadata.get("hidden_omega_0", 15.0)),
        square_output=False,
    )
    return PmtSirenResponse(
        model=model,
        params=jax.tree.map(jnp.asarray, params),
        input_mean=jnp.asarray(metadata["input_mean"], dtype=jnp.float32),
        input_scale=jnp.asarray(metadata["input_scale"], dtype=jnp.float32),
        output_min=float(metadata.get("output_min", 0.0)),
        output_max=float(metadata.get("output_max", 1.0)),
    )


def load_pmt_siren_response(path: str | Path) -> PmtSirenResponse:
    """Load a portable ``.npz`` artifact or Titouan's legacy ``.npy`` format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PMT SIREN model does not exist: {path}")
    if path.suffix == ".npz":
        return _load_npz(path)
    if path.suffix == ".npy":
        return _load_legacy_npy(path)
    raise ValueError("PMT SIREN model must be a .npz or .npy file")


def build_pmt_frames(sensor_centers, sensor_directions=None, sensor_cases=None):
    """Build the local ``(x, y, z)`` frames used by the trained PMT model.

    ``z`` is the inward PMT viewing direction.  Barrel PMTs use global ``+z``
    as local ``+y``; cap PMTs use global ``+x`` as local ``+x``.  This exactly
    matches the coordinate convention in the prototype branch while avoiding
    geometry calculations inside the propagation loop.
    """
    centers = jnp.asarray(sensor_centers, dtype=jnp.float32)
    if sensor_directions is None:
        if sensor_cases is None:
            raise ValueError("PMT SIREN needs sensor directions or cylinder sensor cases")
        cases = jnp.asarray(sensor_cases)
        radial = centers.at[:, 2].set(0.0)
        radial_norm = jnp.linalg.norm(radial, axis=1, keepdims=True)
        wall_z = -radial / jnp.maximum(radial_norm, 1e-10)
        top_z = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), centers.shape)
        bottom_z = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), centers.shape)
        z_axis = jnp.where((cases == 0)[:, None], wall_z,
                           jnp.where((cases == 1)[:, None], top_z, bottom_z))
    else:
        z_axis = jnp.asarray(sensor_directions, dtype=jnp.float32)
        z_axis = z_axis / jnp.maximum(jnp.linalg.norm(z_axis, axis=1, keepdims=True), 1e-10)

    is_barrel = jnp.abs(z_axis[:, 2]) < 0.5
    wall_y = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), centers.shape)
    cap_x = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), centers.shape)
    wall_x = jnp.cross(wall_y, z_axis)
    wall_x = wall_x / jnp.maximum(jnp.linalg.norm(wall_x, axis=1, keepdims=True), 1e-10)
    cap_y = jnp.cross(z_axis, cap_x)
    x_axis = jnp.where(is_barrel[:, None], wall_x, cap_x)
    y_axis = jnp.where(is_barrel[:, None], wall_y, cap_y)
    return jnp.stack((x_axis, y_axis, z_axis), axis=1)


def pmt_siren_inputs(
    hit_positions,
    photon_directions,
    sensor_indices,
    sensor_centers,
    sensor_frames,
):
    """Convert candidate ray/PMT intersections to the model's physical inputs."""
    valid = sensor_indices >= 0
    safe_indices = jnp.where(valid, sensor_indices, 0)
    centers = sensor_centers[safe_indices]
    frames = sensor_frames[safe_indices]
    displacement = hit_positions - centers
    local = jnp.einsum("...ij,...j->...i", frames, displacement)
    radius = jnp.linalg.norm(local, axis=-1)
    local_zenith = -jnp.arccos(jnp.clip(local[..., 2] / jnp.maximum(radius, 1e-10), -1.0, 1.0))
    local_azimuth = jnp.arctan2(local[..., 1], local[..., 0])
    if photon_directions.ndim == hit_positions.ndim:
        ray_dirs = jnp.broadcast_to(photon_directions, hit_positions.shape)
    else:
        ray_dirs = jnp.broadcast_to(
            photon_directions[None, :, :], hit_positions.shape)
    cos_incidence = jnp.clip(-jnp.sum(ray_dirs * frames[..., 2, :], axis=-1), 0.0, 1.0)
    physical = jnp.stack((local_zenith, local_azimuth, cos_incidence), axis=-1)
    return jnp.where(valid[..., None], physical, 0.0)
