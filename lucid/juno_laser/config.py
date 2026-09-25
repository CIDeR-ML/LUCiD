"""Configuration loading for the JUNO laser adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


PARAMETER_NAMES = (
    "n_ls",
    "scat_ls",
    "abs_ls",
    "n_water",
    "scat_water",
    "abs_water",
)


@dataclass(frozen=True)
class JunoLaserConfig:
    """Validated JUNO laser configuration with resolved repository paths."""

    path: Path
    root: Path
    data: Mapping[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "JunoLaserConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open() as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"JUNO laser config must be a JSON object: {config_path}")

        required = ("geometry_config", "physics_config", "pmt_positions",
                    "materials", "source", "transport", "calibration")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"JUNO laser config missing fields: {', '.join(missing)}")

        source = data["source"]
        source_model = source.get("model", "isotropic")
        if source_model not in {"isotropic", "juno_diffuser_ball"}:
            raise ValueError(
                f"unsupported JUNO laser source model: {source_model!r}"
            )
        if source_model == "juno_diffuser_ball":
            axis = source.get("axis")
            weights = source.get("angular_weights")
            if not isinstance(axis, list) or len(axis) != 3:
                raise ValueError("source.axis must contain three values")
            if not isinstance(weights, list) or len(weights) != 4:
                raise ValueError(
                    "source.angular_weights must contain four values"
                )
            if any(float(value) <= 0.0 for value in weights):
                raise ValueError("source.angular_weights must be positive")
            if float(source.get("radius_m", 0.0)) <= 0.0:
                raise ValueError("source.radius_m must be positive")
            if float(source.get("time_sigma_ns", -1.0)) < 0.0:
                raise ValueError(
                    "source.time_sigma_ns must be non-negative"
                )

        calibration = data["calibration"]
        if calibration.get("mode") != "per_pe":
            raise ValueError("JUNO laser calibration mode must be 'per_pe'")
        parameters = tuple(calibration.get("parameters", ()))
        if parameters != PARAMETER_NAMES:
            raise ValueError(
                "JUNO laser calibration parameters must be ordered as "
                + ", ".join(PARAMETER_NAMES))
        bounds = calibration.get("parameter_bounds", {})
        for name in PARAMETER_NAMES:
            if name not in bounds or len(bounds[name]) != 2:
                raise ValueError(
                    "Missing two-sided bounds for calibration parameter "
                    f"{name!r}"
                )
            lo, hi = (float(value) for value in bounds[name])
            if not 0 < lo < hi:
                raise ValueError(f"Invalid bounds for {name!r}: {bounds[name]!r}")

        time_window = calibration.get("time_window_ns")
        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError("calibration.time_window_ns must contain [min, max]")
        if float(time_window[1]) <= float(time_window[0]):
            raise ValueError("calibration.time_window_ns must be increasing")
        if float(calibration.get("time_bin_ns", 0.0)) <= 0:
            raise ValueError("calibration.time_bin_ns must be positive")

        return cls(path=config_path, root=config_path.parent, data=data)

    def resolve(self, relative_path: str | Path) -> Path:
        """Resolve a path relative to the directory containing this config."""
        path = Path(relative_path)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def geometry_path(self) -> Path:
        return self.resolve(self.data["geometry_config"])

    @property
    def physics_path(self) -> Path:
        return self.resolve(self.data["physics_config"])

    @property
    def pmt_positions_path(self) -> Path:
        return self.resolve(self.data["pmt_positions"])

    @property
    def material_paths(self) -> dict[str, Path]:
        return {
            name: self.resolve(path)
            for name, path in self.data["materials"].items()
        }

    @property
    def wavelength_nm(self) -> float:
        return float(self.data.get("wavelength_nm", 532.0))

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            name: tuple(
                float(value)
                for value in self.data["calibration"][
                    "parameter_bounds"
                ][name]
            )
            for name in PARAMETER_NAMES
        }

    @property
    def calibration(self) -> Mapping[str, Any]:
        return self.data["calibration"]
