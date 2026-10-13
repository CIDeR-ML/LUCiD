"""Configuration loading for the JUNO laser adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .time_grid import TimeGrid
PARAMETER_NAMES = ("scat_ls", "abs_ls")

POSITIVE_PARAMETER_NAMES = frozenset(("scat_ls", "abs_ls"))


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
            raise ValueError(
                "JUNO laser config must be a JSON object: "
                f"{config_path}"
            )

        required = ("geometry_config", "physics_config", "pmt_positions",
                    "materials", "source", "transport", "calibration")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "JUNO laser config missing fields: "
                + ", ".join(missing)
            )

        source = data["source"]
        source_model = source.get("model", "isotropic")
        if source_model != "isotropic":
            raise ValueError(
                f"unsupported JUNO laser source model: {source_model!r}"
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
            if lo >= hi:
                raise ValueError(
                    f"Invalid bounds for {name!r}: {bounds[name]!r}"
                )
            if name in POSITIVE_PARAMETER_NAMES and lo <= 0:
                raise ValueError(
                    f"Positive parameter {name!r} requires a positive bound"
                )

        time_window = calibration.get("time_window_ns")
        if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
            raise ValueError(
                "calibration.time_window_ns must contain [min, max]"
            )
        if float(time_window[1]) <= float(time_window[0]):
            raise ValueError("calibration.time_window_ns must be increasing")
        if float(calibration.get("time_bin_ns", 0.0)) <= 0:
            raise ValueError("calibration.time_bin_ns must be positive")
        TimeGrid(
            float(time_window[0]),
            float(time_window[1]),
            float(calibration["time_bin_ns"]),
        )
        required_stages = ("joint",)
        if tuple(calibration.get("stages", ())) != required_stages:
            raise ValueError(
                "calibration stages must be ordered as "
                + ", ".join(required_stages)
            )
        stage_steps = calibration.get("stage_steps", {})
        if any(int(stage_steps.get(name, 0)) <= 0 for name in required_stages):
            raise ValueError("each calibration stage requires positive steps")
        ensemble_fields = (
            "train_ensemble_size",
            "mc_validation_ensemble_size",
        )
        if any(
            int(calibration.get(name, 0)) <= 0 for name in ensemble_fields
        ):
            raise ValueError(
                "train and MC-validation ensembles must be positive"
            )

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

    @property
    def interpolation_sigma_ns(self) -> float:
        """Return the numerical continuous-intensity kernel width."""
        return float(self.calibration["interpolation_sigma_ns"])

    @property
    def time_grid(self) -> TimeGrid:
        """Return the single validated calibration time grid."""
        time_min_ns, time_max_ns = self.calibration["time_window_ns"]
        return TimeGrid(
            time_min_ns,
            time_max_ns,
            self.calibration["time_bin_ns"],
        )
