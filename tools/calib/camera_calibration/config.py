"""Versioned YAML configuration/result helpers and unit validation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import ConfigurationError


FORMAT_VERSION = "1.0"
LENGTH_UNIT = "metre"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping and retain its directory for relative paths."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise ConfigurationError(f"YAML file does not exist: {source}")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read YAML {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Top-level YAML value must be a mapping: {source}")
    result = deepcopy(value)
    result["_config_file"] = str(source.resolve())
    result["_config_dir"] = str(source.resolve().parent)
    return result


def save_yaml(path: Path | str, data: dict[str, Any]) -> Path:
    """Atomically-enough write portable, versioned YAML scalars."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _plain(deepcopy(data))
    payload.setdefault("format_version", FORMAT_VERSION)
    payload.setdefault(
        "generated_at",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    try:
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigurationError(f"Cannot write YAML {destination}: {exc}") from exc
    return destination


def _plain(value: Any) -> Any:
    """Convert NumPy-like scalar/list objects to YAML-safe builtin values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def require(config: dict[str, Any], dotted_key: str) -> Any:
    """Read a required dotted key and reject absent/null values."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigurationError(f"Missing required configuration: {dotted_key}")
        value = value[part]
    if value is None:
        raise ConfigurationError(f"Configuration cannot be null: {dotted_key}")
    return value


def require_keys(config: dict[str, Any], keys: Iterable[str]) -> None:
    """Validate a collection of required keys."""
    for key in keys:
        require(config, key)


def resolve_path(config: dict[str, Any], value: Path | str) -> Path:
    """Resolve a path relative to its YAML, not the Python source tree."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config.get("_config_dir", ".")) / path).resolve()


def output_path(config: dict[str, Any], default: str) -> Path:
    """Resolve ``output.directory``."""
    return resolve_path(config, config.get("output", {}).get("directory", default))


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove loader metadata before embedding a config in output."""
    return {
        key: deepcopy(value)
        for key, value in config.items()
        if not key.startswith("_")
    }


def validate_resolution(camera: dict[str, Any]) -> tuple[int, int]:
    """Return a strictly positive image width/height."""
    width = int(camera.get("image_width", 0))
    height = int(camera.get("image_height", 0))
    if width <= 0 or height <= 0:
        raise ConfigurationError(
            f"Invalid image resolution {width}x{height}; both values must be positive"
        )
    return width, height


def validate_format_version(data: dict[str, Any], source: Path | str) -> None:
    """Reject files with absent or unsupported format versions."""
    version = str(data.get("format_version", ""))
    if version != FORMAT_VERSION:
        raise ConfigurationError(
            f"Unsupported format_version {version!r} in {source}; "
            f"expected {FORMAT_VERSION!r}"
        )


def validate_length_units(data: dict[str, Any], source: Path | str) -> None:
    """Reject result files that declare a non-metric translation unit."""
    units = data.get("units", {})
    declared = str(units.get("length", units.get("translation", LENGTH_UNIT)))
    if declared not in {"metre", "meter", "m"}:
        raise ConfigurationError(
            f"Unsupported length unit {declared!r} in {source}; expected metre"
        )


def metres_to_millimetres(value: float) -> float:
    """Convert metres to millimetres."""
    return float(value) * 1000.0


def millimetres_to_metres(value: float) -> float:
    """Convert millimetres to metres."""
    return float(value) / 1000.0

