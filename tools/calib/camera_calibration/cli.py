"""Unified command-line interface."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import load_yaml
from .errors import CalibrationError, ConfigurationError
from .extrinsic_calibrator import calibrate_extrinsics
from .intrinsic_calibrator import calibrate_intrinsics
from .validation import validate_calibration


COMMANDS: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "calibrate-intrinsics": ("intrinsic", calibrate_intrinsics),
    "calibrate-extrinsics": ("extrinsic", calibrate_extrinsics),
    "validate": ("validation", validate_calibration),
}


def build_parser() -> argparse.ArgumentParser:
    """Create the stable public CLI."""
    parser = argparse.ArgumentParser(
        prog="camera_calibration",
        description="In-medium intrinsics and fixed mechanical extrinsics",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--verbose", action="store_true")
    return parser


def run_command(command: str, config_path: Path) -> dict[str, Any]:
    """Load, mode-check, and dispatch one command."""
    if command not in COMMANDS:
        raise ConfigurationError(f"Unknown command: {command}")
    expected_mode, function = COMMANDS[command]
    config = load_yaml(config_path)
    actual_mode = str(config.get("mode", expected_mode)).lower()
    actual_mode = {"validate": "validation"}.get(actual_mode, actual_mode)
    if actual_mode != expected_mode:
        raise ConfigurationError(
            f"{command} expects mode: {expected_mode}; {config_path} declares "
            f"mode: {actual_mode}"
        )
    return function(config)


def main() -> None:
    """Console entry point."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_command(args.command, args.config)
    except CalibrationError as exc:
        logging.error("%s", exc)
        raise SystemExit(2) from exc
    print(f"Calibration command completed: {result['result_file']}")


if __name__ == "__main__":
    main()
