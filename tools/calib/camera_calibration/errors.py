"""Domain-specific exceptions with actionable CLI messages."""


class CalibrationError(RuntimeError):
    """Base exception for expected calibration failures."""


class ConfigurationError(CalibrationError):
    """Invalid or inconsistent user configuration."""


class DetectionError(CalibrationError):
    """Target detection or observation-count failure."""


class OptimizationError(CalibrationError):
    """An optimizer failed or returned an invalid physical result."""

