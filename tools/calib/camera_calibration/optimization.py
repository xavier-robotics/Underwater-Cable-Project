"""SciPy-backed robust optimization with strict convergence handling."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .errors import OptimizationError


SUPPORTED_LOSSES = {"linear", "soft_l1", "huber", "cauchy", "arctan"}


def scipy_least_squares(
    residual: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    loss: str = "huber",
    f_scale: float = 1.0,
    max_iterations: int = 200,
    bounds: tuple[np.ndarray | float, np.ndarray | float] = (-np.inf, np.inf),
    jac_sparsity: Any | None = None,
    x_scale: str | np.ndarray = "jac",
    tolerance: float = 1e-8,
) -> Any:
    """Run least-squares and fail unless SciPy reports a finite success."""
    if loss not in SUPPORTED_LOSSES:
        raise OptimizationError(
            f"Unsupported robust loss {loss!r}; choose {sorted(SUPPORTED_LOSSES)}"
        )
    if max_iterations <= 0 or f_scale <= 0 or tolerance <= 0:
        raise OptimizationError(
            "max_iterations, f_scale, and tolerance must be positive"
        )
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise OptimizationError(
            "SciPy >=1.10 is required for nonlinear calibration"
        ) from exc
    try:
        result = least_squares(
            residual,
            np.asarray(initial, dtype=np.float64),
            loss=loss,
            f_scale=float(f_scale),
            max_nfev=int(max_iterations),
            bounds=bounds,
            jac_sparsity=jac_sparsity,
            x_scale=x_scale,
            ftol=float(tolerance),
            xtol=float(tolerance),
            gtol=float(tolerance),
        )
    except (ValueError, FloatingPointError) as exc:
        raise OptimizationError(f"Optimization could not start: {exc}") from exc
    if not result.success or not np.all(np.isfinite(result.x)):
        raise OptimizationError(
            f"Optimization failed: status={result.status}, message={result.message}, "
            f"evaluations={result.nfev}, cost={result.cost:.8g}, "
            f"optimality={result.optimality:.8g}"
        )
    final = np.asarray(residual(result.x), dtype=np.float64)
    if final.size == 0 or not np.all(np.isfinite(final)):
        raise OptimizationError("Optimization returned empty/non-finite residuals")
    return result
