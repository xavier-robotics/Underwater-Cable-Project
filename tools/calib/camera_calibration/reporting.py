"""Shared CSV, overlays, coverage plots, and Markdown reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .errors import CalibrationError
from .target_detector import TargetObservation, TargetSpec


def prepare_output_directories(
    output_dir: Path,
    names: Iterable[str],
) -> dict[str, Path]:
    """Create an output root and its requested child directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"root": output_dir}
    for name in names:
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        result[name] = path
    return result


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise CalibrationError(f"Failed to write debug image: {path}")


def save_observation_images(
    observations: list[TargetObservation],
    spec: TargetSpec,
    directories: dict[str, Path],
) -> None:
    """Save every selected/rejected candidate with its explicit reason."""
    for observation in observations:
        canvas = observation.frame.image.copy()
        if observation.image_points is not None:
            cv2.drawChessboardCorners(
                canvas,
                (spec.cols, spec.rows),
                observation.image_points.astype(np.float32).reshape(-1, 1, 2),
                True,
            )
        state = (
            "selected"
            if observation.selected
            else observation.rejection_reason or "not_selected"
        )
        color = (0, 200, 0) if observation.selected else (0, 0, 220)
        cv2.putText(
            canvas,
            state,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        folder = (
            directories["detected"]
            if observation.selected
            else directories["rejected"]
        )
        _write_image(folder / f"{observation.frame.name}.jpg", canvas)


def write_observations_csv(
    path: Path,
    observations: list[TargetObservation],
    extra_rows: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write stable audit data for every frame, including rejection reason."""
    extra_rows = extra_rows or {}
    fields = [
        "name",
        "source",
        "frame_index",
        "timestamp_sec",
        "detected",
        "accepted",
        "selected",
        "rejection_reason",
        "blur_score",
        "mean_intensity",
        "overexposed_ratio",
        "underexposed_ratio",
        "center_x_normalized",
        "center_y_normalized",
        "area_ratio",
        "angle_deg",
        "tilt_score",
        "coverage_cells",
        "estimated_distance_m",
        "reprojection_error_px",
    ]
    extras = sorted({key for row in extra_rows.values() for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extras)
        writer.writeheader()
        for item in observations:
            row: dict[str, Any] = {
                "name": item.frame.name,
                "source": item.frame.source,
                "frame_index": item.frame.frame_index,
                "timestamp_sec": f"{item.frame.timestamp_sec:.6f}",
                "detected": item.detected,
                "accepted": item.accepted,
                "selected": item.selected,
                "rejection_reason": item.rejection_reason,
                "blur_score": f"{item.blur_score:.6f}",
                "mean_intensity": f"{item.mean_intensity:.6f}",
                "overexposed_ratio": f"{item.overexposed_ratio:.8f}",
                "underexposed_ratio": f"{item.underexposed_ratio:.8f}",
                "center_x_normalized": f"{item.center_xy[0]:.8f}",
                "center_y_normalized": f"{item.center_xy[1]:.8f}",
                "area_ratio": f"{item.area_ratio:.8f}",
                "angle_deg": f"{item.angle_deg:.6f}",
                "tilt_score": f"{item.tilt_score:.8f}",
                "coverage_cells": "|".join(map(str, item.coverage_cells)),
                "estimated_distance_m": (
                    ""
                    if item.pose_tvec is None
                    else f"{np.linalg.norm(item.pose_tvec):.8f}"
                ),
                "reprojection_error_px": (
                    ""
                    if item.reprojection_error_px is None
                    else f"{item.reprojection_error_px:.8f}"
                ),
            }
            row.update(extra_rows.get(item.frame.name, {}))
            writer.writerow(row)


def save_coverage_plot(
    path: Path,
    observations: list[TargetObservation],
    image_size: tuple[int, int],
) -> None:
    """Plot image coverage, board scale, distance, and tilt distributions."""
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise CalibrationError("Coverage plot requires a positive image size")
    selected = [item for item in observations if item.selected]
    canvas = np.full((800, 1200, 3), 248, dtype=np.uint8)
    panels = {
        "centers": (20, 55, 580, 765),
        "area": (620, 55, 1180, 270),
        "distance": (620, 315, 1180, 530),
        "tilt": (620, 575, 1180, 765),
    }
    x0, y0, x1, y1 = panels["centers"]
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (110, 110, 110), 1)
    for index in range(1, 3):
        x = int(x0 + (x1 - x0) * index / 3)
        y = int(y0 + (y1 - y0) * index / 3)
        cv2.line(canvas, (x, y0), (x, y1), (190, 190, 190), 1)
        cv2.line(canvas, (x0, y), (x1, y), (190, 190, 190), 1)
    cell_counts = np.zeros(9, dtype=int)
    for item in selected:
        center = (
            int(x0 + item.center_xy[0] * (x1 - x0)),
            int(y0 + item.center_xy[1] * (y1 - y0)),
        )
        radius = max(4, int(np.sqrt(item.area_ratio) * 45))
        cv2.circle(canvas, center, radius, (0, 120, 255), 2)
        for cell in item.coverage_cells:
            if 0 <= cell < 9:
                cell_counts[cell] += 1
    for cell, count in enumerate(cell_counts):
        column, row = cell % 3, cell // 3
        location = (
            int(x0 + (column + 0.08) * (x1 - x0) / 3),
            int(y0 + (row + 0.18) * (y1 - y0) / 3),
        )
        cv2.putText(
            canvas,
            str(int(count)),
            location,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    def histogram(
        panel: tuple[int, int, int, int],
        values: np.ndarray,
        title: str,
        unit: str,
    ) -> None:
        px0, py0, px1, py1 = panel
        cv2.rectangle(canvas, (px0, py0), (px1, py1), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            title,
            (px0 + 8, py0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        finite = values[np.isfinite(values)]
        if not len(finite):
            cv2.putText(
                canvas,
                "not available",
                (px0 + 12, py0 + 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            return
        counts, edges = np.histogram(finite, bins=min(10, max(3, len(finite))))
        chart_top = py0 + 38
        chart_bottom = py1 - 30
        chart_width = px1 - px0 - 24
        bar_width = chart_width / max(len(counts), 1)
        maximum = max(int(np.max(counts)), 1)
        for index, count in enumerate(counts):
            left = int(px0 + 12 + index * bar_width)
            right = int(px0 + 12 + (index + 1) * bar_width - 2)
            top = int(
                chart_bottom
                - (chart_bottom - chart_top) * float(count) / maximum
            )
            cv2.rectangle(
                canvas,
                (left, top),
                (right, chart_bottom),
                (80, 150, 230),
                -1,
            )
        label = (
            f"min/median/max: {finite.min():.4g} / "
            f"{np.median(finite):.4g} / {finite.max():.4g} {unit}"
        )
        cv2.putText(
            canvas,
            label,
            (px0 + 8, py1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        (
            f"Target center and 3x3 cell coverage "
            f"({len(selected)} selected / {len(observations)} candidates)"
        ),
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    areas = np.asarray([item.area_ratio for item in selected], dtype=float)
    distances = np.asarray(
        [
            (
                float(np.linalg.norm(item.pose_tvec))
                if item.pose_tvec is not None
                else np.nan
            )
            for item in selected
        ],
        dtype=float,
    )
    histogram(panels["area"], areas, "Board image area distribution", "ratio")
    histogram(
        panels["distance"],
        distances,
        "Estimated target distance distribution",
        "m",
    )
    histogram(
        panels["tilt"],
        np.asarray([item.tilt_score for item in selected], dtype=float),
        "Target tilt proxy distribution",
        "score",
    )
    _write_image(path, canvas)


def save_reprojection_overlay(
    path: Path,
    image: np.ndarray,
    detected: np.ndarray,
    projected: np.ndarray,
) -> None:
    """Draw detected green points and projected magenta crosses."""
    canvas = image.copy()
    for observed, estimate in zip(detected, projected):
        first = tuple(np.round(observed).astype(int))
        second = tuple(np.round(estimate).astype(int))
        cv2.circle(canvas, first, 4, (0, 220, 0), -1)
        cv2.drawMarker(canvas, second, (220, 0, 220), cv2.MARKER_CROSS, 9, 2)
        cv2.line(canvas, first, second, (0, 180, 255), 1)
    _write_image(path, canvas)


def write_markdown_report(
    path: Path,
    title: str,
    sections: list[tuple[str, str]],
) -> None:
    """Write a concise UTF-8 Markdown report."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.rstrip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
