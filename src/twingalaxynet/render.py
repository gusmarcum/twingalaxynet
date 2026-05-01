"""Rendering utilities for cinematic galaxy density images."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def make_frame(
    position: np.ndarray,
    color: np.ndarray,
    luminosity: np.ndarray,
    resolution: int = 720,
    extent_kpc: float = 72.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 17.0,
    bloom: bool = True,
) -> np.ndarray:
    """Project particles into an RGB image.

    The renderer uses weighted 2D histograms and an asinh stretch, a common
    astronomical visualization technique for preserving bright cores while
    retaining faint tidal structure.
    """

    if resolution < 128:
        raise ValueError("resolution must be at least 128")
    if extent_kpc <= 0:
        raise ValueError("extent_kpc must be positive")
    if not np.isfinite(position).all():
        raise FloatingPointError("cannot render non-finite positions")

    projected = project(position, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    x = projected[:, 0]
    y = projected[:, 1]
    depth = projected[:, 2]
    depth_weight = np.clip(1.15 + depth / (2.5 * extent_kpc), 0.45, 1.55)
    weights = luminosity.astype(np.float32) * depth_weight.astype(np.float32)

    image = _bin_particles(x, y, color, weights, resolution, extent_kpc)

    if bloom:
        image = _soft_bloom(image)
    image = np.arcsinh(image * 0.09)
    max_value = np.percentile(image, 99.82)
    if max_value <= 0 or not np.isfinite(max_value):
        max_value = 1.0
    image = np.clip(image / max_value, 0.0, 1.0)

    background = _background(resolution)
    image = np.clip(background + image**0.72, 0.0, 1.0)
    return image


def project(position: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotate 3D positions for a camera view."""

    yaw = np.deg2rad(yaw_deg).astype(np.float32)
    pitch = np.deg2rad(pitch_deg).astype(np.float32)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    source = position.astype(np.float32, copy=False)
    x = source[:, 0]
    y = source[:, 1]
    z = source[:, 2]

    yaw_x = cos_y * x + sin_y * z
    yaw_z = -sin_y * x + cos_y * z
    out = np.empty_like(source)
    out[:, 0] = yaw_x
    out[:, 1] = cos_p * y - sin_p * yaw_z
    out[:, 2] = sin_p * y + cos_p * yaw_z
    return out


def _bin_particles(
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    weights: np.ndarray,
    resolution: int,
    extent_kpc: float,
) -> np.ndarray:
    """Accumulate particle light with fast integer binning."""

    scale = resolution / (2.0 * extent_kpc)
    xi = ((x + extent_kpc) * scale).astype(np.int32)
    yi = ((y + extent_kpc) * scale).astype(np.int32)
    valid = (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
    flat = yi[valid] * resolution + xi[valid]
    flat_size = resolution * resolution

    image = np.empty((flat_size, 3), dtype=np.float32)
    valid_weights = weights[valid]
    valid_color = color[valid]
    for channel in range(3):
        image[:, channel] = np.bincount(
            flat,
            weights=valid_weights * valid_color[:, channel],
            minlength=flat_size,
        ).astype(np.float32, copy=False)
    return image.reshape((resolution, resolution, 3))


def _soft_bloom(image: np.ndarray) -> np.ndarray:
    """Add a cheap glow with fixed image shifts instead of Python row loops."""

    bloom = image * 0.30
    bloom += 0.105 * (
        _shift(image, 1, 0)
        + _shift(image, -1, 0)
        + _shift(image, 0, 1)
        + _shift(image, 0, -1)
    )
    bloom += 0.055 * (
        _shift(image, 1, 1)
        + _shift(image, 1, -1)
        + _shift(image, -1, 1)
        + _shift(image, -1, -1)
    )
    bloom += 0.035 * (
        _shift(image, 2, 0)
        + _shift(image, -2, 0)
        + _shift(image, 0, 2)
        + _shift(image, 0, -2)
    )
    return image + 0.75 * bloom


def _shift(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Shift an image with zero-filled edges."""

    shifted = np.zeros_like(image)
    row_source = slice(max(-rows, 0), image.shape[0] - max(rows, 0))
    row_target = slice(max(rows, 0), image.shape[0] - max(-rows, 0))
    col_source = slice(max(-cols, 0), image.shape[1] - max(cols, 0))
    col_target = slice(max(cols, 0), image.shape[1] - max(-cols, 0))
    shifted[row_target, col_target] = image[row_source, col_source]
    return shifted


@lru_cache(maxsize=8)
def _background(resolution: int) -> np.ndarray:
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    radius = np.sqrt((xx / resolution - 0.5) ** 2 + (yy / resolution - 0.5) ** 2)
    vignette = np.clip(1.0 - 1.55 * radius, 0.0, 1.0)
    background = np.zeros((resolution, resolution, 3), dtype=np.float32)
    background[:, :, 0] = 0.006 + 0.015 * vignette
    background[:, :, 1] = 0.008 + 0.014 * vignette
    background[:, :, 2] = 0.014 + 0.028 * vignette
    return background
