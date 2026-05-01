"""Torch-based renderer that keeps particle projection on the GPU."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

from twingalaxynet.simulation import TwinGalaxySimulation
from twingalaxynet.themes import VisualTheme, get_theme


def make_frame_from_simulation(
    simulation: TwinGalaxySimulation,
    resolution: int = 540,
    extent_kpc: float = 72.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 17.0,
    theme_name: str = "natural",
    bloom: bool = True,
    dust: bool = True,
    gas: bool = True,
    starburst: bool = True,
) -> np.ndarray:
    """Render the live simulation without copying all particles to CPU.

    The renderer bins particle light with ``torch.bincount`` on each device and
    only transfers the final RGB image to CPU. This keeps dual-GPU particle
    simulations interactive at substantially higher particle counts.
    """

    if resolution < 128:
        raise ValueError("resolution must be at least 128")
    if extent_kpc <= 0.0:
        raise ValueError("extent_kpc must be positive")

    theme = get_theme(theme_name)
    primary = simulation.devices[0]
    flat_size = resolution * resolution
    canvas = torch.zeros((flat_size, 3), dtype=torch.float32, device=primary)
    center_primary = simulation.center_position.to(primary, torch.float32)
    separation = torch.linalg.norm(center_primary[1] - center_primary[0])
    encounter = torch.clamp((46.0 - separation) / 32.0, min=0.0, max=1.0) ** 2

    for shard in simulation.shards:
        shard_image = _render_shard(
            shard_position=shard.position,
            shard_color=shard.color,
            shard_luminosity=shard.luminosity,
            center_position=simulation.center_position.to(shard.device, torch.float32),
            resolution=resolution,
            extent_kpc=extent_kpc,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            theme=theme,
            encounter=encounter.to(shard.device),
            include_gas=gas,
            include_starburst=starburst,
        )
        canvas += shard_image.to(primary)

    image = canvas.reshape((resolution, resolution, 3))
    if dust and theme.dust_strength > 0.0:
        image = _apply_dust(image, theme.dust_strength)
    if bloom and theme.bloom_strength > 0.0:
        image = _bloom(image, theme.bloom_strength)
    image = torch.asinh(image * theme.stretch)
    scale = torch.quantile(image.flatten(), 0.9982)
    if not torch.isfinite(scale) or scale <= 0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=primary)
    image = torch.clamp(image / scale, 0.0, 1.0)
    image = torch.pow(image, theme.gamma)
    image = torch.clamp(_background(resolution, primary, theme) + image, 0.0, 1.0)
    return image.detach().cpu().numpy()


def _render_shard(
    shard_position: torch.Tensor,
    shard_color: torch.Tensor,
    shard_luminosity: torch.Tensor,
    center_position: torch.Tensor,
    resolution: int,
    extent_kpc: float,
    yaw_deg: float,
    pitch_deg: float,
    theme: VisualTheme,
    encounter: torch.Tensor,
    include_gas: bool,
    include_starburst: bool,
) -> torch.Tensor:
    device = shard_position.device
    position = shard_position.to(torch.float32)
    x, y, z = _project(position, yaw_deg, pitch_deg)
    scale = resolution / (2.0 * extent_kpc)
    xi = ((x + extent_kpc) * scale).to(torch.int64)
    yi = ((y + extent_kpc) * scale).to(torch.int64)
    valid = (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
    flat = yi[valid] * resolution + xi[valid]
    flat_size = resolution * resolution

    color_matrix = torch.as_tensor(theme.color_matrix, device=device)
    base_color = torch.clamp(shard_color @ color_matrix.T, 0.0, 2.5)
    depth_weight = torch.clamp(1.15 + z / (2.5 * extent_kpc), 0.45, 1.55)
    weights = shard_luminosity * depth_weight

    image = _accumulate(flat, weights[valid], base_color[valid], flat_size)

    if include_gas and theme.gas_strength > 0.0:
        gas_weight = _gas_weight(position, center_position, weights, encounter)
        gas_color = torch.as_tensor(theme.gas_color, dtype=torch.float32, device=device)
        image += theme.gas_strength * _accumulate_solid(
            flat, gas_weight[valid], gas_color, flat_size
        )

    if include_starburst and theme.starburst_strength > 0.0:
        burst_weight = _starburst_weight(position, center_position, weights, encounter)
        burst_color = torch.as_tensor(
            theme.starburst_color, dtype=torch.float32, device=device
        )
        image += theme.starburst_strength * _accumulate_solid(
            flat, burst_weight[valid], burst_color, flat_size
        )

    return image


def _project(
    position: torch.Tensor,
    yaw_deg: float,
    pitch_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.tensor(np.deg2rad(yaw_deg), dtype=torch.float32, device=position.device)
    pitch = torch.tensor(
        np.deg2rad(pitch_deg), dtype=torch.float32, device=position.device
    )
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    source_x = position[:, 0]
    source_y = position[:, 1]
    source_z = position[:, 2]
    yaw_x = cos_y * source_x + sin_y * source_z
    yaw_z = -sin_y * source_x + cos_y * source_z
    return yaw_x, cos_p * source_y - sin_p * yaw_z, sin_p * source_y + cos_p * yaw_z


def _accumulate(
    flat: torch.Tensor,
    weights: torch.Tensor,
    color: torch.Tensor,
    flat_size: int,
) -> torch.Tensor:
    channels = []
    for channel in range(3):
        channels.append(
            torch.bincount(
                flat,
                weights=weights * color[:, channel],
                minlength=flat_size,
            )
        )
    return torch.stack(channels, dim=1).to(torch.float32)


def _accumulate_solid(
    flat: torch.Tensor,
    weights: torch.Tensor,
    color: torch.Tensor,
    flat_size: int,
) -> torch.Tensor:
    density = torch.bincount(flat, weights=weights, minlength=flat_size).to(torch.float32)
    return density[:, None] * color[None, :]


def _gas_weight(
    position: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    encounter: torch.Tensor,
) -> torch.Tensor:
    nearest = torch.min(
        torch.linalg.norm(position[:, None, :] - centers[None, :, :], dim=2),
        dim=1,
    ).values
    disk_gas = torch.exp(-nearest / 16.0)
    bridge = _bridge_weight(position, centers)
    return weights * (0.28 * disk_gas + encounter * 1.2 * bridge)


def _starburst_weight(
    position: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    encounter: torch.Tensor,
) -> torch.Tensor:
    nearest = torch.min(
        torch.linalg.norm(position[:, None, :] - centers[None, :, :], dim=2),
        dim=1,
    ).values
    nuclear = torch.exp(-(nearest / 5.5) ** 2)
    bridge = _bridge_weight(position, centers)
    return weights * encounter * (1.6 * nuclear + 0.65 * bridge)


def _bridge_weight(position: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    start = centers[0]
    end = centers[1]
    axis = end - start
    length2 = torch.clamp(torch.sum(axis * axis), min=1e-6)
    t = torch.clamp(torch.sum((position - start) * axis, dim=1) / length2, 0.0, 1.0)
    closest = start + t[:, None] * axis
    distance = torch.linalg.norm(position - closest, dim=1)
    return torch.exp(-(distance / 11.0) ** 2)


def _apply_dust(image: torch.Tensor, strength: float) -> torch.Tensor:
    density = torch.mean(image, dim=2, keepdim=True)
    dust = _blur(density, passes=2)
    scale = torch.quantile(dust.flatten(), 0.985)
    if not torch.isfinite(scale) or scale <= 0:
        return image
    lane = torch.clamp(dust / scale, 0.0, 1.0)
    attenuation = 1.0 - strength * lane
    reddening = torch.cat(
        [attenuation + 0.08 * strength * lane, attenuation, attenuation - 0.12 * lane],
        dim=2,
    )
    return torch.clamp(image * reddening, min=0.0)


def _bloom(image: torch.Tensor, strength: float) -> torch.Tensor:
    glow = _blur(image, passes=2)
    wide = _blur(glow, passes=2)
    return image + strength * (0.50 * glow + 0.22 * wide)


def _blur(image: torch.Tensor, passes: int) -> torch.Tensor:
    blurred = image
    for _ in range(passes):
        blurred = (
            blurred * 0.30
            + 0.105
            * (
                _shift(blurred, 1, 0)
                + _shift(blurred, -1, 0)
                + _shift(blurred, 0, 1)
                + _shift(blurred, 0, -1)
            )
            + 0.055
            * (
                _shift(blurred, 1, 1)
                + _shift(blurred, 1, -1)
                + _shift(blurred, -1, 1)
                + _shift(blurred, -1, -1)
            )
        )
    return blurred


def _shift(image: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    shifted = torch.zeros_like(image)
    row_source = slice(max(-rows, 0), image.shape[0] - max(rows, 0))
    row_target = slice(max(rows, 0), image.shape[0] - max(-rows, 0))
    col_source = slice(max(-cols, 0), image.shape[1] - max(cols, 0))
    col_target = slice(max(cols, 0), image.shape[1] - max(-cols, 0))
    shifted[row_target, col_target] = image[row_source, col_source]
    return shifted


@lru_cache(maxsize=16)
def _background_cpu(
    resolution: int,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
) -> np.ndarray:
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    radius = np.sqrt((xx / resolution - 0.5) ** 2 + (yy / resolution - 0.5) ** 2)
    vignette = np.clip(1.0 - 1.55 * radius, 0.0, 1.0).astype(np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    return low_arr[None, None, :] + vignette[:, :, None] * (
        high_arr - low_arr
    )[None, None, :]


def _background(
    resolution: int,
    device: torch.device,
    theme: VisualTheme,
) -> torch.Tensor:
    return torch.as_tensor(
        _background_cpu(resolution, theme.background_low, theme.background_high),
        dtype=torch.float32,
        device=device,
    )
