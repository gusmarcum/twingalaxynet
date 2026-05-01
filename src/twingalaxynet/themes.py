"""Visual themes for TwinGalaxyNET.

Themes are deliberately separated from the physics. The simulation evolves
tracer stars in physical units, while the theme maps stellar populations and
gas/starburst overlays into an observational style.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualTheme:
    """Parameters controlling the rendered observational appearance."""

    name: str
    color_matrix: np.ndarray
    background_low: tuple[float, float, float]
    background_high: tuple[float, float, float]
    gas_color: tuple[float, float, float]
    starburst_color: tuple[float, float, float]
    stretch: float
    gamma: float
    bloom_strength: float
    dust_strength: float
    gas_strength: float
    starburst_strength: float


THEMES = {
    "natural": VisualTheme(
        name="natural",
        color_matrix=np.array(
            [[1.10, 0.03, 0.00], [0.02, 1.00, 0.04], [0.00, 0.05, 1.12]],
            dtype=np.float32,
        ),
        background_low=(0.005, 0.007, 0.014),
        background_high=(0.022, 0.024, 0.042),
        gas_color=(0.42, 0.62, 1.00),
        starburst_color=(1.00, 0.45, 0.72),
        stretch=0.090,
        gamma=0.72,
        bloom_strength=0.70,
        dust_strength=0.22,
        gas_strength=0.18,
        starburst_strength=0.95,
    ),
    "jwst": VisualTheme(
        name="jwst",
        color_matrix=np.array(
            [[1.42, 0.18, 0.02], [0.18, 0.86, 0.12], [0.04, 0.18, 0.64]],
            dtype=np.float32,
        ),
        background_low=(0.011, 0.006, 0.004),
        background_high=(0.040, 0.022, 0.013),
        gas_color=(1.00, 0.38, 0.18),
        starburst_color=(1.00, 0.82, 0.38),
        stretch=0.075,
        gamma=0.66,
        bloom_strength=0.85,
        dust_strength=0.30,
        gas_strength=0.30,
        starburst_strength=1.20,
    ),
    "hubble": VisualTheme(
        name="hubble",
        color_matrix=np.array(
            [[1.05, 0.10, 0.02], [0.04, 0.96, 0.12], [0.02, 0.10, 1.22]],
            dtype=np.float32,
        ),
        background_low=(0.004, 0.006, 0.016),
        background_high=(0.018, 0.025, 0.050),
        gas_color=(0.24, 0.72, 1.00),
        starburst_color=(0.95, 0.35, 0.90),
        stretch=0.086,
        gamma=0.70,
        bloom_strength=0.78,
        dust_strength=0.24,
        gas_strength=0.22,
        starburst_strength=1.05,
    ),
    "xray": VisualTheme(
        name="xray",
        color_matrix=np.array(
            [[0.38, 0.03, 0.15], [0.05, 0.72, 0.35], [0.18, 0.20, 1.55]],
            dtype=np.float32,
        ),
        background_low=(0.002, 0.005, 0.011),
        background_high=(0.006, 0.024, 0.045),
        gas_color=(0.10, 1.00, 0.95),
        starburst_color=(1.00, 0.16, 0.56),
        stretch=0.120,
        gamma=0.62,
        bloom_strength=1.00,
        dust_strength=0.08,
        gas_strength=0.36,
        starburst_strength=1.45,
    ),
    "plate": VisualTheme(
        name="plate",
        color_matrix=np.array(
            [[0.92, 0.92, 0.92], [0.88, 0.88, 0.88], [0.78, 0.78, 0.78]],
            dtype=np.float32,
        ),
        background_low=(0.020, 0.018, 0.014),
        background_high=(0.085, 0.076, 0.055),
        gas_color=(0.80, 0.72, 0.55),
        starburst_color=(1.00, 0.92, 0.72),
        stretch=0.070,
        gamma=0.78,
        bloom_strength=0.35,
        dust_strength=0.34,
        gas_strength=0.08,
        starburst_strength=0.42,
    ),
}


def get_theme(name: str) -> VisualTheme:
    """Return a named visual theme."""

    try:
        return THEMES[name.lower()]
    except KeyError as exc:
        names = ", ".join(sorted(THEMES))
        raise ValueError(f"unknown theme {name!r}; choose one of: {names}") from exc
