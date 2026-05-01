"""Generate README gallery images.

Assumptions:
    - Gallery frames are illustrative previews, not publication-grade
      simulations.
    - Galaxy state uses the existing float64 dynamics path.
    - Collider previews use float32 particle visuals for interactive speed.
    - The script writes only to ``docs/images``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from twingalaxynet.colliders import (
    BodyCollisionSimulation,
    ColliderConfig,
    make_frame_from_collider,
)
from twingalaxynet.gpu_render import make_frame_from_simulation
from twingalaxynet.simulation import SimulationConfig, TwinGalaxySimulation


OUTPUT_DIR = Path("docs/images")


def save_gallery_frame(path: Path, frame) -> None:
    """Save a gallery frame with a stable dark background."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, frame)
    print(f"wrote {path}")


def make_galaxy() -> None:
    """Create an infrared-style twin galaxy preview."""

    sim = TwinGalaxySimulation(
        SimulationConfig(particles=90000, seed=8),
        devices="auto",
    )
    sim.step(130)
    frame = make_frame_from_simulation(
        sim,
        resolution=720,
        extent_kpc=72.0,
        yaw_deg=18.0,
        pitch_deg=20.0,
        theme_name="jwst",
        bloom=True,
        dust=True,
        gas=True,
        starburst=True,
    )
    save_gallery_frame(OUTPUT_DIR / "galaxy_jwst.png", frame)


def make_planet() -> None:
    """Create a rocky planet impact preview."""

    sim = BodyCollisionSimulation(
        ColliderConfig(
            kind="planet",
            particles=90000,
            center_offset=1.75,
            impact_parameter=0.46,
            approach_speed=0.16,
            seed=22,
        ),
        devices="auto",
    )
    sim.step(40)
    frame = make_frame_from_collider(
        sim,
        resolution=720,
        extent=8.5,
        yaw_deg=-12.0,
        pitch_deg=18.0,
        bloom=True,
    )
    save_gallery_frame(OUTPUT_DIR / "planet_impact.png", frame)


def make_star() -> None:
    """Create a stellar merger preview."""

    sim = BodyCollisionSimulation(
        ColliderConfig(
            kind="star",
            particles=90000,
            center_offset=1.85,
            impact_parameter=0.38,
            approach_speed=0.13,
            spring_strength=0.055,
            shock_strength=0.40,
            seed=31,
        ),
        devices="auto",
    )
    sim.step(30)
    frame = make_frame_from_collider(
        sim,
        resolution=720,
        extent=4.2,
        yaw_deg=10.0,
        pitch_deg=14.0,
        bloom=True,
    )
    save_gallery_frame(OUTPUT_DIR / "star_merger.png", frame)


def main() -> None:
    """Generate all gallery images."""

    make_galaxy()
    make_planet()
    make_star()


if __name__ == "__main__":
    main()
