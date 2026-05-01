"""Small non-interactive TwinGalaxyNET smoke test."""

from pathlib import Path

import matplotlib.pyplot as plt

from twingalaxynet.render import make_frame
from twingalaxynet.simulation import SimulationConfig, TwinGalaxySimulation


def main() -> None:
    """Run a tiny simulation and save one preview image."""

    sim = TwinGalaxySimulation(
        SimulationConfig(particles=2000, dt_myr=0.35, seed=11),
        devices="cpu",
    )
    sim.step(2)
    position, color, luminosity = sim.snapshot()
    frame = make_frame(position, color, luminosity, resolution=256)
    output = Path("renders") / "smoke_preview.png"
    output.parent.mkdir(exist_ok=True)
    plt.imsave(output, frame)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
