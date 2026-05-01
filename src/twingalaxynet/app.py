"""Interactive TwinGalaxyNET application."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from twingalaxynet.colliders import (
    BodyCollisionSimulation,
    ColliderConfig,
    make_frame_from_collider,
)
from twingalaxynet.gpu_render import make_frame_from_simulation
from twingalaxynet.render import make_frame
from twingalaxynet.simulation import SimulationConfig, TwinGalaxySimulation
from twingalaxynet.themes import THEMES


class GalaxyApp:
    """Real-time viewer with OpenCV and Matplotlib display backends."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config, self.simulation = self._make_simulation()
        self.paused = False
        self.speed = args.steps_per_frame
        self.extent = args.extent if args.extent is not None else self._default_extent()
        self.yaw = args.yaw
        self.pitch = args.pitch
        self.auto_camera = args.auto_camera
        self.theme_names = sorted(THEMES)
        self.theme_index = self.theme_names.index(args.theme)
        self.frame_index = 0
        self.last_time = perf_counter()
        self.measured_fps = 0.0
        self.current_frame: Optional[np.ndarray] = None
        self.display = "none" if args.export_frames else self._select_display(args.display)
        self.figure = None
        self.axis = None
        self.image = None
        self.overlay = None

        if self.display == "matplotlib":
            self._init_matplotlib()
        elif self.display == "opencv":
            self._init_opencv()

    def _make_simulation(
        self,
    ) -> tuple[SimulationConfig | ColliderConfig, TwinGalaxySimulation | BodyCollisionSimulation]:
        """Create the requested scene simulation."""

        if self.args.mode == "galaxy":
            config = SimulationConfig(
                particles=self.args.particles,
                dt_myr=0.35 if self.args.dt is None else self.args.dt,
                seed=self.args.seed,
            )
            return config, TwinGalaxySimulation(config, devices=self.args.devices)

        config = ColliderConfig(
            kind=self.args.mode,
            particles=self.args.particles,
            dt=self._default_collider_dt() if self.args.dt is None else self.args.dt,
            seed=self.args.seed,
            center_offset=(
                (2.45 if self.args.mode == "planet" else 2.45)
                if self.args.body_offset is None
                else self.args.body_offset
            ),
            impact_parameter=(
                0.52 if self.args.impact_parameter is None else self.args.impact_parameter
            ),
            approach_speed=(
                (0.095 if self.args.mode == "planet" else 0.090)
                if self.args.impact_speed is None
                else self.args.impact_speed
            ),
            gravity_strength=0.0010 if self.args.mode == "planet" else 0.00055,
            spring_strength=0.115 if self.args.mode == "planet" else 0.060,
            shock_strength=0.54 if self.args.mode == "planet" else 0.34,
            runaway_speed=3.0 if self.args.mode == "planet" else 2.4,
        )
        return config, BodyCollisionSimulation(config, devices=self.args.devices)

    def _default_extent(self) -> float:
        """Return a mode-appropriate view extent."""

        if self.args.mode == "galaxy":
            return 72.0
        return 4.6

    def _default_collider_dt(self) -> float:
        """Return mode-specific collider time step."""

        return 0.055 if self.args.mode == "planet" else 0.070

    def _select_display(self, requested: str) -> str:
        """Select a live display backend."""

        if requested != "auto":
            return requested
        try:
            import cv2  # noqa: F401
        except ImportError:
            return "matplotlib"
        return "opencv"

    def _init_matplotlib(self) -> None:
        """Initialize the slower but portable Matplotlib viewer."""

        plt.style.use("dark_background")
        self.figure, self.axis = plt.subplots(figsize=(9, 9))
        self.figure.canvas.manager.set_window_title("TwinGalaxyNET")
        self.axis.set_axis_off()
        self.image = self.axis.imshow(
            np.zeros((self.args.resolution, self.args.resolution, 3), dtype=np.float32),
            origin="lower",
            interpolation="bilinear",
        )
        self.overlay = self.axis.text(
            0.02,
            0.98,
            "",
            transform=self.axis.transAxes,
            va="top",
            ha="left",
            color=(0.92, 0.95, 1.0),
            fontsize=10,
            family="monospace",
        )
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)

    def _init_opencv(self) -> None:
        """Initialize the faster OpenCV image viewer."""

        import cv2

        cv2.namedWindow("TwinGalaxyNET", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TwinGalaxyNET", self.args.resolution, self.args.resolution)

    def on_key(self, event) -> None:  # noqa: ANN001
        """Handle keyboard input from Matplotlib."""

        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "up":
            self.speed = min(self.speed + 1, 40)
        elif event.key == "down":
            self.speed = max(self.speed - 1, 1)
        elif event.key == "left":
            self.yaw -= 5.0
        elif event.key == "right":
            self.yaw += 5.0
        elif event.key == "[":
            self.extent = min(self.extent * 1.12, 180.0)
        elif event.key == "]":
            self.extent = max(self.extent / 1.12, 24.0)
        elif event.key == "s":
            self.save_frame()
        elif event.key == "t":
            self.theme_index = (self.theme_index + 1) % len(self.theme_names)
            self.args.theme = self.theme_names[self.theme_index]
        elif event.key == "c":
            self.auto_camera = not self.auto_camera
        elif event.key == "r":
            self.simulation.reset()
        elif event.key == "q":
            plt.close(self.figure)

    def run(self) -> None:
        """Start the live viewer or export mode."""

        if self.args.export_frames:
            self.export_frames(self.args.export_frames)
            return

        if self.display == "opencv":
            self.run_opencv()
            return

        self.run_matplotlib()

    def run_matplotlib(self) -> None:
        """Run the Matplotlib display loop."""

        if self.figure is None:
            raise RuntimeError("Matplotlib display was not initialized")
        plt.ion()
        while plt.fignum_exists(self.figure.number):
            self.update()
            self.image.set_data(self.current_frame)
            self.overlay.set_text(self.status_text())
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

    def run_opencv(self) -> None:
        """Run the OpenCV display loop."""

        import cv2

        while True:
            self.update()
            display_frame = self.frame_for_opencv()
            cv2.imshow("TwinGalaxyNET", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if self.handle_opencv_key(key):
                break
        cv2.destroyWindow("TwinGalaxyNET")

    def update(self) -> None:
        """Advance physics and redraw the current frame."""

        if not self.paused:
            self.simulation.step(self.speed)
        if self.auto_camera:
            self.yaw += 0.18
        self.current_frame = self.render_frame()
        now = perf_counter()
        self.measured_fps = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now
        self.frame_index += 1

    def status_text(self) -> str:
        """Return the current HUD text."""

        devices = ", ".join(str(device) for device in self.simulation.devices)
        state = "paused" if self.paused else "running"
        controls = (
            "Space pause | W/S speed | A/D camera | T theme | C camera | P save | Q quit"
            if self.display == "opencv"
            else "Space pause | arrows speed/camera | T theme | C camera | S save | Q quit"
        )
        if self.args.mode == "galaxy":
            metrics = (
                f"t = {self.simulation.time_myr:7.1f} Myr   "
                f"speed = {self.speed:02d}x   fps = {self.measured_fps:4.1f}\n"
                f"sep = {self.simulation.separation_kpc():5.1f} kpc   "
                f"vrel = {self.simulation.relative_speed_km_s():5.0f} km/s"
            )
        else:
            metrics = (
                f"t = {self.simulation.time:7.2f} {self.config.time_unit}   "
                f"speed = {self.speed:02d}x   fps = {self.measured_fps:4.1f}\n"
                f"sep = {self.simulation.separation_display():5.2f} "
                f"{self.config.distance_unit}   "
                f"vrel = {self.simulation.relative_speed_display():5.1f} km/s"
            )
        return (
            f"TwinGalaxyNET  {state}  mode={self.args.mode}  theme={self.args.theme}\n"
            f"{metrics}\n"
            f"particles = {self.config.particles:,}   "
            f"renderer = {self.args.renderer}   devices = {devices}\n"
            f"{controls}"
        )

    def frame_for_opencv(self) -> np.ndarray:
        """Return an 8-bit BGR frame with a compact OpenCV HUD."""

        import cv2

        if self.current_frame is None:
            raise RuntimeError("no frame has been rendered yet")
        frame = np.clip(self.current_frame * 255.0, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        lines = self.status_text().splitlines()
        line_height = 18
        pad = 8
        box_height = pad * 2 + line_height * len(lines)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], box_height),
            (8, 10, 18),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, dst=frame)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (pad, pad + 13 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (232, 240, 255),
                1,
                cv2.LINE_AA,
            )
        return frame

    def handle_opencv_key(self, key: int) -> bool:
        """Handle OpenCV key input. Return True when the loop should quit."""

        if key == 255:
            return False
        if key in (ord("q"), 27):
            return True
        if key == ord(" "):
            self.paused = not self.paused
        elif key in (82, ord("w")):
            self.speed = min(self.speed + 1, 40)
        elif key in (84, ord("s")):
            self.speed = max(self.speed - 1, 1)
        elif key in (81, ord("a")):
            self.yaw -= 5.0
        elif key in (83, ord("d")):
            self.yaw += 5.0
        elif key == ord("["):
            self.extent = min(self.extent * 1.12, 180.0)
        elif key == ord("]"):
            self.extent = max(self.extent / 1.12, 24.0)
        elif key == ord("p"):
            self.save_frame()
        elif key == ord("t"):
            self.theme_index = (self.theme_index + 1) % len(self.theme_names)
            self.args.theme = self.theme_names[self.theme_index]
        elif key == ord("c"):
            self.auto_camera = not self.auto_camera
        elif key == ord("r"):
            self.simulation.reset()
        return False

    def render_frame(self) -> np.ndarray:
        """Render the current simulation frame."""

        if self.args.mode != "galaxy":
            return make_frame_from_collider(
                self.simulation,
                resolution=self.args.resolution,
                extent=self.extent,
                yaw_deg=self.yaw,
                pitch_deg=self.pitch,
                bloom=not self.args.no_bloom,
            )

        if self.args.renderer == "gpu":
            return make_frame_from_simulation(
                self.simulation,
                resolution=self.args.resolution,
                extent_kpc=self.extent,
                yaw_deg=self.yaw,
                pitch_deg=self.pitch,
                theme_name=self.args.theme,
                bloom=not self.args.no_bloom,
                dust=not self.args.no_dust,
                gas=not self.args.no_gas,
                starburst=not self.args.no_starburst,
            )

        position, color, luminosity = self.simulation.snapshot()
        return make_frame(
            position,
            color,
            luminosity,
            resolution=self.args.resolution,
            extent_kpc=self.extent,
            yaw_deg=self.yaw,
            pitch_deg=self.pitch,
            bloom=not self.args.no_bloom,
        )

    def save_frame(self) -> Path:
        """Save the current figure frame to the output directory."""

        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.output_dir / f"twingalaxynet_{self.frame_index:05d}.png"
        if self.display == "opencv" and self.current_frame is not None:
            import cv2

            frame = np.clip(self.current_frame * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif self.figure is not None:
            self.figure.savefig(path, dpi=160, facecolor=self.figure.get_facecolor())
        else:
            frame = self.render_frame()
            plt.imsave(path, frame)
        return path

    def export_frames(self, count: int) -> None:
        """Export numbered PNG frames for video editing or ffmpeg."""

        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            if not self.paused:
                self.simulation.step(self.speed)
            if self.auto_camera:
                self.yaw += 0.18
            frame = self.render_frame()
            path = self.args.output_dir / f"frame_{index:05d}.png"
            plt.imsave(path, frame)
            print(f"wrote {path}")
        if self.args.export_mp4:
            self.encode_mp4()

    def encode_mp4(self) -> None:
        """Encode exported PNG frames into an H.264 MP4 with ffmpeg."""

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("ffmpeg not found; PNG frames were exported but MP4 was skipped")
            return
        output = self.args.export_mp4
        if not output.is_absolute():
            output = self.args.output_dir / output
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(self.args.fps),
            "-i",
            str(self.args.output_dir / "frame_%05d.png"),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            str(self.args.crf),
            "-preset",
            "slow",
            str(output),
        ]
        subprocess.run(command, check=True)
        print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["galaxy", "planet", "star"], default="galaxy")
    parser.add_argument("--particles", type=int, default=60000)
    parser.add_argument("--resolution", type=int, default=540)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Step size. Galaxy uses Myr; planet uses minutes; star uses hours.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help='Example: "auto", "cpu", "cuda:0".',
    )
    parser.add_argument("--renderer", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--display",
        choices=["auto", "opencv", "matplotlib"],
        default="auto",
        help="Live display backend. OpenCV is much faster than Matplotlib.",
    )
    parser.add_argument("--theme", choices=sorted(THEMES), default="natural")
    parser.add_argument(
        "--extent",
        type=float,
        default=None,
        help="View half-width. Defaults depend on selected mode.",
    )
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=17.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-bloom", action="store_true", help="Disable glow for speed.")
    parser.add_argument("--no-dust", action="store_true", help="Disable dust absorption.")
    parser.add_argument("--no-gas", action="store_true", help="Disable gas bridge layer.")
    parser.add_argument(
        "--no-starburst",
        action="store_true",
        help="Disable encounter-triggered starburst layer.",
    )
    parser.add_argument("--auto-camera", action="store_true")
    parser.add_argument(
        "--impact-speed",
        type=float,
        default=None,
        help="Planet/star mode approach speed in body radii per time unit.",
    )
    parser.add_argument(
        "--impact-parameter",
        type=float,
        default=None,
        help="Planet/star mode off-center collision offset in body radii.",
    )
    parser.add_argument(
        "--body-offset",
        type=float,
        default=None,
        help="Planet/star mode initial center offset in body radii.",
    )
    parser.add_argument("--export-frames", type=int, default=0)
    parser.add_argument("--export-mp4", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=Path("renders"))
    return parser.parse_args()


def main() -> None:
    """Application entry point."""

    app = GalaxyApp(parse_args())
    app.run()
