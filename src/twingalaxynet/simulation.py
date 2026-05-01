"""GPU galaxy collision dynamics.

The implementation uses softened analytic galaxy potentials and a leapfrog
integrator. This is intentionally closer to the restricted three-body galaxy
encounter experiments of Toomre & Toomre (1972, ApJ, 178, 623) than to a full
self-gravitating direct N-body code. The result is fast enough for real-time
visual exploration while retaining meaningful tidal tails and bridges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch

from twingalaxynet.constants import G_KPC3_PER_MSUN_MYR2, KM_S_TO_KPC_MYR


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a twin galaxy encounter.

    Attributes:
        particles: Total number of tracer star particles.
        dt_myr: Integrator step in megayears.
        mass_msun: Mass of each analytic galaxy potential in solar masses.
        softening_kpc: Softening length for the Plummer-like potential.
        disk_scale_kpc: Exponential disk scale length.
        disk_height_kpc: Gaussian vertical scale height.
        dtype: Torch dtype used for the dynamical state.
        seed: Random seed for reproducible initial conditions.
        runaway_speed_kpc_myr: Safety threshold for non-physical particle speeds.
    """

    particles: int = 60000
    dt_myr: float = 0.35
    mass_msun: float = 7.5e11
    softening_kpc: float = 3.5
    disk_scale_kpc: float = 4.8
    disk_height_kpc: float = 0.22
    dtype: torch.dtype = torch.float64
    seed: int = 7
    runaway_speed_kpc_myr: float = 5.0


@dataclass
class ParticleShard:
    """Particle arrays owned by one device."""

    device: torch.device
    position: torch.Tensor
    velocity: torch.Tensor
    color: torch.Tensor
    luminosity: torch.Tensor


class TwinGalaxySimulation:
    """Evolve two disk galaxies through a tidal encounter."""

    def __init__(
        self,
        config: SimulationConfig,
        devices: Sequence[str] | str = "auto",
    ) -> None:
        self.config = config
        self.devices = self._select_devices(devices)
        self._validate_config()
        self.time_myr = 0.0
        self.center_position: torch.Tensor
        self.center_velocity: torch.Tensor
        self.shards: List[ParticleShard] = []
        self.reset()

    def reset(self) -> None:
        """Reset to deterministic initial galaxy disks."""

        rng = np.random.default_rng(self.config.seed)
        primary = self.devices[0]
        dtype = self.config.dtype

        self.center_position = torch.tensor(
            [[-32.0, -5.0, 0.0], [32.0, 5.0, 3.0]],
            dtype=dtype,
            device=primary,
        )
        approach = 165.0 * KM_S_TO_KPC_MYR
        transverse = 72.0 * KM_S_TO_KPC_MYR
        self.center_velocity = torch.tensor(
            [[approach, transverse, 0.012], [-approach, -transverse, -0.012]],
            dtype=dtype,
            device=primary,
        )

        counts = self._split_counts(self.config.particles, len(self.devices))
        galaxy_ids = np.concatenate(
            [
                np.zeros(self.config.particles // 2, dtype=np.int64),
                np.ones(self.config.particles - self.config.particles // 2, dtype=np.int64),
            ]
        )
        rng.shuffle(galaxy_ids)

        start = 0
        self.shards = []
        for device, count in zip(self.devices, counts):
            ids = galaxy_ids[start : start + count]
            position, velocity, color, luminosity = self._make_particles(rng, ids)
            self.shards.append(
                ParticleShard(
                    device=device,
                    position=torch.as_tensor(position, dtype=dtype, device=device),
                    velocity=torch.as_tensor(velocity, dtype=dtype, device=device),
                    color=torch.as_tensor(color, dtype=torch.float32, device=device),
                    luminosity=torch.as_tensor(
                        luminosity, dtype=torch.float32, device=device
                    ),
                )
            )
            start += count

        self.time_myr = 0.0
        self._assert_physical()

    def step(self, count: int = 1) -> None:
        """Advance the simulation by ``count`` leapfrog steps."""

        if count < 1:
            raise ValueError("step count must be positive")

        dt = self.config.dt_myr
        for _ in range(count):
            center_acc = self._center_acceleration()
            self.center_velocity = self.center_velocity + 0.5 * dt * center_acc
            self.center_position = self.center_position + dt * self.center_velocity
            center_acc = self._center_acceleration()
            self.center_velocity = self.center_velocity + 0.5 * dt * center_acc

            for shard in self.shards:
                acceleration = self._particle_acceleration(shard)
                shard.velocity = shard.velocity + 0.5 * dt * acceleration
                shard.position = shard.position + dt * shard.velocity
                acceleration = self._particle_acceleration(shard)
                shard.velocity = shard.velocity + 0.5 * dt * acceleration

            self.time_myr += dt

        self._assert_physical()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CPU arrays for rendering.

        Returns:
            Positions in kpc, RGB colors, and per-particle luminosities.
        """

        positions = []
        colors = []
        luminosities = []
        for shard in self.shards:
            positions.append(shard.position.detach().to("cpu", torch.float32).numpy())
            colors.append(shard.color.detach().to("cpu").numpy())
            luminosities.append(shard.luminosity.detach().to("cpu").numpy())
        return (
            np.concatenate(positions, axis=0),
            np.concatenate(colors, axis=0),
            np.concatenate(luminosities, axis=0),
        )

    def center_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return galaxy center positions and velocities on CPU."""

        return (
            self.center_position.detach().to("cpu", torch.float64).numpy(),
            self.center_velocity.detach().to("cpu", torch.float64).numpy(),
        )

    def separation_kpc(self) -> float:
        """Return current galaxy-center separation in kiloparsecs."""

        delta = self.center_position[1] - self.center_position[0]
        return float(torch.linalg.norm(delta).detach().to("cpu"))

    def relative_speed_km_s(self) -> float:
        """Return galaxy-center relative speed in km/s."""

        delta = self.center_velocity[1] - self.center_velocity[0]
        speed_kpc_myr = float(torch.linalg.norm(delta).detach().to("cpu"))
        return speed_kpc_myr / KM_S_TO_KPC_MYR

    def _make_particles(
        self,
        rng: np.random.Generator,
        galaxy_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = galaxy_ids.size
        radius = rng.gamma(shape=2.0, scale=self.config.disk_scale_kpc, size=count)
        radius = np.clip(radius, 0.08, 24.0)
        theta = rng.uniform(0.0, 2.0 * np.pi, count)
        z = rng.normal(0.0, self.config.disk_height_kpc, count)

        local = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), z]
        )
        tangent = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(count)])

        enclosed = self.config.mass_msun * radius**3 / (
            radius**2 + self.config.softening_kpc**2
        ) ** 1.5
        circular_speed = np.sqrt(G_KPC3_PER_MSUN_MYR2 * enclosed / radius)
        circular_speed *= rng.normal(1.0, 0.035, count)
        local_velocity = tangent * circular_speed[:, None]
        local_velocity += rng.normal(0.0, 0.012, (count, 3))

        centers = np.array([[-32.0, -5.0, 0.0], [32.0, 5.0, 3.0]])
        center_velocities = np.array(
            [
                [165.0 * KM_S_TO_KPC_MYR, 72.0 * KM_S_TO_KPC_MYR, 0.012],
                [-165.0 * KM_S_TO_KPC_MYR, -72.0 * KM_S_TO_KPC_MYR, -0.012],
            ]
        )
        rotations = [
            _rotation_matrix(np.deg2rad(63.0), np.deg2rad(-18.0)),
            _rotation_matrix(np.deg2rad(-58.0), np.deg2rad(31.0)),
        ]

        position = np.empty((count, 3), dtype=np.float64)
        velocity = np.empty((count, 3), dtype=np.float64)
        for galaxy_id in (0, 1):
            mask = galaxy_ids == galaxy_id
            rotated = local[mask] @ rotations[galaxy_id].T
            rotated_velocity = local_velocity[mask] @ rotations[galaxy_id].T
            position[mask] = rotated + centers[galaxy_id]
            velocity[mask] = rotated_velocity + center_velocities[galaxy_id]

        young = rng.random(count) < np.exp(-radius / 8.0) * 0.42
        color = np.empty((count, 3), dtype=np.float32)
        color[young] = np.array([0.55, 0.72, 1.0], dtype=np.float32)
        color[~young] = np.array([1.0, 0.73, 0.42], dtype=np.float32)
        color *= rng.uniform(0.75, 1.25, (count, 1)).astype(np.float32)
        color = np.clip(color, 0.0, 1.0)

        spiral = 0.5 + 0.5 * np.cos(2.5 * theta - radius * 0.45)
        luminosity = (0.45 + 1.7 * young + 0.75 * spiral).astype(np.float32)
        luminosity *= rng.lognormal(mean=-0.1, sigma=0.45, size=count).astype(np.float32)
        return position, velocity, color, luminosity

    def _particle_acceleration(self, shard: ParticleShard) -> torch.Tensor:
        centers = self.center_position.to(shard.device)
        masses = torch.full(
            (2,),
            self.config.mass_msun,
            dtype=self.config.dtype,
            device=shard.device,
        )
        delta = shard.position[:, None, :] - centers[None, :, :]
        radius2 = torch.sum(delta * delta, dim=2) + self.config.softening_kpc**2
        inv_radius3 = torch.rsqrt(radius2) ** 3
        weighted = masses[None, :, None] * delta * inv_radius3[:, :, None]
        return -G_KPC3_PER_MSUN_MYR2 * torch.sum(weighted, dim=1)

    def _center_acceleration(self) -> torch.Tensor:
        delta = self.center_position[:, None, :] - self.center_position[None, :, :]
        radius2 = torch.sum(delta * delta, dim=2) + self.config.softening_kpc**2
        inv_radius3 = torch.rsqrt(radius2) ** 3
        inv_radius3.fill_diagonal_(0.0)
        masses = torch.full(
            (2,),
            self.config.mass_msun,
            dtype=self.config.dtype,
            device=self.center_position.device,
        )
        weighted = masses[None, :, None] * delta * inv_radius3[:, :, None]
        return -G_KPC3_PER_MSUN_MYR2 * torch.sum(weighted, dim=1)

    def _assert_physical(self) -> None:
        if not torch.isfinite(self.center_position).all():
            raise FloatingPointError("non-physical center position detected")
        if not torch.isfinite(self.center_velocity).all():
            raise FloatingPointError("non-physical center velocity detected")

        for shard in self.shards:
            if not torch.isfinite(shard.position).all():
                raise FloatingPointError("non-physical particle position detected")
            if not torch.isfinite(shard.velocity).all():
                raise FloatingPointError("non-physical particle velocity detected")
            speed = torch.linalg.norm(shard.velocity, dim=1).max()
            if speed > self.config.runaway_speed_kpc_myr:
                raise FloatingPointError(
                    "runaway particle speed detected: "
                    f"{speed.item():.3f} kpc/Myr"
                )

    def _validate_config(self) -> None:
        if self.config.particles < 1000:
            raise ValueError("particles must be at least 1000 for a galaxy image")
        if self.config.dt_myr <= 0.0:
            raise ValueError("dt_myr must be positive")
        if self.config.mass_msun <= 0.0:
            raise ValueError("mass_msun must be positive")
        if self.config.softening_kpc <= 0.0:
            raise ValueError("softening_kpc must be positive")
        if self.config.disk_scale_kpc <= 0.0:
            raise ValueError("disk_scale_kpc must be positive")

    @staticmethod
    def _split_counts(total: int, parts: int) -> List[int]:
        base = total // parts
        counts = [base] * parts
        for index in range(total - base * parts):
            counts[index] += 1
        return counts

    @staticmethod
    def _select_devices(devices: Sequence[str] | str) -> List[torch.device]:
        if devices == "auto":
            if torch.cuda.is_available():
                return [
                    torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())
                ]
            return [torch.device("cpu")]
        if isinstance(devices, str):
            selected: Iterable[str] = [item.strip() for item in devices.split(",")]
        else:
            selected = devices
        parsed = [torch.device(item) for item in selected if item]
        if not parsed:
            raise ValueError("at least one device is required")
        return parsed


def _rotation_matrix(inclination: float, twist: float) -> np.ndarray:
    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)
    cos_t = np.cos(twist)
    sin_t = np.sin(twist)
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]]
    )
    rot_z = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]
    )
    return rot_z @ rot_x
