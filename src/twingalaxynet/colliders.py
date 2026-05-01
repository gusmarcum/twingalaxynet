"""Cinematic planet and star collision simulations.

These modes use deformable tracer particles anchored to two massive bodies.
They are not full smoothed-particle hydrodynamics. The goal is an interactive,
physically motivated visual model: center-of-mass gravity, elastic restoring
forces, impact heating, debris plumes, and explicit non-physical state checks.

Relevant background:
- Benz & Asphaug (1999), catastrophic disruptions of planetesimals.
- Canup (2012), giant impact formation scenarios for the Moon.
- Lombardi et al. (2002), stellar merger hydrodynamics context.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np
import torch
from astropy import constants as const
from astropy import units as u


EARTH_RADIUS_KM = const.R_earth.to(u.km).value
SOLAR_RADIUS_KM = const.R_sun.to(u.km).value


@dataclass(frozen=True)
class ColliderConfig:
    """Configuration for planet or star collision scenes."""

    kind: str = "planet"
    particles: int = 70000
    dt: float = 0.055
    radius: float = 1.0
    center_offset: float = 2.45
    impact_parameter: float = 0.52
    approach_speed: float = 0.065
    spin_rate: float = 0.020
    gravity_strength: float = 0.0025
    spring_strength: float = 0.10
    damping: float = 0.06
    shock_strength: float = 0.46
    contact_restitution: float = 0.0
    contact_friction: float = 0.48
    contact_floor: float = 1.38
    damage_rate: float = 0.22
    debris_ejecta: float = 1.15
    heat_decay: float = 0.985
    dtype: torch.dtype = torch.float32
    seed: int = 19
    runaway_speed: float = 3.0

    @property
    def time_unit(self) -> str:
        """Human-readable time unit."""

        return "hr" if self.kind == "star" else "min"

    @property
    def distance_unit(self) -> str:
        """Human-readable distance unit."""

        return "Rsun" if self.kind == "star" else "Rearth"

    @property
    def speed_factor_km_s(self) -> float:
        """Convert internal distance/time speed into km/s."""

        if self.kind == "star":
            return SOLAR_RADIUS_KM / 3600.0
        return EARTH_RADIUS_KM / 60.0


@dataclass
class ColliderShard:
    """Particle arrays owned by one device."""

    device: torch.device
    body_id: torch.Tensor
    anchor: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    color: torch.Tensor
    luminosity: torch.Tensor
    heat: torch.Tensor
    damage: torch.Tensor


class BodyCollisionSimulation:
    """Interactive two-body planet/star collision scene."""

    def __init__(
        self,
        config: ColliderConfig,
        devices: Sequence[str] | str = "auto",
    ) -> None:
        self.config = config
        self.devices = self._select_devices(devices)
        self._validate_config()
        self.time = 0.0
        self.peak_impact = 0.0
        self.disruption_started = False
        self.center_position: torch.Tensor
        self.center_velocity: torch.Tensor
        self.shards: List[ColliderShard] = []
        self.reset()

    def reset(self) -> None:
        """Reset deterministic initial body states."""

        rng = np.random.default_rng(self.config.seed)
        primary = self.devices[0]
        dtype = self.config.dtype
        offset = self.config.center_offset
        impact = self.config.impact_parameter
        speed = self.config.approach_speed
        self.center_position = torch.tensor(
            [[-offset, -0.5 * impact, 0.0], [offset, 0.5 * impact, 0.0]],
            dtype=dtype,
            device=primary,
        )
        self.center_velocity = torch.tensor(
            [[speed, 0.010, 0.0], [-speed, -0.010, 0.0]],
            dtype=dtype,
            device=primary,
        )

        counts = self._split_counts(self.config.particles, len(self.devices))
        body_ids = np.concatenate(
            [
                np.zeros(self.config.particles // 2, dtype=np.int64),
                np.ones(
                    self.config.particles - self.config.particles // 2,
                    dtype=np.int64,
                ),
            ]
        )
        rng.shuffle(body_ids)
        start = 0
        self.shards = []
        for device, count in zip(self.devices, counts):
            ids = body_ids[start : start + count]
            anchor, color, luminosity = self._make_particles(rng, ids)
            center = np.array(
                [[-offset, -0.5 * impact, 0.0], [offset, 0.5 * impact, 0.0]],
                dtype=np.float32,
            )
            center_velocity = np.array(
                [[speed, 0.010, 0.0], [-speed, -0.010, 0.0]], dtype=np.float32
            )
            spin = self.config.spin_rate * (1.0 if self.config.kind == "planet" else 0.35)
            position = center[ids] + anchor
            velocity = center_velocity[ids] + np.cross(
                np.array([0.0, 0.0, spin], dtype=np.float32), anchor
            )
            self.shards.append(
                ColliderShard(
                    device=device,
                    body_id=torch.as_tensor(ids, dtype=torch.long, device=device),
                    anchor=torch.as_tensor(anchor, dtype=dtype, device=device),
                    position=torch.as_tensor(position, dtype=dtype, device=device),
                    velocity=torch.as_tensor(velocity, dtype=dtype, device=device),
                    color=torch.as_tensor(color, dtype=torch.float32, device=device),
                    luminosity=torch.as_tensor(
                        luminosity, dtype=torch.float32, device=device
                    ),
                    heat=torch.zeros(count, dtype=torch.float32, device=device),
                    damage=torch.zeros(count, dtype=torch.float32, device=device),
                )
            )
            start += count
        self.time = 0.0
        self.peak_impact = 0.0
        self.disruption_started = False
        self._assert_physical()

    def step(self, count: int = 1) -> None:
        """Advance the collider by ``count`` semi-implicit steps."""

        if count < 1:
            raise ValueError("step count must be positive")
        dt = self.config.dt
        for _ in range(count):
            center_acc = self._center_acceleration()
            self.center_velocity = self.center_velocity + dt * center_acc
            self.center_position = self.center_position + dt * self.center_velocity
            self._resolve_center_contact()
            shock = self.impact_strength()
            if self.config.kind == "planet":
                shock_value = float(shock.detach().to("cpu"))
                self.peak_impact = max(self.peak_impact, shock_value)
                if self.peak_impact >= 0.22:
                    self.disruption_started = True

            for shard in self.shards:
                acceleration, heat_source, damage_source = self._particle_acceleration(
                    shard,
                    shock.to(shard.device),
                )
                shard.velocity = shard.velocity + dt * acceleration
                shard.position = shard.position + dt * shard.velocity
                shard.heat = torch.clamp(
                    shard.heat * self.config.heat_decay + heat_source,
                    0.0,
                    8.0,
                )
                shard.damage = torch.clamp(
                    shard.damage + damage_source,
                    0.0,
                    1.0,
                )
            self.time += dt
        self._assert_physical()

    def impact_strength(self) -> torch.Tensor:
        """Return current overlap-driven impact strength in [0, 1]."""

        separation = torch.linalg.norm(self.center_position[1] - self.center_position[0])
        overlap = torch.clamp(2.0 * self.config.radius - separation, min=0.0)
        return torch.clamp(overlap / (1.35 * self.config.radius), 0.0, 1.0)

    def separation_display(self) -> float:
        """Return center separation in the mode's distance unit."""

        delta = self.center_position[1] - self.center_position[0]
        return float(torch.linalg.norm(delta).detach().to("cpu"))

    def relative_speed_display(self) -> float:
        """Return center relative speed in km/s."""

        delta = self.center_velocity[1] - self.center_velocity[0]
        speed = float(torch.linalg.norm(delta).detach().to("cpu"))
        return speed * self.config.speed_factor_km_s

    def _make_particles(
        self,
        rng: np.random.Generator,
        body_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = body_ids.size
        radius = self.config.radius * np.cbrt(rng.random(count))
        cos_theta = rng.uniform(-1.0, 1.0, count)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = rng.uniform(0.0, 2.0 * np.pi, count)
        anchor = np.column_stack(
            [
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                radius * cos_theta,
            ]
        ).astype(np.float32)
        norm_radius = radius / self.config.radius
        if self.config.kind == "star":
            color = self._star_colors(norm_radius, body_ids, rng)
            luminosity = (1.2 + 3.0 * np.exp(-(norm_radius / 0.42) ** 2)).astype(
                np.float32
            )
        else:
            color = self._planet_colors(norm_radius, body_ids, rng)
            core = np.exp(-(norm_radius / 0.34) ** 2)
            surface = np.clip((norm_radius - 0.70) / 0.30, 0.0, 1.0)
            luminosity = (0.20 + 0.65 * core + 0.18 * surface).astype(np.float32)
        return anchor, color, luminosity

    def _planet_colors(
        self,
        norm_radius: np.ndarray,
        body_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        iron = np.array([0.75, 0.36, 0.18], dtype=np.float32)
        mantle = np.array([0.48, 0.38, 0.30], dtype=np.float32)
        crust_a = np.array([0.23, 0.34, 0.30], dtype=np.float32)
        crust_b = np.array([0.52, 0.46, 0.38], dtype=np.float32)
        color = np.empty((norm_radius.size, 3), dtype=np.float32)
        crust_mix = (body_ids[:, None] * crust_a + (1 - body_ids[:, None]) * crust_b)
        color[:] = mantle
        color[norm_radius < 0.34] = iron
        surface = norm_radius > 0.72
        color[surface] = crust_mix[surface]
        color *= rng.uniform(0.82, 1.18, (norm_radius.size, 1)).astype(np.float32)
        return np.clip(color, 0.0, 1.0)

    def _star_colors(
        self,
        norm_radius: np.ndarray,
        body_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        core = np.array([1.00, 0.95, 0.68], dtype=np.float32)
        envelope_a = np.array([1.00, 0.43, 0.18], dtype=np.float32)
        envelope_b = np.array([0.45, 0.68, 1.00], dtype=np.float32)
        envelope = body_ids[:, None] * envelope_a + (1 - body_ids[:, None]) * envelope_b
        core_weight = np.exp(-(norm_radius[:, None] / 0.54) ** 2)
        color = envelope * (1.0 - core_weight) + core * core_weight
        color *= rng.uniform(0.88, 1.22, (norm_radius.size, 1)).astype(np.float32)
        return np.clip(color, 0.0, 1.5)

    def _center_acceleration(self) -> torch.Tensor:
        delta = self.center_position[0] - self.center_position[1]
        radius = torch.clamp(torch.linalg.norm(delta), min=0.15)
        direction = delta / radius
        gravity = self.config.gravity_strength / radius**2
        overlap = torch.clamp(2.0 * self.config.radius - radius, min=0.0)
        contact = 0.022 * overlap if self.config.kind == "planet" else 0.006 * overlap
        acc = torch.zeros_like(self.center_position)
        acc[0] = -gravity * direction + contact * direction
        acc[1] = gravity * direction - contact * direction
        return acc

    def _resolve_center_contact(self) -> None:
        """Apply an equal-mass inelastic contact impulse to overlapping bodies.

        This is not a rigid-body solver. It is a conservative correction that
        prevents planet cores from numerically passing through each other while
        still allowing deformation, heating, and debris ejection.
        """

        delta = self.center_position[1] - self.center_position[0]
        separation = torch.clamp(torch.linalg.norm(delta), min=1e-6)
        normal = delta / separation
        overlap = 2.0 * self.config.radius - separation
        if overlap <= 0.0:
            return

        contact_trigger = (
            1.62 * self.config.radius
            if self.config.kind == "planet"
            else 1.88 * self.config.radius
        )
        if separation > contact_trigger:
            return

        if self.config.kind == "planet":
            floor = self.config.contact_floor * self.config.radius
            correction = torch.clamp(floor - separation, min=0.0)
            self.center_position[0] -= 0.5 * correction * normal
            self.center_position[1] += 0.5 * correction * normal

        relative_velocity = self.center_velocity[1] - self.center_velocity[0]
        normal_speed = torch.sum(relative_velocity * normal)
        if normal_speed < 0.0:
            restitution = (
                self.config.contact_restitution
                if self.config.kind == "planet"
                else 0.0
            )
            impulse = -0.5 * (1.0 + restitution) * normal_speed * normal
            self.center_velocity[0] -= impulse
            self.center_velocity[1] += impulse

        relative_velocity = self.center_velocity[1] - self.center_velocity[0]
        tangent = relative_velocity - torch.sum(relative_velocity * normal) * normal
        friction = self.config.contact_friction if self.config.kind == "planet" else 0.18
        self.center_velocity[0] += 0.5 * friction * tangent
        self.center_velocity[1] -= 0.5 * friction * tangent

        barycentric_velocity = torch.mean(self.center_velocity, dim=0, keepdim=True)
        impact_damping = 0.18 if self.config.kind == "planet" else 0.04
        self.center_velocity = barycentric_velocity + (
            self.center_velocity - barycentric_velocity
        ) * (1.0 - impact_damping * self.impact_strength())

    def _particle_acceleration(
        self,
        shard: ColliderShard,
        shock: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = self.center_position.to(shard.device, self.config.dtype)
        velocities = self.center_velocity.to(shard.device, self.config.dtype)
        body_center = centers[shard.body_id]
        body_velocity = velocities[shard.body_id]
        damage = shard.damage.to(self.config.dtype)
        intact = torch.clamp(1.0 - damage, 0.0, 1.0)
        target = body_center + shard.anchor * (1.0 + 0.06 * shock * intact[:, None])
        if self.config.kind == "planet" and self.disruption_started:
            spring = self.config.spring_strength * intact**2
        else:
            spring = self.config.spring_strength * (0.08 + 0.92 * intact**2)
        spring = spring * (1.0 - 0.55 * shock)
        acceleration = spring[:, None] * (target - shard.position)
        acceleration += self.config.damping * intact[:, None] * (
            body_velocity - shard.velocity
        )
        center_gravity = self._center_gravity(shard.position, centers)
        if self.config.kind == "planet" and self.disruption_started:
            center_gravity = center_gravity * torch.clamp(intact[:, None] ** 2, 0.03, 1.0)
        acceleration += center_gravity

        mid = torch.mean(centers, dim=0)
        axis = centers[1] - centers[0]
        axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-5)
        from_mid = shard.position - mid
        distance = torch.linalg.norm(from_mid, dim=1)
        radial = from_mid / torch.clamp(distance[:, None], min=1e-5)
        longitudinal = torch.abs(torch.sum(from_mid * axis[None, :], dim=1))
        contact_weight = torch.exp(
            -((longitudinal / (0.95 * self.config.radius)) ** 2)
            -((distance / (2.25 * self.config.radius)) ** 2)
        )
        plume = torch.cross(
            axis.expand_as(radial),
            radial,
            dim=1,
        )
        turbulence = torch.sin(7.0 * shard.anchor[:, 0:1] + 5.0 * self.time)
        surface = torch.clamp(
            torch.linalg.norm(shard.anchor, dim=1) / self.config.radius,
            0.0,
            1.0,
        )
        weakness = 0.25 + 0.75 * surface
        if self.config.kind == "planet":
            damage_source = (
                self.config.damage_rate
                * shock
                * contact_weight.to(torch.float32)
                * weakness.to(torch.float32)
            )
            if self.disruption_started:
                damage_source = damage_source + 0.010 * weakness.to(torch.float32)
        else:
            damage_source = 0.08 * shock * contact_weight.to(torch.float32)

        debris_gate = torch.clamp(damage + damage_source.to(damage.dtype), 0.0, 1.0)
        debris_gate = debris_gate[:, None]
        shock_acc = self.config.shock_strength * shock * contact_weight[:, None]
        radial_push = 0.42 if self.config.kind == "planet" else 0.95
        swirl_push = 0.22 if self.config.kind == "planet" else 0.22
        acceleration += shock_acc * (radial_push * radial + swirl_push * turbulence * plume)
        if self.config.kind == "planet":
            disruption_boost = 1.65 if self.disruption_started else 1.0
            ejecta = (
                disruption_boost
                * self.config.debris_ejecta
                * shock
                * contact_weight[:, None]
            )
            side = torch.sign(torch.sum(from_mid * axis[None, :], dim=1, keepdim=True))
            chaos = torch.stack(
                [
                    torch.sin(13.0 * shard.anchor[:, 1] + 0.7),
                    torch.cos(17.0 * shard.anchor[:, 2] - 0.4),
                    torch.sin(11.0 * (shard.anchor[:, 0] + shard.anchor[:, 1])),
                ],
                dim=1,
            )
            chaos = torch.nn.functional.normalize(chaos, dim=1)
            sheet_direction = torch.nn.functional.normalize(
                0.50 * radial
                + 0.35 * side * axis[None, :]
                + 0.45 * turbulence * plume
                + 0.32 * chaos,
                dim=1,
            )
            acceleration += ejecta * debris_gate * sheet_direction
        if self.config.kind == "star":
            acceleration += 0.24 * shock_acc * axis[None, :] * torch.sign(
                torch.sum(from_mid * axis[None, :], dim=1, keepdim=True)
            )

        heat_source = (0.18 if self.config.kind == "planet" else 0.32) * shock
        heat_source = heat_source * contact_weight.to(torch.float32)
        return acceleration, heat_source, damage_source

    def _center_gravity(
        self,
        position: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Apply softened gravity from both body centers to tracer particles."""

        softening = 0.34 if self.config.kind == "planet" else 0.55
        delta = centers[None, :, :] - position[:, None, :]
        radius2 = torch.sum(delta * delta, dim=2) + softening**2
        inv_radius3 = torch.rsqrt(radius2) ** 3
        strength = self.config.gravity_strength * (
            0.55 if self.config.kind == "planet" else 0.38
        )
        return strength * torch.sum(delta * inv_radius3[:, :, None], dim=1)

    def _assert_physical(self) -> None:
        if not torch.isfinite(self.center_position).all():
            raise FloatingPointError("non-physical collider center position detected")
        if not torch.isfinite(self.center_velocity).all():
            raise FloatingPointError("non-physical collider center velocity detected")
        for shard in self.shards:
            if not torch.isfinite(shard.position).all():
                raise FloatingPointError("non-physical collider particle position")
            if not torch.isfinite(shard.velocity).all():
                raise FloatingPointError("non-physical collider particle velocity")
            speed = torch.linalg.norm(shard.velocity, dim=1).max()
            if speed > self.config.runaway_speed:
                raise FloatingPointError(
                    f"runaway collider particle speed: {speed.item():.3f}"
                )

    def _validate_config(self) -> None:
        if self.config.kind not in {"planet", "star"}:
            raise ValueError("collider kind must be 'planet' or 'star'")
        if self.config.particles < 1000:
            raise ValueError("particles must be at least 1000")
        if self.config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.config.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.config.approach_speed <= 0.0:
            raise ValueError("approach_speed must be positive")

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


def make_frame_from_collider(
    simulation: BodyCollisionSimulation,
    resolution: int,
    extent: float,
    yaw_deg: float,
    pitch_deg: float,
    bloom: bool = True,
) -> np.ndarray:
    """Render a planet/star collider from GPU particle buffers."""

    if resolution < 128:
        raise ValueError("resolution must be at least 128")
    if extent <= 0:
        raise ValueError("extent must be positive")
    primary = simulation.devices[0]
    flat_size = resolution * resolution
    canvas = torch.zeros((flat_size, 3), dtype=torch.float32, device=primary)
    for shard in simulation.shards:
        canvas += _render_shard(shard, simulation, resolution, extent, yaw_deg, pitch_deg).to(
            primary
        )
    image = canvas.reshape((resolution, resolution, 3))
    if simulation.config.kind == "planet":
        image = _apply_planet_heat(image)
        stretch = 0.42
        gamma = 0.72
        background = _background(resolution, primary, (0.004, 0.005, 0.010), (0.018, 0.020, 0.034))
    else:
        stretch = 0.16
        gamma = 0.58
        background = _background(resolution, primary, (0.006, 0.003, 0.010), (0.025, 0.011, 0.038))
    if bloom:
        image = image + (0.55 if simulation.config.kind == "planet" else 1.1) * _blur(
            image, passes=2
        )
    image = torch.asinh(image * stretch)
    scale = torch.quantile(image.flatten(), 0.9975)
    if not torch.isfinite(scale) or scale <= 0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=primary)
    image = torch.clamp(image / scale, 0.0, 1.0)
    image = torch.pow(image, gamma)
    image = torch.clamp(background + image, 0.0, 1.0)
    return image.detach().cpu().numpy()


def _render_shard(
    shard: ColliderShard,
    simulation: BodyCollisionSimulation,
    resolution: int,
    extent: float,
    yaw_deg: float,
    pitch_deg: float,
) -> torch.Tensor:
    position = shard.position.to(torch.float32)
    x, y, z = _project(position, yaw_deg, pitch_deg)
    scale = resolution / (2.0 * extent)
    xi = ((x + extent) * scale).to(torch.int64)
    yi = ((y + extent) * scale).to(torch.int64)
    valid = (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
    flat = yi[valid] * resolution + xi[valid]
    flat_size = resolution * resolution
    depth_weight = torch.clamp(1.10 + z / (2.2 * extent), 0.35, 1.75)
    heat = shard.heat
    if simulation.config.kind == "planet":
        heat_color = torch.tensor([1.0, 0.28, 0.04], device=shard.device)
        color = torch.clamp(shard.color + heat[:, None] * heat_color[None, :], 0.0, 4.0)
        weights = shard.luminosity * depth_weight * (1.0 + 2.4 * heat)
    else:
        heat_color = torch.tensor([0.55, 0.75, 1.0], device=shard.device)
        color = torch.clamp(shard.color + heat[:, None] * heat_color[None, :], 0.0, 5.0)
        weights = shard.luminosity * depth_weight * (1.0 + 3.8 * heat)
    channels = []
    for channel in range(3):
        channels.append(
            torch.bincount(
                flat,
                weights=weights[valid] * color[valid, channel],
                minlength=flat_size,
            )
        )
    return torch.stack(channels, dim=1).to(torch.float32)


def _project(
    position: torch.Tensor,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.tensor(np.deg2rad(yaw_deg), dtype=torch.float32, device=position.device)
    pitch = torch.tensor(np.deg2rad(pitch_deg), dtype=torch.float32, device=position.device)
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


def _apply_planet_heat(image: torch.Tensor) -> torch.Tensor:
    density = torch.mean(image, dim=2, keepdim=True)
    shadow = _blur(density, passes=1)
    scale = torch.quantile(shadow.flatten(), 0.985)
    if not torch.isfinite(scale) or scale <= 0:
        return image
    dust = torch.clamp(shadow / scale, 0.0, 1.0)
    return torch.clamp(image * (1.0 - 0.18 * dust), min=0.0)


def _blur(image: torch.Tensor, passes: int) -> torch.Tensor:
    blurred = image
    for _ in range(passes):
        blurred = (
            0.34 * blurred
            + 0.11
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
    vignette = np.clip(1.0 - 1.45 * radius, 0.0, 1.0).astype(np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    return low_arr[None, None, :] + vignette[:, :, None] * (
        high_arr - low_arr
    )[None, None, :]


def _background(
    resolution: int,
    device: torch.device,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
) -> torch.Tensor:
    return torch.as_tensor(
        _background_cpu(resolution, low, high), dtype=torch.float32, device=device
    )
