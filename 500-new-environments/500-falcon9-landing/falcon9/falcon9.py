"""Gymnasium-compatible Falcon 9 Box2D platform landing environment.

This module is intentionally self-contained. It includes its own Box2D physics,
Gymnasium API, reward logic, and Pygame renderer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import EzPickle


class DependencyNotInstalled(ImportError):
    """Raised when an optional rendering or physics dependency is missing."""


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
    )
except ImportError as exc:  # pragma: no cover - import-time dependency check
    raise DependencyNotInstalled(
        "Box2D is not installed. Install it with `pip install swig box2d-py`."
    ) from exc

if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0

VIEWPORT_W = 600
VIEWPORT_H = 400

BOTTOM_ENGINE_POWER = 18.0
TOP_ENGINE_POWER = 1.25
INITIAL_RANDOM = 700.0

BOOSTER_HALF_WIDTH = 8
BOOSTER_TOP = 52
BOOSTER_BOTTOM = 60
BOOSTER_START_CLEARANCE = 0.06
BOOSTER_START_MIN_ANGLE = 0.0
BOOSTER_START_MAX_ANGLE = math.radians(45.0)
BOOSTER_BODY_DENSITY = 4.055244755244756
TOP_ENGINE_Y = 43
TOP_ENGINE_AWAY = 9
BOTTOM_ENGINE_Y = 54
LANDING_LEG_TOP_X = 5
LANDING_LEG_TOP_Y = 49
LANDING_LEG_FOOT_X = 13
LANDING_LEG_FOOT_Y = 60
LANDING_LEG_WIDTH = 3
LANDING_LEG_DENSITY = 6.0
LANDING_FOOT_CONTACT_SLOP = 2.0 / SCALE
CRASH_EFFECT_FRAMES = 36

PLATFORM_WIDTH = 118
PLATFORM_HEIGHT = 14
PLATFORM_HALF_X = (PLATFORM_WIDTH / 2) / (VIEWPORT_W / 2)
BOOSTER_BOTTOM_X_OFFSET = BOOSTER_BOTTOM / (VIEWPORT_W / 2)
PLATFORM_SPEED = 1.15 / 3.0
MAX_JET_FIRES = 50
DEFAULT_WIND_POWER = 5.0
DEFAULT_SUCCESS_REWARD = 100.0
DEFAULT_FAILURE_REWARD = -100.0
DEFAULT_SHAPING_FACTOR = 1.0
STABLE_LANDING_STEPS = 20
STABLE_LANDING_Y = 0.08
STABLE_LANDING_VX = 0.80
STABLE_LANDING_VY = 0.35
STABLE_LANDING_ANGLE = math.radians(8.0)
STABLE_LANDING_ANGULAR_V = 0.80
SAFE_IMPACT_VX = 1.25
SAFE_IMPACT_VY = 2.00
SAFE_IMPACT_ANGULAR_V = 1.50

ACTION_NAMES = {
    0: "no-op",
    1: "upper-left",
    2: "bottom",
    3: "upper-right",
}


def sample_booster_start_angle(rng) -> float:
    angle = float(rng.uniform(BOOSTER_START_MIN_ANGLE, BOOSTER_START_MAX_ANGLE))
    sign = float(rng.choice([-1.0, 1.0]))
    return sign * angle


class ContactDetector(contactListener):
    """Track booster contacts with the platform and ocean."""

    def __init__(self, env: "Falcon9") -> None:
        contactListener.__init__(self)
        self.env = env

    @staticmethod
    def _data(contact) -> tuple[object, object]:
        return contact.fixtureA.userData, contact.fixtureB.userData

    def BeginContact(self, contact) -> None:
        a, b = self._data(contact)
        labels = {a, b}
        booster_labels = {"booster_body", "left_foot", "right_foot"}

        if "ocean" in labels and labels.intersection(booster_labels):
            self.env.ocean_contact = True
            return

        if "platform" not in labels:
            return

        if labels.intersection(booster_labels) and not self.env.platform_contact:
            booster_body = contact.fixtureA.body if a in booster_labels else contact.fixtureB.body
            platform_body = contact.fixtureA.body if a == "platform" else contact.fixtureB.body
            self.env.platform_impact_velocity = (
                float(booster_body.linearVelocity.x - platform_body.linearVelocity.x),
                float(booster_body.linearVelocity.y - platform_body.linearVelocity.y),
                float(booster_body.angularVelocity),
            )

        if "booster_body" in labels:
            self.env.body_platform_contact = True
            self.env.platform_contact = True
        if "left_foot" in labels:
            self.env.left_foot_contact = True
            self.env.platform_contact = True
            self.env._queue_leg_anchor("left_foot", contact)
        if "right_foot" in labels:
            self.env.right_foot_contact = True
            self.env.platform_contact = True
            self.env._queue_leg_anchor("right_foot", contact)

    def EndContact(self, contact) -> None:
        a, b = self._data(contact)
        labels = {a, b}
        if "platform" not in labels:
            return
        if "left_foot" in labels and not self.env._is_leg_anchor_active_or_pending("left_foot"):
            self.env.left_foot_contact = False
        if "right_foot" in labels and not self.env._is_leg_anchor_active_or_pending("right_foot"):
            self.env.right_foot_contact = False
        if "booster_body" in labels:
            self.env.body_platform_contact = False
        self.env.platform_contact = (
            self.env.left_foot_contact
            or self.env.right_foot_contact
            or self.env.body_platform_contact
        )
        if not self.env.platform_contact:
            self.env.platform_impact_velocity = None


class Falcon9(gym.Env, EzPickle):
    """Land a reusable booster vertically on a moving ocean platform.

    Actions are ``Discrete(4)`` by default:

    - 0: do nothing
    - 1: fire the upper-left attitude jet
    - 2: fire the bottom engine
    - 3: fire the upper-right attitude jet

    With ``continuous=True``, actions are ``Box(-1, 1, shape=(2,))`` where the
    first value controls the bottom engine and the second controls the top jets.

    Observations contain eleven float values:
    relative x/y to the platform landing point, x/y velocity, booster angle,
    angular velocity, left/right foot contact flags, platform x, and platform
    velocity, and the fraction of jet fires remaining. Wind is applied from
    ``wind_direction`` with horizontal force ``wind_power``. If
    ``variable_wind=True`` the force varies over time using LunarLander v3's
    deterministic wind pattern.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = DEFAULT_WIND_POWER,
        wind_direction: float | tuple[float, float] = 0.0,
        variable_wind: bool = True,
        platform_speed: float = PLATFORM_SPEED,
        max_jet_fires: int = MAX_JET_FIRES,
        success_reward: float = DEFAULT_SUCCESS_REWARD,
        failure_reward: float = DEFAULT_FAILURE_REWARD,
        shaping_factor: float = DEFAULT_SHAPING_FACTOR,
        dashboard: bool = False,
        max_episode_steps: int | None = 1200,
    ) -> None:
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            wind_direction,
            variable_wind,
            platform_speed,
            max_jet_fires,
            success_reward,
            failure_reward,
            shaping_factor,
            dashboard,
            max_episode_steps,
        )
        if not -12.0 < gravity < 0.0:
            raise ValueError(f"gravity must be between -12 and 0, got {gravity}")
        if max_jet_fires < 1:
            raise ValueError(f"max_jet_fires must be at least 1, got {max_jet_fires}")
        if shaping_factor < 0.0:
            raise ValueError(f"shaping_factor must be nonnegative, got {shaping_factor}")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be positive or None, got {max_episode_steps}")

        self.gravity = gravity
        self.continuous = continuous
        self.enable_wind = enable_wind
        self.wind_power = float(wind_power)
        self.wind_direction = wind_direction
        self.variable_wind = variable_wind
        self.platform_speed = float(platform_speed)
        self.max_jet_fires = int(max_jet_fires)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.shaping_factor = float(shaping_factor)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.dashboard = bool(dashboard)
        self.run_count = 0
        self.episode_step = 0
        self.episode_return = 0.0
        self.last_action = None
        self.action_counts = {"upper-left": 0, "bottom": 0, "upper-right": 0}
        self.last_info = {}

        self.screen: pygame.Surface | None = None
        self.clock = None
        self.isopen = True

        self.world = Box2D.b2World(gravity=(0, gravity))
        self.booster: Box2D.b2Body | None = None
        self.platform: Box2D.b2Body | None = None
        self.ocean: Box2D.b2Body | None = None
        self.bottom_flame_power = 0.0
        self.top_flame_power = 0.0
        self.top_flame_direction = 0
        self.crash_effect_position: tuple[float, float] | None = None
        self.crash_effect_frame = 0
        self.crash_effect_frames = 0
        self.jet_fires_used = 0
        self.stable_landing_steps = 0
        self.anchored_feet: set[str] = set()
        self.pending_leg_anchor_points: dict[str, tuple[float, float]] = {}
        self.leg_anchor_joints: dict[str, object] = {}
        self.prev_shaping = None

        low = np.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, 0.0, 0.0, -1.0, -2.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [2.5, 2.5, 10.0, 10.0, 2 * math.pi, 10.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = (
            spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            if continuous
            else spaces.Discrete(4)
        )

    def _destroy(self) -> None:
        self.world.contactListener = None
        for body_name in ("booster", "platform", "ocean"):
            body = getattr(self, body_name)
            if body is not None:
                self.world.DestroyBody(body)
                setattr(self, body_name, None)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.ocean_contact = False
        self.platform_contact = False
        self.body_platform_contact = False
        self.left_foot_contact = False
        self.right_foot_contact = False
        self.platform_impact_velocity: tuple[float, float, float] | None = None
        self.failure_reason: str | None = None
        self.prev_shaping = None
        self.bottom_flame_power = 0.0
        self.top_flame_power = 0.0
        self.top_flame_direction = 0
        self.crash_effect_position = None
        self.crash_effect_frame = 0
        self.crash_effect_frames = 0
        self.jet_fires_used = 0
        self.stable_landing_steps = 0
        self.anchored_feet = set()
        self.pending_leg_anchor_points = {}
        self.leg_anchor_joints = {}

        w = VIEWPORT_W / SCALE
        h = VIEWPORT_H / SCALE
        self.platform_y = h / 4
        self.platform_half_width = PLATFORM_WIDTH / SCALE / 2
        self.platform_half_height = PLATFORM_HEIGHT / SCALE / 2
        self.platform_min_x = self.platform_half_width + 0.25
        self.platform_max_x = w - self.platform_half_width - 0.25
        self.platform_direction = int(self.np_random.choice([-1, 1]))

        platform_x = float(self.np_random.uniform(self.platform_min_x, self.platform_max_x))
        self.platform = self.world.CreateKinematicBody(position=(platform_x, self.platform_y))
        platform_fixture = self.platform.CreateFixture(
            fixtureDef(
                shape=polygonShape(box=(self.platform_half_width, self.platform_half_height)),
                density=0.0,
                friction=0.9,
                restitution=0.0,
                categoryBits=0x0001,
            )
        )
        platform_fixture.userData = "platform"
        self.platform.color1 = (38, 42, 48)
        self.platform.color2 = (235, 235, 235)

        self.ocean_y = self.platform_y - self.platform_half_height * 0.75
        self.ocean = self.world.CreateStaticBody()
        ocean_fixture = self.ocean.CreateFixture(
            fixtureDef(
                shape=edgeShape(vertices=[(0, self.ocean_y), (w, self.ocean_y)]),
                isSensor=True,
                categoryBits=0x0001,
            )
        )
        ocean_fixture.userData = "ocean"

        initial_x = platform_x
        initial_y = h + BOOSTER_BOTTOM / SCALE + BOOSTER_START_CLEARANCE
        self.booster = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=sample_booster_start_angle(self.np_random),
        )
        self.booster.bullet = True
        body_fixture = self.booster.CreateFixture(
            fixtureDef(
                shape=polygonShape(
                    vertices=[
                        (-BOOSTER_HALF_WIDTH / SCALE, -48 / SCALE),
                        (-BOOSTER_HALF_WIDTH / SCALE, 43 / SCALE),
                        (-5 / SCALE, BOOSTER_TOP / SCALE),
                        (5 / SCALE, BOOSTER_TOP / SCALE),
                        (BOOSTER_HALF_WIDTH / SCALE, 43 / SCALE),
                        (BOOSTER_HALF_WIDTH / SCALE, -48 / SCALE),
                    ]
                ),
                density=BOOSTER_BODY_DENSITY,
                friction=0.25,
                categoryBits=0x0010,
                maskBits=0x0001,
                restitution=0.0,
            )
        )
        body_fixture.userData = "booster_body"

        left_foot_fixture = self.booster.CreateFixture(
            fixtureDef(
                shape=polygonShape(
                    vertices=[
                        (-(LANDING_LEG_TOP_X + LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_TOP_Y / SCALE),
                        (-(LANDING_LEG_TOP_X - LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_TOP_Y / SCALE),
                        (-(LANDING_LEG_FOOT_X - LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_FOOT_Y / SCALE),
                        (-(LANDING_LEG_FOOT_X + LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_FOOT_Y / SCALE),
                    ]
                ),
                density=LANDING_LEG_DENSITY,
                friction=1.0,
                categoryBits=0x0010,
                maskBits=0x0001,
                restitution=0.0,
            )
        )
        left_foot_fixture.userData = "left_foot"

        right_foot_fixture = self.booster.CreateFixture(
            fixtureDef(
                shape=polygonShape(
                    vertices=[
                        ((LANDING_LEG_TOP_X - LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_TOP_Y / SCALE),
                        ((LANDING_LEG_TOP_X + LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_TOP_Y / SCALE),
                        ((LANDING_LEG_FOOT_X + LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_FOOT_Y / SCALE),
                        ((LANDING_LEG_FOOT_X - LANDING_LEG_WIDTH) / SCALE, -LANDING_LEG_FOOT_Y / SCALE),
                    ]
                ),
                density=LANDING_LEG_DENSITY,
                friction=1.0,
                categoryBits=0x0010,
                maskBits=0x0001,
                restitution=0.0,
            )
        )
        right_foot_fixture.userData = "right_foot"

        self.booster.color1 = (230, 232, 235)
        self.booster.color2 = (40, 44, 52)
        self.booster.linearVelocity = (0.0, 0.0)
        self.booster.angularVelocity = 0.0
        self.platform.linearVelocity = (self.platform_speed * self.platform_direction, 0.0)

        if self.enable_wind:
            self.wind_idx = int(self.np_random.integers(-9999, 9999))

        self.drawlist = [self.platform, self.booster]

        self.run_count += 1
        self.episode_step = 0
        self.episode_return = 0.0
        self.last_action = None
        self.action_counts = {"upper-left": 0, "bottom": 0, "upper-right": 0}
        self.last_info = {}

        if self.render_mode == "human":
            self.render()
        return self._get_state(), {}

    def _wind_unit(self) -> tuple[float, float]:
        if isinstance(self.wind_direction, tuple):
            x = float(self.wind_direction[0])
        else:
            x = math.cos(float(self.wind_direction))
        if x == 0:
            return 0.0, 0.0
        return math.copysign(1.0, x), 0.0

    def _wind_scale(self) -> float:
        if not self.variable_wind:
            return 1.0
        scale = math.tanh(
            math.sin(0.02 * self.wind_idx)
            + math.sin(math.pi * 0.01 * self.wind_idx)
        )
        self.wind_idx += 1
        return scale

    def set_wind(
        self,
        *,
        power: float | None = None,
        direction: float | tuple[float, float] | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Adjust wind during an episode."""

        if power is not None:
            self.wind_power = float(power)
        if direction is not None:
            self.wind_direction = direction
        if enabled is not None:
            self.enable_wind = bool(enabled)

    def _update_platform(self) -> None:
        assert self.platform is not None
        x = self.platform.position.x
        if x <= self.platform_min_x + 1e-6 and self.platform_direction < 0:
            self.platform_direction = 1
        elif x >= self.platform_max_x - 1e-6 and self.platform_direction > 0:
            self.platform_direction = -1
        self.platform.linearVelocity = (self.platform_speed * self.platform_direction, 0.0)

    def _clamp_platform(self) -> None:
        assert self.platform is not None
        x = min(max(self.platform.position.x, self.platform_min_x), self.platform_max_x)
        if x != self.platform.position.x:
            if x <= self.platform_min_x + 1e-6:
                self.platform_direction = 1
            elif x >= self.platform_max_x - 1e-6:
                self.platform_direction = -1
            self.platform.position = (x, self.platform.position.y)
            self.platform.linearVelocity = (self.platform_speed * self.platform_direction, 0.0)

    def _apply_wind(self) -> None:
        assert self.booster is not None
        if not self.enable_wind or self.platform_contact:
            return
        wx, wy = self._wind_unit()
        wind_mag = self.wind_power * self._wind_scale()
        self.booster.ApplyForceToCenter((wx * wind_mag, wy * wind_mag), True)

    def _apply_engines(self, action) -> tuple[float, float]:
        assert self.booster is not None

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            if not self.action_space.contains(action):
                raise AssertionError(f"{action!r} ({type(action)}) is not a valid action")

        tip = (math.sin(self.booster.angle), math.cos(self.booster.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, 1.0) / SCALE for _ in range(2)]

        bottom_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            if self.jet_fires_used < self.max_jet_fires:
                bottom_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5 if self.continuous else 1.0
                self.jet_fires_used += 1
                ox = -tip[0] * (BOTTOM_ENGINE_Y / SCALE + dispersion[0]) + side[0] * dispersion[1]
                oy = -tip[1] * (BOTTOM_ENGINE_Y / SCALE + dispersion[0]) + side[1] * dispersion[1]
                impulse_pos = (self.booster.position[0] + ox, self.booster.position[1] + oy)
                impulse = (tip[0] * BOTTOM_ENGINE_POWER * bottom_power, tip[1] * BOTTOM_ENGINE_POWER * bottom_power)
                self.booster.ApplyLinearImpulse(impulse, impulse_pos, True)

        top_power = 0.0
        top_direction = 0
        if self.continuous and abs(action[1]) > 0.5:
            top_direction = -1 if action[1] < 0 else 1
            top_power = float(np.clip(abs(action[1]), 0.5, 1.0))
        elif not self.continuous and action in (1, 3):
            top_direction = -1 if action == 1 else 1
            top_power = 1.0

        if top_direction and self.jet_fires_used < self.max_jet_fires:
            # top_direction < 0 fires the upper-left jet; > 0 fires the upper-right jet.
            self.jet_fires_used += 1
            local_side = TOP_ENGINE_AWAY / SCALE * top_direction
            impulse_sign = top_direction
            ox = tip[0] * (TOP_ENGINE_Y / SCALE + dispersion[0]) + side[0] * local_side
            oy = tip[1] * (TOP_ENGINE_Y / SCALE + dispersion[0]) + side[1] * local_side
            impulse_pos = (self.booster.position[0] + ox, self.booster.position[1] + oy)
            impulse = (
                side[0] * impulse_sign * TOP_ENGINE_POWER * top_power,
                side[1] * impulse_sign * TOP_ENGINE_POWER * top_power,
            )
            self.booster.ApplyLinearImpulse(impulse, impulse_pos, True)
        elif top_direction:
            top_power = 0.0

        self.bottom_flame_power = max(bottom_power, self.bottom_flame_power * 0.68)
        if top_direction:
            self.top_flame_direction = top_direction
        self.top_flame_power = max(top_power, self.top_flame_power * 0.68)

        return float(bottom_power), float(top_power)

    def _get_state(self) -> np.ndarray:
        assert self.booster is not None
        assert self.platform is not None

        pos = self.booster.position
        vel = self.booster.linearVelocity
        platform_pos = self.platform.position
        platform_vel = self.platform.linearVelocity
        target_y = self.platform_y + self.platform_half_height + BOOSTER_BOTTOM / SCALE
        half_w = VIEWPORT_W / SCALE / 2
        half_h = VIEWPORT_H / SCALE / 2

        state = np.array(
            [
                (pos.x - platform_pos.x) / half_w,
                (pos.y - target_y) / half_h,
                (vel.x - platform_vel.x) * half_w / FPS,
                vel.y * half_h / FPS,
                self._normalized_angle(self.booster.angle),
                20.0 * self.booster.angularVelocity / FPS,
                1.0 if self.left_foot_contact else 0.0,
                1.0 if self.right_foot_contact else 0.0,
                (platform_pos.x - half_w) / half_w,
                platform_vel.x,
                max(0.0, (self.max_jet_fires - self.jet_fires_used) / self.max_jet_fires),
            ],
            dtype=np.float32,
        )
        return state

    @staticmethod
    def _normalized_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _foot_anchor_world(self, foot_label: str) -> tuple[float, float]:
        assert self.booster is not None
        foot_sign = -1.0 if foot_label == "left_foot" else 1.0
        anchor = self.booster.GetWorldPoint(
            (
                foot_sign * LANDING_LEG_FOOT_X / SCALE,
                -LANDING_LEG_FOOT_Y / SCALE,
            )
        )
        return float(anchor.x), float(anchor.y)

    def _platform_top_y(self) -> float:
        return float(self.platform_y + self.platform_half_height)

    def _platform_surface_anchor(self, anchor: tuple[float, float]) -> tuple[float, float]:
        return float(anchor[0]), self._platform_top_y()

    def _lift_booster_until_foot_is_on_platform(self, foot_label: str) -> None:
        assert self.booster is not None
        _foot_x, foot_y = self._foot_anchor_world(foot_label)
        top_y = self._platform_top_y()
        if foot_y < top_y:
            self.booster.position = (self.booster.position.x, self.booster.position.y + top_y - foot_y)

    def _is_leg_anchor_active_or_pending(self, foot_label: str) -> bool:
        return foot_label in self.anchored_feet or foot_label in self.pending_leg_anchor_points

    def _foot_is_visually_touching_platform(self, foot_label: str) -> bool:
        assert self.platform is not None
        foot_x, foot_y = self._foot_anchor_world(foot_label)
        platform_x = float(self.platform.position.x)
        top_y = self._platform_top_y()
        return bool(
            platform_x - self.platform_half_width - LANDING_FOOT_CONTACT_SLOP
            <= foot_x
            <= platform_x + self.platform_half_width + LANDING_FOOT_CONTACT_SLOP
            and top_y - LANDING_FOOT_CONTACT_SLOP
            <= foot_y
            <= top_y + LANDING_FOOT_CONTACT_SLOP
        )

    def _queue_feet_visually_touching_platform(self) -> None:
        if self.booster is None or self.platform is None:
            return
        for foot_label in ("left_foot", "right_foot"):
            if self._is_leg_anchor_active_or_pending(foot_label):
                continue
            if self._foot_is_visually_touching_platform(foot_label):
                self._queue_leg_anchor(foot_label)

    def _queue_leg_anchor(self, foot_label: str, contact=None) -> None:
        if foot_label in self.anchored_feet:
            return
        point: tuple[float, float] | None = None
        if contact is not None:
            try:
                world_manifold = contact.worldManifold
                if world_manifold.points:
                    contact_point = world_manifold.points[0]
                    point = (float(contact_point[0]), float(contact_point[1]))
            except Exception:
                point = None
        self.pending_leg_anchor_points[foot_label] = self._platform_surface_anchor(point or self._foot_anchor_world(foot_label))

    def _process_pending_leg_anchors(self) -> None:
        if self.booster is None or self.platform is None:
            self.pending_leg_anchor_points.clear()
            return
        for foot_label, anchor in list(self.pending_leg_anchor_points.items()):
            if foot_label in self.anchored_feet:
                continue
            self._lift_booster_until_foot_is_on_platform(foot_label)
            joint = self.world.CreateRevoluteJoint(
                bodyA=self.platform,
                bodyB=self.booster,
                anchor=anchor,
                collideConnected=True,
            )
            self.leg_anchor_joints[foot_label] = joint
            self.anchored_feet.add(foot_label)
            if foot_label == "left_foot":
                self.left_foot_contact = True
            else:
                self.right_foot_contact = True
            self.platform_contact = True
        self.pending_leg_anchor_points.clear()

    def _is_settled_on_platform(self) -> bool:
        return bool(self.left_foot_contact and self.right_foot_contact and self.platform_contact)

    def _is_stable_platform_landing(self, state: np.ndarray) -> bool:
        assert self.booster is not None
        assert self.platform is not None

        bottom_x = state[0] - math.sin(float(state[4])) * BOOSTER_BOTTOM_X_OFFSET
        rel_vx = float(self.booster.linearVelocity.x - self.platform.linearVelocity.x)
        vy = float(self.booster.linearVelocity.y)
        angular_v = float(self.booster.angularVelocity)
        return bool(
            self._is_settled_on_platform()
            and not self.body_platform_contact
            and abs(bottom_x) <= PLATFORM_HALF_X
            and abs(state[1]) <= STABLE_LANDING_Y
            and abs(rel_vx) <= STABLE_LANDING_VX
            and abs(vy) <= STABLE_LANDING_VY
            and abs(state[4]) <= STABLE_LANDING_ANGLE
            and abs(angular_v) <= STABLE_LANDING_ANGULAR_V
        )

    def _update_landing_stability(self, state: np.ndarray) -> None:
        if self._is_stable_platform_landing(state):
            self.stable_landing_steps += 1
        else:
            self.stable_landing_steps = 0

    def _is_hard_platform_impact(self) -> bool:
        if self.platform_impact_velocity is None:
            return False

        rel_vx, rel_vy, angular_v = self.platform_impact_velocity
        return bool(
            abs(rel_vx) > SAFE_IMPACT_VX
            or rel_vy < -SAFE_IMPACT_VY
            or abs(angular_v) > SAFE_IMPACT_ANGULAR_V
        )

    def _terminal_status(self, state: np.ndarray) -> tuple[bool, bool, str | None]:
        assert self.booster is not None
        if self.ocean_contact or self.booster.position.y < self.ocean_y - 0.5:
            return True, False, "ocean"
        if abs(state[0]) > 2.0:
            return True, False, "out_of_bounds"
        if self.body_platform_contact:
            return True, False, "booster_body_hit_platform"
        if self.platform_contact and self._is_hard_platform_impact():
            return True, False, "hard_platform_impact"
        if self.stable_landing_steps >= STABLE_LANDING_STEPS:
            return True, True, None
        if not self.booster.awake:
            if self._is_stable_platform_landing(state):
                return True, True, None
            if self._is_settled_on_platform():
                return True, False, "unstable_landing"
            return True, False, "settled_not_on_platform"
        return False, False, None

    def _settle_successful_landing(self) -> None:
        assert self.booster is not None
        assert self.platform is not None
        target_y = self.platform_y + self.platform_half_height + BOOSTER_BOTTOM / SCALE
        platform_x = self.platform.position.x
        half_width = self.platform_half_width - 0.15
        x = float(np.clip(self.booster.position.x, platform_x - half_width, platform_x + half_width))

        self.booster.position = (x, target_y)
        self.booster.angle = 0.0
        self.booster.linearVelocity = self.platform.linearVelocity
        self.booster.angularVelocity = 0.0
        self.booster.awake = False
        self.left_foot_contact = True
        self.right_foot_contact = True
        self.body_platform_contact = False
        self.platform_contact = True

    def _start_crash_effect(self, failure_reason: str | None) -> None:
        if self.booster is None:
            return
        x = float(self.booster.position.x)
        y = float(self.booster.position.y)
        if failure_reason in {"ocean", "booster_body_hit_platform"}:
            y = max(y, float(getattr(self, "ocean_y", 0.0)))
        self.crash_effect_position = (x, y)
        self.crash_effect_frame = 0
        self.crash_effect_frames = CRASH_EFFECT_FRAMES

    @staticmethod
    def _dense_shaping(state: np.ndarray) -> float:
        bottom_x = state[0] - math.sin(float(state[4])) * BOOSTER_BOTTOM_X_OFFSET
        vertical_position_penalty = 100 * abs(state[1]) if abs(bottom_x) <= PLATFORM_HALF_X else 0.0
        return float(
            -100 * abs(state[0])
            - vertical_position_penalty
            - 100 * abs(state[2])
            - 100 * abs(state[3])
            - 120 * abs(state[4])
            - 80 * abs(state[5])
            + 12 * state[6]
            + 12 * state[7]
        )

    def _step_info(self, success: bool, failure_reason: str | None) -> dict:
        return {
            "success": success,
            "failure_reason": failure_reason,
            "platform_x": float(self.platform.position.x) if self.platform else None,
            "platform_impact_velocity": self.platform_impact_velocity,
            "wind_power": self.wind_power if self.enable_wind else 0.0,
            "wind_direction": self.wind_direction,
            "jet_fires_used": self.jet_fires_used,
            "jet_fires_remaining": max(0, self.max_jet_fires - self.jet_fires_used),
            "stable_landing_steps": self.stable_landing_steps,
            "anchored_feet": tuple(sorted(self.anchored_feet)),
            "left_foot_contact": self.left_foot_contact,
            "right_foot_contact": self.right_foot_contact,
            "body_platform_contact": self.body_platform_contact,
            "platform_contact": self.platform_contact,
        }

    def _terminal_step_result(self, state: np.ndarray, success: bool, failure_reason: str | None):
        self.failure_reason = failure_reason
        reward = self.success_reward if success else self.failure_reward
        if success:
            self._settle_successful_landing()
            state = self._get_state()
        else:
            self._start_crash_effect(failure_reason)
        if self.render_mode == "human":
            self.render()
        return state, reward, True, False, self._step_info(success, failure_reason)

    def step(self, action):
        if self.booster is None:
            raise AssertionError("You forgot to call reset()")

        self.last_action = self._action_label(action)
        self._count_action(action)

        state = self._get_state()
        terminated, success, failure_reason = self._terminal_status(state)
        if terminated:
            result = self._terminal_step_result(state, success, failure_reason)
            self.episode_return += float(result[1])
            self.last_info = result[4]
            return result

        self._update_platform()
        self._apply_wind()
        bottom_power, top_power = self._apply_engines(action)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self._queue_feet_visually_touching_platform()
        self._process_pending_leg_anchors()
        self._clamp_platform()

        state = self._get_state()
        shaping = self._dense_shaping(state)
        reward = 0.0 if self.prev_shaping is None else float(self.shaping_factor * (shaping - self.prev_shaping))
        self.prev_shaping = shaping
        reward -= bottom_power * 0.30
        reward -= top_power * 0.03

        self._update_landing_stability(state)
        terminated, success, failure_reason = self._terminal_status(state)
        if terminated:
            result = self._terminal_step_result(state, success, failure_reason)
            self.episode_step += 1
            self.episode_return += float(result[1])
            self.last_info = result[4]
            return result
        self.failure_reason = failure_reason

        next_episode_step = self.episode_step + 1
        if self.max_episode_steps is not None and next_episode_step >= self.max_episode_steps:
            reward = self.failure_reward
            self.failure_reason = "timeout"
            self.episode_step = next_episode_step
            self.episode_return += float(reward)
            info = self._step_info(False, "timeout")
            self.last_info = info
            if self.render_mode == "human":
                self.render()
            return state, reward, False, True, info

        if self.render_mode == "human":
            self.render()

        self.episode_step = next_episode_step
        self.episode_return += float(reward)
        info = self._step_info(success, failure_reason)
        self.last_info = info
        return state, reward, terminated, False, info

    def render(self, *, dashboard: bool | None = None):
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as exc:  # pragma: no cover - optional rendering dependency
            raise DependencyNotInstalled("pygame is not installed. Install it with `pip install pygame`.") from exc

        show_dashboard = self.dashboard if dashboard is None else bool(dashboard)

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        surf.fill((186, 220, 235))
        w = VIEWPORT_W / SCALE
        h = VIEWPORT_H / SCALE

        ocean_top = (self.ocean_y if hasattr(self, "ocean_y") else h / 4) * SCALE
        ocean_rect = pygame.Rect(0, 0, VIEWPORT_W, ocean_top)
        pygame.draw.rect(surf, (31, 91, 133), ocean_rect)
        for i in range(0, VIEWPORT_W, 48):
            pygame.draw.arc(surf, (62, 132, 176), (i, ocean_top - 8, 56, 18), 0, math.pi, 2)

        self._draw_flames(surf, pygame, gfxdraw)

        for obj in self.drawlist:
            for fixture in obj.fixtures:
                if fixture.userData == "ocean":
                    continue
                trans = fixture.body.transform
                if type(fixture.shape) is circleShape:
                    center = trans * fixture.shape.pos * SCALE
                    radius = max(2, int(fixture.shape.radius * SCALE))
                    pygame.draw.circle(surf, obj.color1, center, radius)
                    pygame.draw.circle(surf, obj.color2, center, max(1, radius - 1))
                    continue

                path = [trans * vertex * SCALE for vertex in fixture.shape.vertices]
                color = obj.color1
                outline = obj.color2
                if fixture.userData == "platform":
                    color = (45, 48, 54)
                    outline = (235, 235, 235)
                elif fixture.userData in {"left_foot", "right_foot"}:
                    color = (42, 46, 54)
                    outline = (245, 245, 245)
                pygame.draw.polygon(surf, color, path)
                gfxdraw.aapolygon(surf, path, color)
                pygame.draw.aalines(surf, outline, True, path)

        self._draw_booster_details(surf, pygame)
        self._draw_crash_effect(surf, pygame, gfxdraw)
        surf = pygame.transform.flip(surf, False, True)

        if show_dashboard:
            self._draw_dashboard(surf, pygame)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))
        return None

    def _draw_flames(self, surf, pygame, gfxdraw) -> None:
        if self.booster is None:
            return

        layer = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)

        if self.bottom_flame_power > 0.03:
            nozzle, direction = self._local_flame_anchor(
                (0.0, -BOTTOM_ENGINE_Y / SCALE),
                (0.0, -(BOTTOM_ENGINE_Y + 1.0) / SCALE),
            )
            self._draw_flame_cone(
                layer,
                pygame,
                gfxdraw,
                nozzle=nozzle,
                direction=direction,
                power=self.bottom_flame_power,
                length_px=34,
                base_width_px=7,
                end_width_px=20,
            )

        if self.top_flame_power > 0.03 and self.top_flame_direction:
            local_x = TOP_ENGINE_AWAY / SCALE * self.top_flame_direction
            nozzle, direction = self._local_flame_anchor(
                (local_x, TOP_ENGINE_Y / SCALE),
                (
                    local_x + self.top_flame_direction / SCALE,
                    TOP_ENGINE_Y / SCALE,
                ),
            )
            self._draw_flame_cone(
                layer,
                pygame,
                gfxdraw,
                nozzle=nozzle,
                direction=direction,
                power=self.top_flame_power,
                length_px=18,
                base_width_px=4,
                end_width_px=10,
            )

        surf.blit(layer, (0, 0))

    def _draw_crash_effect(self, surf, pygame, gfxdraw) -> None:
        if self.crash_effect_position is None or self.crash_effect_frame >= self.crash_effect_frames:
            return

        progress = self.crash_effect_frame / max(1, self.crash_effect_frames - 1)
        fade = 1.0 - progress
        center = (
            int(round(self.crash_effect_position[0] * SCALE)),
            int(round(self.crash_effect_position[1] * SCALE)),
        )
        layer = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)

        flash_radius = int(10 + 34 * progress)
        shock_radius = int(18 + 60 * progress)
        smoke_radius = int(16 + 46 * progress)
        flash_alpha = int(230 * fade)
        smoke_alpha = int(120 * fade)

        if shock_radius > 2:
            pygame.draw.circle(layer, (255, 238, 170, int(160 * fade)), center, shock_radius, width=3)
            gfxdraw.aacircle(layer, center[0], center[1], shock_radius, (255, 238, 170, int(160 * fade)))

        if smoke_radius > 2:
            pygame.draw.circle(layer, (42, 45, 50, smoke_alpha), center, smoke_radius)
            gfxdraw.aacircle(layer, center[0], center[1], smoke_radius, (42, 45, 50, smoke_alpha))

        pygame.draw.circle(layer, (255, 116, 34, flash_alpha), center, flash_radius)
        gfxdraw.aacircle(layer, center[0], center[1], flash_radius, (255, 116, 34, flash_alpha))
        inner_radius = max(3, int(flash_radius * 0.45))
        pygame.draw.circle(layer, (255, 236, 142, int(245 * fade)), center, inner_radius)

        for i in range(14):
            angle = i * (2 * math.pi / 14.0) + 0.35 * self.crash_effect_frame
            distance = 16 + 86 * progress * (0.55 + 0.45 * ((i * 7) % 11) / 10.0)
            start = np.array(center, dtype=np.float64)
            end = start + np.array([math.cos(angle), math.sin(angle)]) * distance
            spark_start = start + (end - start) * 0.55
            color = (255, 186, 74, int(210 * fade))
            pygame.draw.line(
                layer,
                color,
                tuple(np.round(spark_start).astype(int)),
                tuple(np.round(end).astype(int)),
                width=2,
            )

        surf.blit(layer, (0, 0))
        self.crash_effect_frame += 1

    def _local_flame_anchor(
        self,
        nozzle_local: tuple[float, float],
        direction_local: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        assert self.booster is not None
        nozzle_world = self.booster.transform * nozzle_local
        direction_world = self.booster.transform * direction_local
        nozzle = (nozzle_world[0] * SCALE, nozzle_world[1] * SCALE)
        direction = (
            direction_world[0] * SCALE - nozzle[0],
            direction_world[1] * SCALE - nozzle[1],
        )
        return nozzle, direction

    @staticmethod
    def _draw_flame_cone(
        layer,
        pygame,
        gfxdraw,
        *,
        nozzle: tuple[float, float],
        direction: tuple[float, float],
        power: float,
        length_px: float,
        base_width_px: float,
        end_width_px: float,
    ) -> None:
        direction_vec = np.array(direction, dtype=np.float64)
        norm = np.linalg.norm(direction_vec)
        if norm == 0:
            return

        direction_vec /= norm
        perp = np.array([-direction_vec[1], direction_vec[0]])
        power = float(np.clip(power, 0.0, 1.0))
        start = np.array(nozzle, dtype=np.float64)
        end = start + direction_vec * length_px * (0.45 + 0.55 * power)

        def poly(base_width: float, end_width: float, length_scale: float) -> list[tuple[int, int]]:
            scaled_end = start + (end - start) * length_scale
            return [
                tuple(np.round(start + perp * base_width / 2).astype(int)),
                tuple(np.round(start - perp * base_width / 2).astype(int)),
                tuple(np.round(scaled_end - perp * end_width / 2).astype(int)),
                tuple(np.round(scaled_end + perp * end_width / 2).astype(int)),
            ]

        alpha = int(210 * power)
        outer = poly(base_width_px, end_width_px, 1.0)
        middle = poly(base_width_px * 0.65, end_width_px * 0.58, 0.78)
        inner = poly(base_width_px * 0.32, end_width_px * 0.24, 0.48)

        pygame.draw.polygon(layer, (69, 39, 24, int(120 * power)), outer)
        gfxdraw.aapolygon(layer, outer, (69, 39, 24, int(120 * power)))
        pygame.draw.polygon(layer, (255, 112, 28, alpha), middle)
        gfxdraw.aapolygon(layer, middle, (255, 112, 28, alpha))
        pygame.draw.polygon(layer, (255, 235, 132, int(245 * power)), inner)
        gfxdraw.aapolygon(layer, inner, (255, 235, 132, int(245 * power)))

    def _draw_booster_details(self, surf, pygame) -> None:
        if self.booster is None:
            return
        trans = self.booster.transform
        stripe_points = [
            (-6 / SCALE, 20 / SCALE),
            (6 / SCALE, 20 / SCALE),
            (6 / SCALE, 28 / SCALE),
            (-6 / SCALE, 28 / SCALE),
        ]
        stripe = [trans * point * SCALE for point in stripe_points]
        pygame.draw.polygon(surf, (35, 40, 48), stripe)

        logo_center = trans * (0, 35 / SCALE) * SCALE
        font_rect = pygame.Rect(0, 0, 18, 7)
        font_rect.center = logo_center
        pygame.draw.rect(surf, (35, 40, 48), font_rect, border_radius=1)

    def _action_label(self, action) -> str:
        if self.continuous:
            arr = np.asarray(action, dtype=np.float32)
            if arr[0] > 0.0:
                return "bottom"
            if arr[1] < -0.5:
                return "upper-left"
            if arr[1] > 0.5:
                return "upper-right"
            return "no-op"
        return ACTION_NAMES.get(int(action), str(action))

    def _count_action(self, action) -> None:
        label = self._action_label(action)
        if label in self.action_counts:
            self.action_counts[label] += 1

    def _draw_dashboard(self, surf: "pygame.Surface", pygame) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

        font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 20)

        panel = pygame.Surface((242, 262), pygame.SRCALPHA)
        panel.fill((21, 32, 42, 214))
        surf.blit(panel, (6, 6))

        last_action = self.last_action or "reset"
        total_fires = sum(self.action_counts.values())
        anchored = ",".join(sorted(getattr(self, "anchored_feet", []))) or "-"
        success = bool(self.last_info.get("success", False))

        lines = [
            (f"run {self.run_count:>2}   step {self.episode_step:<4} action {last_action}", (238, 244, 247), font),
            (f"upper-left:  {self.action_counts['upper-left']:>3}", (238, 244, 247), font),
            (f"bottom:      {self.action_counts['bottom']:>3}", (255, 214, 95), font),
            (f"upper-right: {self.action_counts['upper-right']:>3}", (238, 244, 247), font),
            (f"total fires: {total_fires}/{self.max_jet_fires}", (238, 244, 247), font),
            (f"return: {self.episode_return:8.2f}", (238, 244, 247), font),
            (
                "contact: "
                f"L={int(bool(getattr(self, 'left_foot_contact', False)))} "
                f"R={int(bool(getattr(self, 'right_foot_contact', False)))} "
                f"B={int(bool(getattr(self, 'body_platform_contact', False)))}",
                (238, 244, 247),
                small_font,
            ),
            (f"anchored: {anchored}", (238, 244, 247), small_font),
            (f"stable: {getattr(self, 'stable_landing_steps', 0)}", (238, 244, 247), font),
            ("success" if success else "success: -", (89, 218, 130) if success else (154, 166, 176), font),
        ]

        y = 18
        for text, color, line_font in lines:
            surf.blit(line_font.render(text, True, color), (14, y))
            y += 23

    def close(self) -> None:
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def heuristic(env: Falcon9, state: np.ndarray):
    """A simple controller for quick smoke tests and demos."""

    angle_target = state[0] * 0.55 + state[2] * 0.9
    angle_target = float(np.clip(angle_target, -0.35, 0.35))
    hover_target = 0.45 * abs(state[0])

    angle_todo = (angle_target - state[4]) * 0.55 - state[5] * 0.85
    hover_todo = (hover_target - state[1]) * 0.55 - state[3] * 0.55

    if state[6] or state[7]:
        angle_todo = -state[4] * 0.9 - state[5] * 0.5
        hover_todo = -state[3] * 0.5

    if env.unwrapped.continuous:
        action = np.array([hover_todo * 18 - 1, angle_todo * 18], dtype=np.float32)
        return np.clip(action, -1, +1)

    action = 0
    if hover_todo > abs(angle_todo) and hover_todo > 0.04:
        action = 2
    elif angle_todo < -0.04:
        action = 1
    elif angle_todo > 0.04:
        action = 3
    return action


def demo_heuristic_lander(seed: int | None = None, render: bool = False) -> float:
    env = Falcon9(render_mode="human" if render else None)
    total_reward = 0.0
    state, _ = env.reset(seed=seed)
    for _ in range(1000):
        action = heuristic(env, state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return total_reward


if __name__ == "__main__":
    demo_heuristic_lander(render=True)


def register_falcon9_env() -> None:
    """Register ``Falcon9-v0`` with Gymnasium.

    Notebook kernels can keep an older registration alive after files change.
    Replacing this package's id makes ``gym.make("Falcon9-v0", ...)`` resolve to
    the current self-contained environment instead of a stale entry point.
    """

    env_id = "Falcon9-v0"
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]
    register(
        id=env_id,
        entry_point="falcon9.falcon9:Falcon9",
        reward_threshold=100.0,
    )


register_falcon9_env()
