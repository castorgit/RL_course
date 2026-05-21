"""
Cafe Barista Environment — a custom Gymnasium environment for RL practice.

The agent runs a small cafe during a rush. Customers arrive with orders that
take time to prepare on limited stations. Each customer has a patience timer
and a tip that decays as they wait. The agent must decide WHO to serve next
and WHICH station to use — a scheduling / prioritization problem.

Single-agent, discrete action space, low-dim observation. Fast to train.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# --- Game constants -----------------------------------------------------------
MAX_QUEUE = 4          # max customers waiting at any time
NUM_STATIONS = 3       # espresso machine, milk steamer, oven
NUM_ORDER_TYPES = 3    # 0=espresso, 1=latte, 2=pastry
QUEUE_SLOT_OBS_DIM = 6 # order one-hot + patience + present + being served

# Each order type needs a specific station and has a prep time (in steps)
ORDER_STATION = {0: 0, 1: 1, 2: 2}        # espresso->0, latte->1, pastry->2
ORDER_PREP_TIME = {0: 3, 1: 5, 2: 4}      # how many steps the station is busy
ORDER_REVENUE = {0: 5.0, 1: 8.0, 2: 6.0}  # menu price collected when served
ORDER_BASE_TIP = {0: 2.0, 1: 1.4, 2: 1.8} # max tip if served immediately

INITIAL_PATIENCE = 15   # steps before a customer leaves
ARRIVAL_PROB = 0.65     # per-step probability a new customer arrives
EPISODE_LENGTH = 120    # one "morning rush"

# Reward shaping
OPT_REVENUE = True
OPT_TIPS = True
REVENUE_REWARD_MULTIPLIER = 1.0
TIP_REWARD_MULTIPLIER = 5.0
LEAVE_PENALTY = -5.0
IDLE_PENALTY = -0.05    # tiny nudge so the agent doesn't just idle forever


class CafeBaristaEnv(gym.Env):
    """
    Observation (flat Box, normalized roughly to [0, 1]):
        - For each queue slot (MAX_QUEUE):
            * order type (one-hot, 3 dims)
            * patience remaining (1 dim, normalized)
            * present flag (1 dim)
            * being served flag (1 dim)
          => 6 dims per slot
        - For each station (NUM_STATIONS):
            * busy time remaining (1 dim, normalized)
          => 1 dim per station

    Action (Discrete):
        0           = idle / do nothing
        1..MAX_QUEUE = serve customer in that queue slot (1-indexed)
        => total = 1 + MAX_QUEUE actions

    Reward:
        + revenue, tip, or both when a customer is served, depending on
          opt_revenue / opt_tips
        - LEAVE_PENALTY when a customer runs out of patience
        - IDLE_PENALTY each step the agent does nothing (small)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        opt_revenue: bool = OPT_REVENUE,
        opt_tips: bool = OPT_TIPS,
    ):
        super().__init__()

        if not opt_revenue and not opt_tips:
            raise ValueError("At least one reward objective must be enabled.")

        self.render_mode = render_mode
        self.opt_revenue = bool(opt_revenue)
        self.opt_tips = bool(opt_tips)

        # Observation: 6 dims per queue slot + 1 dim per station
        obs_dim = MAX_QUEUE * QUEUE_SLOT_OBS_DIM + NUM_STATIONS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: 0 = idle, 1..MAX_QUEUE = pick that queue slot
        self.action_space = spaces.Discrete(1 + MAX_QUEUE)

        # State, initialized in reset()
        self._queue: list[dict | None] = []
        self._stations: list[int] = []   # busy time remaining per station
        self._station_target: list[int] = []  # which queue slot the station is preparing for (-1 if none)
        self._t: int = 0
        self._revenue: float = 0.0
        self._tips: float = 0.0
        self._objective_reward: float = 0.0
        self._penalties: float = 0.0
        self._idle_penalties: float = 0.0
        self._leave_penalties: float = 0.0
        self._customers_left: int = 0
        self._np_random: np.random.Generator | None = None

        # Renderer is lazy-initialized
        self._renderer = None

        if seed is not None:
            self.reset(seed=seed)

    # --------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)

        self._queue = [None] * MAX_QUEUE
        self._stations = [0] * NUM_STATIONS
        self._station_target = [-1] * NUM_STATIONS
        self._t = 0
        self._revenue = 0.0
        self._tips = 0.0
        self._objective_reward = 0.0
        self._penalties = 0.0
        self._idle_penalties = 0.0
        self._leave_penalties = 0.0
        self._customers_left = 0

        # Seed the cafe with one customer so the agent has something to do
        self._spawn_customer(force=True)

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = 0.0

        # --- 1. Resolve agent action -----------------------------------------
        if action == 0:
            # Idle
            reward += self._add_penalty(IDLE_PENALTY, "idle")
        else:
            slot = action - 1
            customer = self._queue[slot]
            if customer is None:
                # Invalid pick (empty slot) — small penalty, treat as idle
                reward += self._add_penalty(IDLE_PENALTY, "idle")
            else:
                station = ORDER_STATION[customer["order"]]
                if self._stations[station] > 0:
                    # Station busy — small penalty, treat as idle
                    reward += self._add_penalty(IDLE_PENALTY, "idle")
                else:
                    # Start preparing this customer's order
                    self._stations[station] = ORDER_PREP_TIME[customer["order"]]
                    self._station_target[station] = slot
                    customer["being_served"] = True

        # --- 2. Advance stations; finished orders deliver rewards ------------
        for s in range(NUM_STATIONS):
            if self._stations[s] > 0:
                self._stations[s] -= 1
                if self._stations[s] == 0:
                    # Done — deliver to target customer
                    slot = self._station_target[s]
                    customer = self._queue[slot]
                    if customer is not None:
                        order = customer["order"]
                        patience_frac = customer["patience"] / INITIAL_PATIENCE
                        tip = ORDER_BASE_TIP[order] * patience_frac
                        revenue = ORDER_REVENUE[order]
                        self._revenue += revenue
                        self._tips += tip
                        service_reward = self._service_reward(revenue, tip)
                        self._objective_reward += service_reward
                        reward += service_reward
                        self._queue[slot] = None
                    self._station_target[s] = -1

        # --- 3. Tick patience for all queued customers ------------------------
        for i, c in enumerate(self._queue):
            if c is None:
                continue
            if c.get("being_served"):
                continue  # patience freezes once being served
            c["patience"] -= 1
            if c["patience"] <= 0:
                reward += self._add_penalty(LEAVE_PENALTY, "leave")
                self._customers_left += 1
                self._queue[i] = None

        # --- 4. Spawn new customers ------------------------------------------
        if self._np_random.random() < ARRIVAL_PROB:
            if not self._spawn_customer():
                self._customers_left += 1
                reward += self._add_penalty(LEAVE_PENALTY, "leave")

        # --- 5. Time & termination -------------------------------------------
        self._t += 1
        terminated = False
        truncated = self._t >= EPISODE_LENGTH

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self._renderer is None:
            from cafe_render import CafeRenderer  # lazy import — only needs pygame
            self._renderer = CafeRenderer(mode=self.render_mode)
        return self._renderer.draw(
            queue=self._queue,
            stations=self._stations,
            station_target=self._station_target,
            t=self._t,
            episode_length=EPISODE_LENGTH,
            revenue=self._revenue,
            tips=self._tips,
            customers_left=self._customers_left,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --------------------------------------------------------------- internals
    def _spawn_customer(self, force: bool = False) -> bool:
        # Find an empty queue slot
        for i in range(MAX_QUEUE):
            if self._queue[i] is None:
                order_type = int(self._np_random.integers(0, NUM_ORDER_TYPES))
                self._queue[i] = {
                    "order": order_type,
                    "patience": INITIAL_PATIENCE,
                    "being_served": False,
                }
                return True
        # If the queue is full, the arriving customer leaves immediately.
        return False

    def _service_reward(self, revenue: float, tip: float) -> float:
        reward = 0.0
        if self.opt_revenue:
            reward += REVENUE_REWARD_MULTIPLIER * revenue
        if self.opt_tips:
            reward += TIP_REWARD_MULTIPLIER * tip
        return reward

    def _add_penalty(self, penalty: float, kind: str) -> float:
        self._penalties += penalty
        if kind == "idle":
            self._idle_penalties += penalty
        elif kind == "leave":
            self._leave_penalties += penalty
        return penalty

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(MAX_QUEUE * QUEUE_SLOT_OBS_DIM + NUM_STATIONS, dtype=np.float32)
        for i, c in enumerate(self._queue):
            base = i * QUEUE_SLOT_OBS_DIM
            if c is not None:
                obs[base + c["order"]] = 1.0  # one-hot order
                obs[base + 3] = c["patience"] / INITIAL_PATIENCE
                obs[base + 4] = 1.0  # present
                obs[base + 5] = 1.0 if c.get("being_served") else 0.0
        for s in range(NUM_STATIONS):
            max_prep = max(ORDER_PREP_TIME.values())
            obs[MAX_QUEUE * QUEUE_SLOT_OBS_DIM + s] = self._stations[s] / max_prep
        return obs

    def _get_info(self) -> dict:
        return {
            "t": self._t,
            "queue_size": sum(1 for c in self._queue if c is not None),
            "busy_stations": sum(1 for s in self._stations if s > 0),
            "revenue": self._revenue,
            "tips": self._tips,
            "objective_reward": self._objective_reward,
            "penalties": self._penalties,
            "idle_penalties": self._idle_penalties,
            "leave_penalties": self._leave_penalties,
            "customers_left": self._customers_left,
            "opt_revenue": self.opt_revenue,
            "opt_tips": self.opt_tips,
            "revenue_reward_multiplier": REVENUE_REWARD_MULTIPLIER,
            "tip_reward_multiplier": TIP_REWARD_MULTIPLIER,
        }
