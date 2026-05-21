from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from gymnasium import Env, spaces
    from gymnasium.envs.registration import register
    from gymnasium.utils import seeding
except ImportError:
    Env = object
    register = None

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def contains(self, x: object) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class _Spaces:
        Discrete = _Discrete

    class _Seeding:
        @staticmethod
        def np_random(seed: Optional[int] = None):
            return np.random.default_rng(seed), seed

    spaces = _Spaces()
    seeding = _Seeding()


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ACTION_TO_STR = {
    NORTH: "N",
    EAST: "E",
    SOUTH: "S",
    WEST: "W",
}


@dataclass(frozen=True)
class Cell:
    symbol: str
    reward: float = 0.0
    terminal: bool = False


DEFAULT_MAP = (
    (".", ".", "L", "V"),
    ("S", ".", "L", "."),
    ("G", ".", ".", "."),
)
START_POS = (1, 0)

CELL_TYPES = {
    "S": Cell("S"),
    ".": Cell("."),
    "L": Cell("L", reward=-50.0, terminal=True),
    "V": Cell("V", reward=20.0, terminal=True),
    "G": Cell("G", reward=2.0, terminal=True),
}

TILE_ASSETS = {
    "L": "lava",
    "G": "goal",
    "S": "start",
    "V": "view",
}

ELF_ARROW_POINTS = {
    NORTH: ((0.5, 0.12), (0.42, 0.28), (0.58, 0.28)),
    SOUTH: ((0.5, 0.88), (0.42, 0.72), (0.58, 0.72)),
    EAST: ((0.88, 0.5), (0.72, 0.42), (0.72, 0.58)),
    WEST: ((0.12, 0.5), (0.28, 0.42), (0.28, 0.58)),
}


class VolcanoWorldEnv(Env):
    """
    Small GridWorld inspired by Gymnasium's FrozenLake.

    Layout (3x4):
        . . L V
        S . L .
        G . . .

    Coordinates are (row, col). Terminal states:
        - Start at (1, 0)
        - Lava at (0, 2) and (1, 2): -50 reward
        - View at (0, 3): +20 reward
        - Safe at (2, 0): +2 reward

    Actions:
        0 = North, 1 = East, 2 = South, 3 = West

    When slippery, the agent can move to the intended direction or slip to
    the adjacent compass directions with configurable probabilities.
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        is_slippery: bool = True,
        slip_probabilities: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        self.render_mode = render_mode
        self.is_slippery = is_slippery
        self.slip_probabilities = self._validate_slip_probabilities(slip_probabilities)

        self.desc = np.asarray(DEFAULT_MAP, dtype="U1")
        self.nrow, self.ncol = self.desc.shape
        self.nS = self.nrow * self.ncol
        self.nA = 4

        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.to_s(*START_POS)] = 1.0

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )

        self.P = {state: {action: [] for action in range(self.nA)} for state in range(self.nS)}
        self._build_transitions()

        self.np_random = None
        self.s = self.to_s(*START_POS)
        self.lastaction = None
        self._asset_cache = None
        self._human_window = None
        self._human_clock = None

    @staticmethod
    def _validate_slip_probabilities(
        slip_probabilities: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if len(slip_probabilities) != 3:
            raise ValueError("slip_probabilities must contain three values: (left, intended, right)")
        total = sum(slip_probabilities)
        if not np.isclose(total, 1.0):
            raise ValueError("slip_probabilities must sum to 1.0")
        if any(prob < 0 for prob in slip_probabilities):
            raise ValueError("slip_probabilities cannot contain negative values")
        return slip_probabilities

    def to_s(self, row: int, col: int) -> int:
        return row * self.ncol + col

    def from_s(self, state: int) -> tuple[int, int]:
        return divmod(state, self.ncol)

    def _move(self, row: int, col: int, action: int) -> tuple[int, int]:
        if action == NORTH:
            row = max(row - 1, 0)
        elif action == EAST:
            col = min(col + 1, self.ncol - 1)
        elif action == SOUTH:
            row = min(row + 1, self.nrow - 1)
        elif action == WEST:
            col = max(col - 1, 0)
        else:
            raise ValueError(f"Unknown action: {action}")
        return row, col

    def _cell(self, row: int, col: int) -> Cell:
        return CELL_TYPES[self.desc[row, col]]

    def _transition_candidates(self, action: int) -> tuple[tuple[int, float], ...]:
        if not self.is_slippery:
            return ((action, 1.0),)
        return (
            ((action - 1) % self.nA, self.slip_probabilities[0]),
            (action, self.slip_probabilities[1]),
            ((action + 1) % self.nA, self.slip_probabilities[2]),
        )

    def _build_transitions(self) -> None:
        for row in range(self.nrow):
            for col in range(self.ncol):
                state = self.to_s(row, col)
                current_cell = self._cell(row, col)

                for action in range(self.nA):
                    transitions = self.P[state][action]
                    if current_cell.terminal:
                        transitions.append((1.0, state, 0.0, True))
                        continue

                    for actual_action, prob in self._transition_candidates(action):
                        next_row, next_col = self._move(row, col, actual_action)
                        next_state = self.to_s(next_row, next_col)
                        next_cell = self._cell(next_row, next_col)
                        transitions.append(
                            (prob, next_state, next_cell.reward, next_cell.terminal)
                        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        self.np_random, _ = seeding.np_random(seed)
        self.s = int(self.np_random.choice(self.nS, p=self.initial_state_distrib))
        self.lastaction = None

        if self.render_mode == "human":
            self.render()

        return int(self.s), {"prob": 1.0}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{self.nA - 1}")

        transitions = self.P[self.s][action]
        probs = [transition[0] for transition in transitions]
        idx = int(self.np_random.choice(len(transitions), p=probs))
        prob, next_state, reward, terminated = transitions[idx]

        self.s = next_state
        self.lastaction = action

        if self.render_mode == "human":
            self.render()

        return int(next_state), float(reward), bool(terminated), False, {"prob": prob}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

        output = StringIO()
        row, col = self.from_s(self.s)
        desc = self.desc.tolist()
        desc[row][col] = f"[{desc[row][col]}]"
        output.write("\n".join(" ".join(cell for cell in line) for line in desc))
        if self.lastaction is not None:
            output.write(f"\nLast action: {ACTION_TO_STR[self.lastaction]}")
        output.write("\n")

        rendered = output.getvalue()
        if self.render_mode == "human":
            self._render_human()
            return None
        return rendered

    def close(self):
        if self._human_window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._human_window = None
            self._human_clock = None

    def _render_rgb_array(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", self.window_size)
        draw = ImageDraw.Draw(canvas)
        assets = self._load_render_assets()

        for row in range(self.nrow):
            for col in range(self.ncol):
                pos = (col * self.cell_size[0], row * self.cell_size[1])
                rect = (
                    pos[0],
                    pos[1],
                    pos[0] + self.cell_size[0] - 1,
                    pos[1] + self.cell_size[1] - 1,
                )
                canvas.paste(assets["ice"], pos)

                overlay_name = TILE_ASSETS.get(self.desc[row, col])
                if overlay_name is not None:
                    overlay = assets[overlay_name]
                    canvas.paste(overlay, pos, overlay)

                draw.rectangle(rect, outline=(180, 200, 230), width=1)

        bot_row, bot_col = self.from_s(self.s)
        bot_pos = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else SOUTH
        player_sprite = assets["elf"][last_action]
        canvas.paste(player_sprite, bot_pos, player_sprite)

        return np.array(canvas, dtype=np.uint8)

    def _render_human(self) -> None:
        frame = self._render_rgb_array()

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for render_mode='human'") from None

        if self._human_window is None:
            pygame.init()
            pygame.display.init()
            self._human_window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Volcano World")
            self._human_clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._human_window.blit(surface, (0, 0))
        pygame.display.flip()
        self._human_clock.tick(self.metadata["render_fps"])

    def _load_render_assets(self) -> dict[str, object]:
        if self._asset_cache is not None:
            return self._asset_cache

        assets = self._load_local_assets(Path(__file__).resolve().parent / "assets")
        if assets is not None:
            self._asset_cache = assets
            return assets

        self._asset_cache = self._build_fallback_assets()
        return self._asset_cache

    def _load_local_assets(self, asset_dir: Path) -> Optional[dict[str, object]]:
        from PIL import Image

        if not asset_dir.exists():
            return None

        resampling = getattr(Image, "Resampling", Image)

        def load_sprite(filename: str):
            path = asset_dir / filename
            if not path.exists():
                path = asset_dir / "blank.png"
            if not path.exists():
                return None
            return Image.open(path).convert("RGBA").resize(self.cell_size, resampling.NEAREST)

        blank = load_sprite("blank.png")
        if blank is None:
            return None

        green = load_sprite("green.png")
        safe = load_sprite("safe.png")
        sprites = {
            "ice": green.convert("RGB") if green is not None else blank.convert("RGB"),
            "goal": safe if safe is not None else blank,
            "start": green if green is not None else blank,
            "lava": load_sprite("lava.png") or blank,
            "view": load_sprite("view.png") or blank,
            "elf": {
                WEST: load_sprite("elf_left.png") or blank,
                SOUTH: load_sprite("elf_down.png") or blank,
                EAST: load_sprite("elf_right.png") or blank,
                NORTH: load_sprite("elf_up.png") or blank,
            },
        }

        return sprites

    def _build_fallback_assets(self) -> dict[str, object]:
        from PIL import Image, ImageDraw

        w, h = self.cell_size
        xy = lambda *pts: [self._scale_point(x, y, w, h) for x, y in pts]

        ice = Image.new("RGB", self.cell_size, (194, 229, 255))
        ice_draw = ImageDraw.Draw(ice)
        ice_draw.rounded_rectangle((2, 2, w - 3, h - 3), radius=8, fill=(204, 236, 255))
        ice_draw.line((w * 0.15, h * 0.25, w * 0.85, h * 0.45), fill=(245, 252, 255), width=3)
        ice_draw.line((w * 0.25, h * 0.7, w * 0.6, h * 0.2), fill=(170, 215, 245), width=2)
        ice_draw.line((w * 0.55, h * 0.8, w * 0.8, h * 0.55), fill=(170, 215, 245), width=2)

        goal, goal_draw = self._new_rgba_canvas()
        box = (w * 0.22, h * 0.3, w * 0.78, h * 0.78)
        goal_draw.rounded_rectangle(box, radius=8, fill=(210, 45, 45, 255), outline=(130, 15, 15, 255), width=2)
        goal_draw.rectangle((w * 0.46, h * 0.3, w * 0.54, h * 0.78), fill=(255, 230, 80, 255))
        goal_draw.rectangle((w * 0.22, h * 0.48, w * 0.78, h * 0.56), fill=(255, 230, 80, 255))
        goal_draw.ellipse((w * 0.35, h * 0.16, w * 0.52, h * 0.4), outline=(255, 230, 80, 255), width=3)
        goal_draw.ellipse((w * 0.48, h * 0.16, w * 0.65, h * 0.4), outline=(255, 230, 80, 255), width=3)

        start, start_draw = self._new_rgba_canvas()
        start_draw.rounded_rectangle(
            (w * 0.2, h * 0.48, w * 0.8, h * 0.72),
            radius=6,
            fill=(118, 82, 47, 255),
            outline=(70, 45, 25, 255),
            width=2,
        )
        start_draw.rectangle((w * 0.28, h * 0.32, w * 0.36, h * 0.5), fill=(86, 58, 35, 255))
        start_draw.rectangle((w * 0.64, h * 0.32, w * 0.72, h * 0.5), fill=(86, 58, 35, 255))

        view, view_draw = self._new_rgba_canvas()
        view_draw.rectangle((w * 0.2, h * 0.18, w * 0.26, h * 0.82), fill=(92, 68, 40, 255))
        view_draw.polygon(xy((0.26, 0.2), (0.78, 0.34), (0.78, 0.08)), fill=(255, 255, 255, 240), outline=(70, 100, 150, 255))
        view_draw.arc((w * 0.42, h * 0.46, w * 0.82, h * 0.86), 200, 340, fill=(45, 95, 170, 255), width=4)

        def make_elf(action: int):
            sprite, sdraw = self._new_rgba_canvas()
            sdraw.ellipse((w * 0.32, h * 0.12, w * 0.68, h * 0.42), fill=(255, 222, 180, 255))
            sdraw.polygon(xy((0.5, 0.24), (0.22, 0.76), (0.78, 0.76)), fill=(28, 130, 85, 255), outline=(18, 80, 50, 255))
            sdraw.polygon(xy(*ELF_ARROW_POINTS[action]), fill=(230, 85, 45, 255))
            return sprite

        return {
            "ice": ice,
            "goal": goal,
            "start": start,
            "view": view,
            "elf": {
                NORTH: make_elf(NORTH),
                EAST: make_elf(EAST),
                SOUTH: make_elf(SOUTH),
                WEST: make_elf(WEST),
            },
        }

    def _new_rgba_canvas(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", self.cell_size, (0, 0, 0, 0))
        return image, ImageDraw.Draw(image)

    @staticmethod
    def _scale_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
        return (width * x, height * y)


def register_env() -> None:
    if register is None:
        raise ImportError("gymnasium is not installed, so environment registration is unavailable")

    register(
        id="VolcanoWorld-v0",
        entry_point="volcano_rl.volcano_world:VolcanoWorldEnv",
    )
