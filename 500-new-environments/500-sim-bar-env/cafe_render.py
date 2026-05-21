"""
Pixel-art renderer for the Cafe Barista environment.

Renders a chunky 8-bit cafe scene with LEGO-block style figures:
- barista behind the counter, moving toward active stations
- queue of customers waiting
- espresso, steamer, and oven stations
- HUD with current step and queue size
"""

from __future__ import annotations

import os
import numpy as np
import pygame


# --- Palette (limited 8-bit-ish palette) --------------------------------------
BG_TOP = (90, 60, 45)         # wall
BG_BOT = (140, 100, 70)       # floor
WALL_STRIPE = (110, 75, 55)
COUNTER = (130, 50, 50)
COUNTER_DARK = (95, 30, 30)
COUNTER_TOP = (160, 80, 80)
MENU_BOARD = (35, 35, 35)
MENU_TEXT = (240, 230, 180)
SHELF = (80, 55, 40)

# Figure colors — vibrant LEGO-block tones
SKIN = (250, 205, 170)
SKIN_DARK = (200, 150, 120)
HAIR_BARISTA = (60, 35, 20)
APRON = (200, 200, 210)
APRON_DARK = (160, 160, 175)

CUSTOMER_PALETTES = [
    # (shirt, shirt_dark, hair)
    ((220, 70, 70),  (160, 40, 40),  (60, 35, 20)),   # red
    ((70, 130, 220), (40, 90, 170),  (80, 50, 30)),   # blue
    ((90, 180, 90),  (50, 130, 50),  (200, 160, 60)), # green/blonde
    ((220, 150, 60), (170, 100, 30), (40, 25, 15)),   # orange
]

# Order icon colors
ORDER_COLORS = {
    0: (90, 50, 30),      # espresso = dark brown
    1: (230, 220, 200),   # latte = cream
    2: (220, 170, 90),    # pastry = golden
}
ORDER_NAMES = {0: "ESP", 1: "LAT", 2: "PAS"}

# Patience bar colors
BAR_BG = (40, 40, 40)
BAR_OK = (90, 200, 90)
BAR_MID = (230, 200, 60)
BAR_LOW = (220, 70, 70)

STATION_NAMES = ["ESPRESSO", "STEAMER", "OVEN"]
STATION_COLORS = [(180, 180, 200), (200, 200, 220), (190, 130, 80)]
STATION_LAYOUT = [
    {"x": 74, "y": 20, "w": 18, "barista_x": 76},
    {"x": 96, "y": 20, "w": 18, "barista_x": 98},
    {"x": 118, "y": 20, "w": 19, "barista_x": 121},
]
BARISTA_HOME_X = 8
BARISTA_Y = 35

# Window size — big chunky pixels
PIXEL = 4                  # base pixel scale (each "logical" pixel is 4x4)
LOGICAL_W, LOGICAL_H = 200, 110
WIN_W, WIN_H = LOGICAL_W * PIXEL, LOGICAL_H * PIXEL


def _rect(surf, color, x, y, w, h):
    """Draw a logical-pixel rectangle (scaled up by PIXEL)."""
    pygame.draw.rect(surf, color, (x * PIXEL, y * PIXEL, w * PIXEL, h * PIXEL))


def _px(surf, color, x, y):
    """Draw a single chunky logical pixel."""
    pygame.draw.rect(surf, color, (x * PIXEL, y * PIXEL, PIXEL, PIXEL))


def _draw_figure(surf, x: int, y: int, shirt, shirt_dark, hair, is_barista=False):
    """
    Draw a chunky 12-tall LEGO-style figure with anchor at top-left (x, y).
    Footprint: ~10 wide x 18 tall (logical pixels).
    """
    # Hair / hat (3 tall)
    _rect(surf, hair, x + 2, y, 6, 2)
    _rect(surf, hair, x + 1, y + 1, 8, 2)
    # Face
    _rect(surf, SKIN, x + 2, y + 3, 6, 4)
    _rect(surf, SKIN_DARK, x + 2, y + 6, 6, 1)  # chin shadow
    # Eyes
    _px(surf, (20, 20, 20), x + 3, y + 4)
    _px(surf, (20, 20, 20), x + 6, y + 4)
    # Body / shirt
    if is_barista:
        # Apron over shirt
        _rect(surf, shirt, x + 1, y + 7, 8, 1)        # collar/shoulders
        _rect(surf, APRON, x + 2, y + 8, 6, 6)
        _rect(surf, APRON_DARK, x + 2, y + 13, 6, 1)
        # Arms
        _rect(surf, shirt, x, y + 8, 2, 4)
        _rect(surf, shirt, x + 8, y + 8, 2, 4)
        _rect(surf, SKIN, x, y + 12, 2, 2)
        _rect(surf, SKIN, x + 8, y + 12, 2, 2)
    else:
        _rect(surf, shirt, x + 1, y + 7, 8, 6)
        _rect(surf, shirt_dark, x + 1, y + 12, 8, 1)
        # Arms
        _rect(surf, shirt, x, y + 8, 2, 4)
        _rect(surf, shirt, x + 8, y + 8, 2, 4)
        _rect(surf, SKIN, x, y + 12, 2, 1)
        _rect(surf, SKIN, x + 8, y + 12, 2, 1)
    # Legs
    _rect(surf, (50, 50, 70), x + 2, y + 14, 2, 4)
    _rect(surf, (50, 50, 70), x + 6, y + 14, 2, 4)
    # Feet
    _rect(surf, (30, 25, 20), x + 1, y + 17, 3, 1)
    _rect(surf, (30, 25, 20), x + 6, y + 17, 3, 1)


def _draw_order_bubble(surf, x: int, y: int, order_type: int, font_small):
    """Tiny thought bubble showing the order icon above a customer."""
    # Bubble background
    _rect(surf, (250, 250, 250), x, y, 14, 8)
    _rect(surf, (200, 200, 200), x, y + 7, 14, 1)
    # Order swatch
    _rect(surf, ORDER_COLORS[order_type], x + 1, y + 1, 4, 6)
    # Label
    label = font_small.render(ORDER_NAMES[order_type], True, (30, 30, 30))
    surf.blit(label, (x * PIXEL + 6 * PIXEL, y * PIXEL + 1 * PIXEL))


def _draw_patience_bar(surf, x: int, y: int, frac: float):
    """5-wide patience bar under a customer."""
    _rect(surf, BAR_BG, x, y, 10, 2)
    fill_w = max(0, int(round(10 * frac)))
    if frac > 0.5:
        color = BAR_OK
    elif frac > 0.25:
        color = BAR_MID
    else:
        color = BAR_LOW
    if fill_w > 0:
        _rect(surf, color, x, y, fill_w, 2)


class CafeRenderer:
    def __init__(self, mode: str = "human"):
        self.mode = mode
        # If we're in a headless environment but still need rgb_array, use dummy driver
        if mode == "rgb_array" and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.init()
        if mode == "human":
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption("Cafe Barista RL")
        else:
            self.screen = pygame.Surface((WIN_W, WIN_H))

        self.clock = pygame.time.Clock()
        # Tiny pixel-y fonts
        self.font = pygame.font.SysFont("monospace", 10 * PIXEL // 2, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 3 * PIXEL, bold=True)
        self.font_tiny = pygame.font.SysFont("monospace", 4 * PIXEL, bold=True)
        self.barista_x = float(BARISTA_HOME_X)

    def _draw_background(self, surf):
        # Wall (upper 60%)
        _rect(surf, BG_TOP, 0, 0, LOGICAL_W, 70)
        # Stripes on wall
        for x in range(0, LOGICAL_W, 8):
            _rect(surf, WALL_STRIPE, x, 0, 2, 70)
        # Floor
        _rect(surf, BG_BOT, 0, 70, LOGICAL_W, LOGICAL_H - 70)
        # Floor tile lines
        for x in range(0, LOGICAL_W, 12):
            _rect(surf, (110, 75, 55), x, 70, 1, LOGICAL_H - 70)

        # Counter, extended under the prep stations through the oven.
        _rect(surf, COUNTER, 0, 55, 140, 25)
        _rect(surf, COUNTER_TOP, 0, 53, 140, 3)
        _rect(surf, COUNTER_DARK, 0, 78, 140, 2)
        # Counter panel detail
        for x in (10, 30, 50, 78, 100, 122):
            _rect(surf, COUNTER_DARK, x, 60, 12, 15)

        # Shelf with bottles
        _rect(surf, SHELF, 5, 18, 30, 3)
        bottle_colors = [(180, 50, 50), (230, 180, 60), (60, 130, 200), (180, 80, 180)]
        for i, c in enumerate(bottle_colors):
            _rect(surf, c, 7 + i * 7, 11, 4, 7)
            _rect(surf, (40, 40, 40), 8 + i * 7, 9, 2, 2)

        # Menu board
        _rect(surf, MENU_BOARD, 38, 12, 25, 22)
        _rect(surf, (60, 60, 60), 38, 12, 25, 1)
        menu_lines = ["MENU", "ESP $5", "LAT $8", "PAS $6"]
        for i, line in enumerate(menu_lines):
            label = self.font_small.render(line, True, MENU_TEXT)
            surf.blit(label, (40 * PIXEL, (14 + i * 5) * PIXEL))

    def _active_station(self, stations):
        busy = [i for i, remaining in enumerate(stations) if remaining > 0]
        if not busy:
            return None
        return max(busy, key=lambda i: stations[i])

    def _draw_barista(self, surf, active_station=None):
        if active_station is None:
            target_x = BARISTA_HOME_X
        else:
            target_x = STATION_LAYOUT[active_station]["barista_x"]

        # Ease toward the selected station so repeated renders show movement.
        self.barista_x += (target_x - self.barista_x) * 0.65
        draw_x = int(round(self.barista_x))

        if active_station is not None:
            _rect(surf, (255, 230, 120), draw_x - 1, BARISTA_Y - 1, 12, 20)
        _draw_figure(surf, x=draw_x, y=BARISTA_Y, shirt=(80, 60, 40),
                     shirt_dark=(50, 35, 25), hair=HAIR_BARISTA, is_barista=True)

    def _draw_station_label(self, surf, x, y, text):
        label = self.font_small.render(text, True, (30, 30, 30))
        surf.blit(label, (x * PIXEL, y * PIXEL))

    def _draw_espresso_station(self, surf, x, y):
        _rect(surf, (185, 185, 205), x, y + 2, 17, 15)
        _rect(surf, (80, 80, 90), x + 1, y + 3, 15, 2)
        _rect(surf, (55, 55, 65), x + 3, y + 6, 5, 5)
        _rect(surf, (35, 35, 45), x + 5, y + 11, 8, 2)
        _rect(surf, ORDER_COLORS[0], x + 10, y + 6, 4, 5)
        _rect(surf, (40, 25, 15), x + 11, y + 11, 3, 3)
        _rect(surf, (230, 230, 240), x + 7, y, 3, 2)
        self._draw_station_label(surf, x + 1, y + 18, "ESP")

    def _draw_steamer_station(self, surf, x, y):
        _rect(surf, (195, 205, 220), x, y + 4, 17, 13)
        _rect(surf, (70, 80, 90), x + 2, y + 5, 13, 2)
        _rect(surf, (225, 225, 235), x + 3, y + 8, 5, 7)
        _rect(surf, ORDER_COLORS[1], x + 10, y + 8, 4, 6)
        for sx in (4, 8, 12):
            _rect(surf, (235, 235, 245), x + sx, y, 1, 3)
        self._draw_station_label(surf, x + 1, y + 18, "STM")

    def _draw_oven_station(self, surf, x, y):
        _rect(surf, (185, 120, 70), x, y + 2, 18, 15)
        _rect(surf, (85, 45, 30), x + 2, y + 5, 14, 8)
        _rect(surf, (240, 180, 70), x + 4, y + 7, 10, 4)
        _rect(surf, (60, 35, 25), x + 3, y + 14, 12, 1)
        _rect(surf, (220, 210, 190), x + 14, y + 3, 2, 2)
        self._draw_station_label(surf, x + 1, y + 18, "OVEN")

    def _draw_stations(self, surf, stations, station_target, active_station=None):
        """Draw espresso, steamer, and oven stations on the back counter."""
        for i in range(len(stations)):
            layout = STATION_LAYOUT[i]
            x = layout["x"]
            y = layout["y"]

            if active_station == i:
                _rect(surf, (255, 230, 120), x - 1, y - 1, layout["w"] + 2, 24)

            if i == 0:
                self._draw_espresso_station(surf, x, y)
            elif i == 1:
                self._draw_steamer_station(surf, x, y)
            else:
                self._draw_oven_station(surf, x, y)

            # Status light
            if stations[i] > 0:
                light = (220, 70, 70)  # busy = red
            else:
                light = (90, 200, 90)  # ready = green
            _rect(surf, light, x + layout["w"] - 3, y + 2, 2, 2)

            # Busy bar
            if stations[i] > 0:
                from cafe_env import ORDER_PREP_TIME  # for max
                max_prep = max(ORDER_PREP_TIME.values())
                frac = stations[i] / max_prep
                _rect(surf, BAR_BG, x + 1, y + 16, layout["w"] - 2, 2)
                fill_w = max(0, int(round((layout["w"] - 2) * frac)))
                if fill_w > 0:
                    _rect(surf, (220, 100, 100), x + 1, y + 16, fill_w, 2)

    def _draw_queue(self, surf, queue):
        """Draw queued customers + their order bubbles + patience bars."""
        from cafe_env import INITIAL_PATIENCE, MAX_QUEUE
        base_x = 80
        base_y = 65
        for i in range(MAX_QUEUE):
            x = base_x + i * 24
            y = base_y
            customer = queue[i]
            if customer is None:
                # Draw faint queue position marker
                _rect(surf, (110, 75, 55), x + 2, y + 22, 6, 1)
                # Slot number for clarity
                label = self.font_small.render(str(i + 1), True, (180, 140, 110))
                surf.blit(label, (x * PIXEL + 3 * PIXEL, y * PIXEL + 24 * PIXEL))
                continue

            palette = CUSTOMER_PALETTES[i % len(CUSTOMER_PALETTES)]
            shirt, shirt_dark, hair = palette

            # If being served, draw with a subtle glow background
            if customer.get("being_served"):
                _rect(surf, (255, 240, 150), x - 1, y - 1, 12, 22)

            _draw_figure(surf, x=x, y=y, shirt=shirt, shirt_dark=shirt_dark, hair=hair)

            # Order bubble above the customer's head
            _draw_order_bubble(surf, x=x - 2, y=y - 10,
                               order_type=customer["order"],
                               font_small=self.font_small)
            # Patience bar at the feet
            frac = customer["patience"] / INITIAL_PATIENCE
            _draw_patience_bar(surf, x=x, y=y + 19, frac=frac)
            # Slot number
            label = self.font_small.render(str(i + 1), True, (230, 230, 230))
            surf.blit(label, (x * PIXEL + 3 * PIXEL, y * PIXEL + 22 * PIXEL))

    def _draw_hud(self, surf, t, episode_length, queue, revenue=0.0, tips=0.0, customers_left=0):
        # Top bar
        _rect(surf, (20, 20, 20), 0, 0, LOGICAL_W, 8)
        queue_size = sum(1 for c in queue if c is not None)
        text = f"STEP {t:3d}/{episode_length}   QUEUE {queue_size}/{len(queue)}   LEFT {customers_left}"
        label = self.font_tiny.render(text, True, (240, 230, 180))
        surf.blit(label, (2 * PIXEL, 1 * PIXEL))

        money_text = f"REV ${revenue:5.1f}   TIPS ${tips:5.1f}"
        money_label = self.font_tiny.render(money_text, True, (150, 235, 170))
        money_x = WIN_W - money_label.get_width() - (2 * PIXEL)
        surf.blit(money_label, (money_x, 1 * PIXEL))

    def draw(
        self, queue, stations, station_target, t, episode_length,
        revenue=0.0, tips=0.0, customers_left=0,
    ):
        active_station = self._active_station(stations)
        self._draw_background(self.screen)
        self._draw_stations(self.screen, stations, station_target, active_station)
        self._draw_barista(self.screen, active_station)
        self._draw_queue(self.screen, queue)
        self._draw_hud(self.screen, t, episode_length, queue, revenue, tips, customers_left)

        if self.mode == "human":
            # Pump events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            pygame.display.flip()
            self.clock.tick(8)  # 8 fps — readable pace
            return None
        else:
            # rgb_array
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
