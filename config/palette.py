"""config/palette.py — paleta BGR (OpenCV) inmutable."""

from __future__ import annotations

from typing import Final

from config.strings import (
    GESTURE_CLOSED_FIST,
    GESTURE_I_LOVE_YOU,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_POINTING_UP,
    GESTURE_THUMB_DOWN,
    GESTURE_THUMB_UP,
    GESTURE_VICTORY,
)

# Base
WHITE: Final = (245, 245, 245)
BLACK: Final = (0, 0, 0)
GREEN: Final = (0, 255, 0)
CYAN: Final = (0, 255, 255)
RED: Final = (0, 0, 255)
BLUE: Final = (255, 60, 20)

GRAY: Final = (140, 140, 140)

SIDEBAR_BG: Final = (18, 18, 18)
SIDEBAR_BORDER: Final = (70, 70, 70)

# Gestos → color (presentación). Claves tomadas de config/strings (fuente canónica).
GESTURE_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    GESTURE_NONE: RED,
    GESTURE_CLOSED_FIST: (0, 140, 255),
    GESTURE_OPEN_PALM: GREEN,
    GESTURE_POINTING_UP: CYAN,
    GESTURE_THUMB_DOWN: (0, 0, 255),
    GESTURE_THUMB_UP: (0, 255, 100),
    GESTURE_VICTORY: (255, 200, 0),
    GESTURE_I_LOVE_YOU: (200, 100, 255),
}
