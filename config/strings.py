"""config/strings.py — textos UI inmutables."""

from __future__ import annotations

from typing import Final

WINDOW: Final = "the-machine"
USAGE: Final = "uso: python main.py [{}]"

HAND_TITLE: Final = "deteccion de manos"
LINE_TITLE: Final = "deteccion de linea"
POSITION_TITLE: Final = "deteccion de gestos"
FPS_FMT: Final = "FPS {:05.1f}"

KEYBIND_HINT: Final = "[n]ext [q]uit"

MODES_ORDER: Final[tuple[str, ...]] = ("hand", "line", "position")
MODE_LABELS: Final[dict[str, str]] = {
    "hand": HAND_TITLE,
    "line": LINE_TITLE,
    "position": POSITION_TITLE,
}

# --- Gestos (MediaPipe GestureRecognizer) — fuente canónica ---
# Nombres exactos que devuelve MediaPipe; centralizados aquí para
# evitar strings mágicos dispersos en core/palette/presentation.
GESTURE_NONE: Final = "None"
GESTURE_CLOSED_FIST: Final = "Closed_Fist"
GESTURE_OPEN_PALM: Final = "Open_Palm"
GESTURE_POINTING_UP: Final = "Pointing_Up"
GESTURE_THUMB_DOWN: Final = "Thumb_Down"
GESTURE_THUMB_UP: Final = "Thumb_Up"
GESTURE_VICTORY: Final = "Victory"
GESTURE_I_LOVE_YOU: Final = "ILoveYou"

GESTURES: Final[tuple[str, ...]] = (
    GESTURE_NONE,
    GESTURE_CLOSED_FIST,
    GESTURE_OPEN_PALM,
    GESTURE_POINTING_UP,
    GESTURE_THUMB_DOWN,
    GESTURE_THUMB_UP,
    GESTURE_VICTORY,
    GESTURE_I_LOVE_YOU,
)
# Alias usado por código legacy
GESTURE_NAMES: Final = GESTURES
