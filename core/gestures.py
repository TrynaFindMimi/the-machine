"""core/gestures.py — dominio puro: re-exporta vocabulario canónico de config/strings.

Mantiene compatibilidad con `from core.gestures import GESTURES/GESTURE_COLORS`.
La fuente canónica es `config/strings.GESTURES` (hoja config).
Capa core → config permitida.
"""

from __future__ import annotations

from typing import Final

from config.palette import GESTURE_COLORS  # noqa: F401  — re-export para compat
from config.strings import (
    GESTURE_CLOSED_FIST,
    GESTURE_I_LOVE_YOU,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_POINTING_UP,
    GESTURE_THUMB_DOWN,
    GESTURE_THUMB_UP,
    GESTURE_VICTORY,
    GESTURES,
    GESTURE_NAMES,
)

__all__ = [
    "GESTURES",
    "GESTURE_NAMES",
    "GESTURE_COLORS",
    "GESTURE_NONE",
    "GESTURE_CLOSED_FIST",
    "GESTURE_OPEN_PALM",
    "GESTURE_POINTING_UP",
    "GESTURE_THUMB_DOWN",
    "GESTURE_THUMB_UP",
    "GESTURE_VICTORY",
    "GESTURE_I_LOVE_YOU",
]
