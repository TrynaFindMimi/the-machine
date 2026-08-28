"""app/registry.py — registro tipado de modos."""

from __future__ import annotations

from typing import Final

from presentation.modes import hand as hand_mode
from presentation.modes import line as line_mode
from presentation.modes import position as position_mode
from presentation.modes import ModeHandler

TESTS: Final[dict[str, ModeHandler]] = {
    "hand": hand_mode.draw,
    "line": line_mode.draw,
    "position": position_mode.draw,
}
TEST_NAMES: Final[tuple[str, ...]] = tuple(TESTS.keys())
