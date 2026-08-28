"""presentation/ui/layout.py — sidebar B/W."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import GRAY, SIDEBAR_BG, SIDEBAR_BORDER, WHITE
from config.strings import MODES_ORDER
from presentation.ui.theme import FONT, SIDEBAR_W


def draw_sidebar(
    frame: NDArray[np.uint8], current_mode: str, hand_count: int, fps: float
) -> None:
    h, w = frame.shape[:2]
    sx = w - SIDEBAR_W
    overlay = frame.copy()
    cv2.rectangle(overlay, (sx, 0), (w, h), SIDEBAR_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (sx, 0), (sx, h), SIDEBAR_BORDER, 1, cv2.LINE_AA)
    y = 36
    cv2.putText(frame, "MODE", (sx + 14, y), FONT, 0.35, GRAY, 1, cv2.LINE_AA)
    y += 18
    for mode in MODES_ORDER:
        is_active = mode == current_mode
        dot_color = WHITE if is_active else GRAY
        text_color = WHITE if is_active else GRAY
        cv2.circle(frame, (sx + 18, y - 3), 3, dot_color, -1, cv2.LINE_AA)
        cv2.putText(frame, mode.upper(), (sx + 28, y), FONT, 0.35, text_color, 1, cv2.LINE_AA)
        y += 16
    y += 8
    cv2.line(frame, (sx + 10, y), (w - 10, y), SIDEBAR_BORDER, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, "HANDS", (sx + 14, y), FONT, 0.32, GRAY, 1, cv2.LINE_AA)
    y += 16
    count_color = WHITE if hand_count > 0 else GRAY
    cv2.putText(frame, str(hand_count), (sx + 14, y), FONT, 0.9, count_color, 1, cv2.LINE_AA)
    y += 18
    cv2.line(frame, (sx + 10, y), (w - 10, y), SIDEBAR_BORDER, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, "FPS", (sx + 14, y), FONT, 0.32, GRAY, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, f"{fps:05.1f}", (sx + 14, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    y += 18
    cv2.line(frame, (sx + 10, y), (w - 10, y), SIDEBAR_BORDER, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, "CONTROLS", (sx + 14, y), FONT, 0.3, GRAY, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, "N  next", (sx + 14, y), FONT, 0.3, GRAY, 1, cv2.LINE_AA)
    y += 12
    cv2.putText(frame, "Q  quit", (sx + 14, y), FONT, 0.3, GRAY, 1, cv2.LINE_AA)
