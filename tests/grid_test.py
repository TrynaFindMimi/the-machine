"""Capa de presentación - modo grid. Delega a utils/grid.py y utils/hand.py."""

import cv2  # noqa: F401 (re-export compatibilidad)
from utils.fps import FPSCounter
from utils.grid import GRID_STEP, AXIS_COLOR, draw_axes, draw_grid_landmarks  # noqa: F401
from utils.hand import FONT, WHITE, draw_skeleton, spaced
from utils.text import FPS_FMT, GRID_TITLE

_meter = FPSCounter()

# Re-export para compatibilidad con imports antiguos
from utils.grid import draw_axes as _draw_axes_reexport  # noqa: F401
from utils.hand import draw_landmarks as _draw_hand_landmarks  # noqa: F401


def draw(frame, results):
    draw_axes(frame)

    fps = _meter.tick()
    title = spaced(GRID_TITLE)
    fps_txt = FPS_FMT.format(fps)
    cv2.putText(frame, title, (16, 24), FONT, 1.0, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (16, 42), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    h, w = frame.shape[:2]
    for hand_landmarks in results.hand_landmarks[:2]:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        draw_skeleton(frame, pts)
        draw_grid_landmarks(frame, pts)

    return frame
