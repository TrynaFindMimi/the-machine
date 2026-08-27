import cv2
import numpy as np

from common.fps import FPSCounter
from config.palette import BLUE, WHITE
from config.strings import FPS_FMT, LINE_TITLE
from core.perceptron import FINGERS_TOGETHER_THRESH, Perceptron, build_dataset
from core.results import count_hands, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_landmarks, draw_line, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

P_A = 4
P_B = 8
EPOCH_BUDGET = 200

_meter = FPSCounter()
_perc: list[Perceptron | None] = [None, None]


def draw(frame, results):
    fps = _meter.tick()
    cv2.putText(frame, spaced(LINE_TITLE), (10, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, FPS_FMT.format(fps), (10, 32), FONT, 0.35, WHITE, 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    hands = results.hand_landmarks
    hand_count = count_hands(results)
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
        _perc[0] = None
        _perc[1] = None
        return frame, hand_count
    if not hands:
        return frame, hand_count
    for i, lm in enumerate(hands[:2]):
        pts = to_pixel_points(lm, w, h)
        draw_bbox(frame, pts, WHITE)
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)
        p5 = np.array([lm[P_A].x, lm[P_A].y])
        p9 = np.array([lm[P_B].x, lm[P_B].y])
        if float(np.linalg.norm(p9 - p5)) < FINGERS_TOGETHER_THRESH:
            _perc[i] = None
            continue
        data = build_dataset(p5, p9)
        if data[0] is None:
            continue
        X, y = data
        if _perc[i] is None:
            _perc[i] = Perceptron()
        _perc[i].train_budget(X, y, EPOCH_BUDGET)
        draw_line(frame, pts[P_A], pts[P_B], color=BLUE, thickness=1)
    return frame, hand_count
