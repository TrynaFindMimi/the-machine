import cv2
import numpy as np

from utils.fps import FPSCounter
from utils.hand import FONT, WHITE, draw_landmarks, draw_skeleton, spaced
from utils.line import BLUE, FINGERS_TOGETHER_THRESH, Perceptron, build_dataset, draw_line
from utils.text import FPS_FMT, LINE_TITLE

P_A = 4
P_B = 8
EPOCH_BUDGET = 200

_meter = FPSCounter()
_perc: list[Perceptron | None] = [None, None]


def draw(frame, results):
    fps = _meter.tick()
    title = spaced(LINE_TITLE)
    fps_txt = FPS_FMT.format(fps)
    cv2.putText(frame, title, (16, 24), FONT, 1.0, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (16, 42), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    h, w = frame.shape[:2]
    hands = results.hand_landmarks

    if not hands:
        _perc[0] = None
        _perc[1] = None
        return frame

    for i, lm in enumerate(hands[:2]):
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]
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

    return frame
