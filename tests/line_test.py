"""Capa de presentación - modo line. Delega lógica a utils/line.py y dibujo a utils/hand.py."""

import cv2
import numpy as np

from utils.fps import FPSCounter
from utils.hand import FONT, WHITE, draw_landmarks, draw_skeleton, spaced
from utils.line import BLUE, FINGERS_TOGETHER_THRESH, Perceptron, build_dataset, draw_line
from utils.line import MAX_EPOCHS as _MAX_EPOCHS  # re-export compat
from utils.text import FPS_FMT, LINE_TITLE

# Re-export para compatibilidad
from utils.line import MARGIN, N_SAMPLES, LR, boundary_points  # noqa: F401

P_A = 4  # etiqueta 5 en pantalla (yema pulgar)
P_B = 8  # etiqueta 9 en pantalla (yema indice)

_meter = FPSCounter()


def draw(frame, results):
    fps = _meter.tick()
    title = spaced(LINE_TITLE)
    fps_txt = FPS_FMT.format(fps)
    cv2.putText(frame, title, (16, 24), FONT, 1.0, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (16, 42), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    h, w = frame.shape[:2]
    hands = results.hand_landmarks

    if not hands:
        return frame

    for lm in hands[:2]:
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)

        p5 = np.array([lm[P_A].x, lm[P_A].y])
        p9 = np.array([lm[P_B].x, lm[P_B].y])

        # si dedos juntos no dibujar línea
        if float(np.linalg.norm(p9 - p5)) < FINGERS_TOGETHER_THRESH:
            continue

        data = build_dataset(p5, p9)
        if data[0] is None:
            continue
        X, y = data

        perc = Perceptron()
        perc.train(X, y)

        draw_line(frame, pts[P_A], pts[P_B], color=BLUE, thickness=3)
        cv2.circle(frame, pts[P_A], 8, BLUE, 2, cv2.LINE_AA)
        cv2.circle(frame, pts[P_B], 8, BLUE, 2, cv2.LINE_AA)

    return frame
