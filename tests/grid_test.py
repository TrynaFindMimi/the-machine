import cv2
from utils.fps import FPSCounter
from utils.grid import draw_grid_landmarks
from utils.hand import FONT, WHITE, draw_skeleton, spaced
from utils.text import FPS_FMT, GRID_TITLE

_meter = FPSCounter()


def draw(frame, results):
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
