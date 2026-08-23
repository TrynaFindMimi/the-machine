import cv2

from utils.fps import FPSCounter
from utils.hand import FONT, WHITE, draw_hands, spaced
from utils.text import FPS_FMT, HAND_TITLE

_meter = FPSCounter()


def draw(frame, results):
    fps = _meter.tick()
    title = spaced(HAND_TITLE)
    fps_txt = FPS_FMT.format(fps)
    cv2.putText(frame, title, (16, 24), FONT, 1.0, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (16, 42), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    draw_hands(frame, results)

    return frame
