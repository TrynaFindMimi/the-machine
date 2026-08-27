import cv2

from common.fps import FPSCounter
from config.palette import WHITE
from config.strings import FPS_FMT, HAND_TITLE
from core.handedness import get_handedness
from core.results import count_hands, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_landmarks, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

_meter = FPSCounter()


def draw(frame, results):
    fps = _meter.tick()
    cv2.putText(frame, spaced(HAND_TITLE), (10, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, FPS_FMT.format(fps), (10, 32), FONT, 0.35, WHITE, 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    hand_count = count_hands(results)
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
    for i, hand_landmarks in enumerate(results.hand_landmarks[:2]):
        pts = to_pixel_points(hand_landmarks, w, h)
        draw_bbox(frame, pts, WHITE)
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)
        label = get_handedness(results, i)
        x1 = min(p[0] for p in pts) - 10
        y1 = min(p[1] for p in pts) - 10
        cv2.putText(frame, label, (max(0, x1), max(12, y1 - 6)), FONT, 0.35, WHITE, 1, cv2.LINE_AA)
    return frame, hand_count
