import cv2

from utils.fps import FPSCounter
from utils.hand import FONT, WHITE, draw_skeleton, draw_landmarks, spaced
from utils.text import FPS_FMT, HAND_TITLE

_meter = FPSCounter()


def _get_handedness(results, idx):
    if not results.handedness or idx >= len(results.handedness):
        return "?"
    cats = results.handedness[idx]
    if not cats:
        return "?"
    return cats[0].category_name


def draw(frame, results):
    fps = _meter.tick()
    title = spaced(HAND_TITLE)
    fps_txt = FPS_FMT.format(fps)
    cv2.putText(frame, title, (16, 24), FONT, 1.0, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (16, 42), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    h, w = frame.shape[:2]
    for i, hand_landmarks in enumerate(results.hand_landmarks[:2]):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)

        hand_label = _get_handedness(results, i)
        wrist = pts[0]
        cv2.putText(
            frame, hand_label, (wrist[0] - 30, wrist[1] + 30),
            FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )

    return frame
