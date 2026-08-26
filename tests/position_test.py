import cv2

from utils.fps import FPSCounter
from utils.hand import FONT, WHITE, draw_skeleton, draw_landmarks, spaced
from utils.text import FPS_FMT

_meter = FPSCounter()

TITLE = "deteccion de posicion"

_thumb_up_count = 0
_prev_gestures = {}


def _get_handedness(results, idx):
    if not results.handedness or idx >= len(results.handedness):
        return "?"
    cats = results.handedness[idx]
    if not cats:
        return "?"
    return cats[0].category_name


def _is_thumb_up(lm):
    return lm[4].y < lm[3].y and lm[4].y < lm[2].y


def _is_finger_curled(lm, tip, pip):
    return lm[tip].y > lm[pip].y


def _is_thumbs_up_with_fist(lm):
    if not _is_thumb_up(lm):
        return False
    if not _is_finger_curled(lm, 8, 6):
        return False
    if not _is_finger_curled(lm, 12, 10):
        return False
    if not _is_finger_curled(lm, 16, 14):
        return False
    if not _is_finger_curled(lm, 20, 18):
        return False
    return True


def draw(frame, results):
    global _thumb_up_count, _prev_gestures
    fps = _meter.tick()
    title = spaced(TITLE)
    fps_txt = FPS_FMT.format(fps)
    cv2.putText(frame, title, (16, 24), FONT, 1.0, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, fps_txt, (16, 42), FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    h, w = frame.shape[:2]
    current_gestures = {}

    hands = results.hand_landmarks if results.hand_landmarks else []
    for i, hand_landmarks in enumerate(hands):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)

        hand_label = _get_handedness(results, i)
        wrist = pts[0]
        cv2.putText(
            frame, hand_label, (wrist[0] - 30, wrist[1] + 30),
            FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )

        is_gesture = _is_thumbs_up_with_fist(hand_landmarks)
        key = f"{i}_{hand_label}"
        current_gestures[key] = is_gesture

        if is_gesture:
            cv2.putText(
                frame, "THUMBS UP + FIST", (wrist[0] - 60, wrist[1] + 55),
                FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

    new_gestures = {
        k: v for k, v in current_gestures.items()
        if v and not _prev_gestures.get(k, False)
    }
    _thumb_up_count += len(new_gestures)
    _prev_gestures = current_gestures

    count_txt = f"THUMBS UP: {_thumb_up_count}"
    cv2.putText(frame, count_txt, (16, 68), FONT, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    detected = any(current_gestures.values())
    status = "DETECTADO" if detected else "ESPERANDO..."
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, status, (16, 94), FONT, 0.75, color, 2, cv2.LINE_AA)

    return frame
