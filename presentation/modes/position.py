import cv2

from common.fps import FPSCounter
from config.palette import BLACK, WHITE
from config.strings import FPS_FMT, POSITION_TITLE
from core.handedness import get_handedness
from core.results import count_hands, get_gesture, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_landmarks, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

_meter = FPSCounter()


def _put_text_box(frame, text, org, font_scale, thickness, color=WHITE, bg=BLACK, pad_x=6, pad_y=4):
    """Dibuja texto con caja negra detrás para máxima legibilidad (paleta B/W)."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    # caja con padding
    x1, y1 = x - pad_x, y - th - pad_y
    x2, y2 = x + tw + pad_x, y + baseline + pad_y // 2
    # clamp dentro del frame
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw(frame, results):
    fps = _meter.tick()
    _put_text_box(frame, spaced(POSITION_TITLE), (10, 22), 0.70, 2, WHITE, BLACK)
    _put_text_box(frame, FPS_FMT.format(fps), (10, 50), 0.50, 1, WHITE, BLACK)
    h, w = frame.shape[:2]
    hands = results.hand_landmarks if results.hand_landmarks else []
    hand_count = len(hands)
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
    for i, hand_landmarks in enumerate(hands[:2]):
        pts = to_pixel_points(hand_landmarks, w, h)
        gesture_name, gesture_score = get_gesture(results, i)
        # paleta predominante B/W: bbox, skeleton y landmarks en blanco sin borde negro
        draw_bbox(frame, pts, WHITE, thickness=2)
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)
        hand_label = get_handedness(results, i)
        x1 = min(p[0] for p in pts) - 10
        y1 = min(p[1] for p in pts) - 10
        # etiquetas sobre la mano: más grandes y con caja negra para legibilidad
        _put_text_box(frame, hand_label, (max(0, x1), max(18, y1 - 10)), 0.60, 2, WHITE, BLACK, pad_x=5, pad_y=3)
        _put_text_box(
            frame,
            f"{gesture_name} {gesture_score:.0%}",
            (max(0, x1), max(18, y1 - 38)),
            0.65,
            2,
            WHITE,
            BLACK,
            pad_x=5,
            pad_y=3,
        )
    # HUD inferior: lista de manos detectadas, fuente grande y legible
    y_off = 78
    for i in range(min(hand_count, 2)):
        gesture_name, _ = get_gesture(results, i)
        label = get_handedness(results, i)
        _put_text_box(frame, f"{label}: {gesture_name}", (10, y_off), 0.60, 2, WHITE, BLACK, pad_x=6, pad_y=4)
        y_off += 30
    return frame, hand_count
