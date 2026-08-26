import cv2

from utils.line import boundary_points, draw_line

WHITE = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def spaced(text: str) -> str:
    return " ".join(text.upper())


def draw_skeleton(frame, pts) -> None:
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], WHITE, 1, cv2.LINE_AA)


def draw_landmarks(frame, pts) -> None:
    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 4, WHITE, -1, cv2.LINE_AA)
        label = str(i)
        x, y = pt[0] + 7, pt[1] - 7
        cv2.putText(frame, label, (x, y), FONT, 0.4, WHITE, 1, cv2.LINE_AA)


def draw_label(frame, text: str, pt, scale: float = 0.4, color=WHITE, thickness: int = 1) -> None:
    x, y = int(pt[0]), int(pt[1])
    cv2.putText(frame, text, (x + 7, y - 7), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hands(frame, results) -> None:
    h, w = frame.shape[:2]
    for hand_landmarks in results.hand_landmarks[:2]:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        draw_skeleton(frame, pts)
        draw_landmarks(frame, pts)
