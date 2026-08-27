import cv2


def apply_cctv_effect(frame, alpha: float = 0.0) -> None:
    return


def draw_viewfinder_crosshair(frame, color, thickness: int = 1) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size, gap = 18, 6
    cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)
