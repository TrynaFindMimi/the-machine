import cv2

from utils.hand import FONT, WHITE


def draw_grid_landmarks(frame, pts) -> None:
    h, w = frame.shape[:2]
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, WHITE, -1, cv2.LINE_AA)
        label = f"({int(round(px))},{int(round(py))})"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        lx = px + 8 if px + tw + 12 < w else px - tw - 8
        ly = py - 8 if py - 8 > th + 4 else py + th + 12
        cv2.putText(frame, label, (lx, ly), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
