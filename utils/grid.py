"""Capa de presentación - utilidades de grilla y cuadrantes.

Arquitectura en capas:
- utils/grid.py pertenece a la capa de presentación (dibujo)
- Separa la lógica de ejes/coordenadas del orquestador tests/grid_test.py
"""

import cv2

from utils.hand import FONT, WHITE, draw_label

AXIS_COLOR = (120, 120, 120)
GRID_STEP = 60
QUADRANT_COLOR = (90, 90, 90)


def draw_axes(frame) -> None:
    """Dibuja ejes X/Y centrados formando 4 cuadrantes con origen 0,0 al centro."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.line(frame, (cx, 0), (cx, h), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.line(frame, (0, cy), (w, cy), AXIS_COLOR, 1, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), 4, WHITE, -1, cv2.LINE_AA)
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), WHITE, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "0,0", (cx + 12, cy - 10), FONT, 0.6, WHITE, 1, cv2.LINE_AA)

    for txt, pos in [("X", (w - 18, cy - 10)), ("Y", (cx + 10, 16))]:
        cv2.putText(frame, txt, pos, FONT, 0.8, AXIS_COLOR, 1, cv2.LINE_AA)


def draw_grid_landmarks(frame, pts) -> None:
    """Dibuja landmarks blancos con etiqueta (x,y) redondeada a 0 decimales."""
    h, w = frame.shape[:2]
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, WHITE, -1, cv2.LINE_AA)
        label = f"({int(round(px))},{int(round(py))})"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        lx = px + 8 if px + tw + 12 < w else px - tw - 8
        ly = py - 8 if py - 8 > th + 4 else py + th + 12
        cv2.putText(frame, label, (lx, ly), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
