import cv2
import numpy as np

from utils.text import KEYBIND_HINT, ROOT_PROMPT

ROOT_BG = (8, 12, 8)
ROOT_GREEN = (60, 255, 80)
ROOT_AMBER = (40, 180, 255)
ROOT_BLUE = (255, 90, 20)
ROOT_DIM = (35, 55, 35)
ROOT_RED = (40, 40, 220)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


def apply_root_overlay(frame, alpha: float = 0.18) -> None:
    h, w = frame.shape[:2]
    overlay = np.full_like(frame, ROOT_BG)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for y in range(0, h, 4):
        cv2.line(frame, (0, y), (w, y), (0, 0, 0), 1, cv2.LINE_AA)
        frame[y, :] = (frame[y, :].astype(np.int16) * 0.92).astype(np.uint8)
    vignette = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(vignette, (w // 2, h // 2), (w, h), 0, 0, 360, 255, -1)
    vignette = cv2.GaussianBlur(vignette, (51, 51), 0)
    vignette = vignette.astype(float) / 255.0 * 0.35 + 0.65
    for c in range(3):
        frame[:, :, c] = (frame[:, :, c].astype(float) * vignette).astype(np.uint8)


def draw_root_header(frame, title: str, fps: float | None = None) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.line(frame, (0, 28), (w, 28), ROOT_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, ROOT_PROMPT, (8, 18), FONT, 0.85, ROOT_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, title.lower(), (w - 220, 18), FONT, 0.75, ROOT_DIM, 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"FPS {fps:05.1f}", (w - 90, 18), FONT, 0.7, ROOT_AMBER, 1, cv2.LINE_AA)


def draw_root_footer(frame, text: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 18), (w, h), (0, 0, 0), -1)
    cv2.line(frame, (0, h - 18), (w, h - 18), ROOT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (8, h - 6), FONT, 0.6, ROOT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, KEYBIND_HINT, (w - 110, h - 6), FONT, 0.6, ROOT_GREEN, 1, cv2.LINE_AA)
