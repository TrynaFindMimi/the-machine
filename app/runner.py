"""app/runner.py — loop principal orquestando captura → visión → modo → UI."""

from __future__ import annotations

import contextlib
import time
from typing import Final

import cv2
import mediapipe as mp
import numpy as np
import pygame

from app.registry import TEST_NAMES, TESTS
from app.vision import make_landmarker, make_recognizer
from config.settings import WINDOW, VisionSettings, WindowSettings
from infrastructure.capture import Camera
from infrastructure.display import Window
from presentation.ui.effects import apply_cctv_effect
from presentation.ui.layout import draw_sidebar

WINDOW_W: Final = WINDOW.width
WINDOW_H: Final = WINDOW.height
CAMERA_W: Final = WINDOW.camera_width
CAMERA_H: Final = WINDOW.camera_height


def _handle_events(mode_idx: int) -> tuple[bool, int]:
    """Procesa eventos pygame. Returns (should_quit, new_mode_idx)."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, mode_idx
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True, mode_idx
            if event.key == pygame.K_n:
                return False, (mode_idx + 1) % len(TEST_NAMES)
    return False, mode_idx


def run(
    mode_idx: int = 0,
    window_cfg: WindowSettings = WINDOW,
    vision_cfg: VisionSettings | None = None,
) -> None:
    """Ejecuta el loop principal. Gestiona ciclo de vida de recursos con contextlib."""
    # Validación temprana
    if not 0 <= mode_idx < len(TEST_NAMES):
        raise ValueError(f"mode_idx fuera de rango: {mode_idx}")

    landmarker = make_landmarker(vision_cfg) if vision_cfg else make_landmarker()
    recognizer = make_recognizer(vision_cfg) if vision_cfg else make_recognizer()
    cam = Camera(window_cfg.camera_width, window_cfg.camera_height)
    window = Window(window_cfg.width, window_cfg.height, window_cfg.title)
    clock = pygame.time.Clock()

    with contextlib.ExitStack() as stack:
        # Garantiza liberación incluso si hay excepción en inicialización
        stack.callback(cam.release)
        stack.callback(landmarker.close)
        stack.callback(recognizer.close)
        stack.callback(pygame.quit)

        current = mode_idx
        while True:
            frame = cam.read()
            if frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)

            name = TEST_NAMES[current]
            if name == "position":
                results = recognizer.recognize_for_video(mp_img, ts)
            else:
                results = landmarker.detect_for_video(mp_img, ts)

            out, hand_count = TESTS[name](frame, results)
            fps = clock.get_fps()
            apply_cctv_effect(out)
            draw_sidebar(out, name, hand_count, fps)

            canvas = np.zeros((window_cfg.height, window_cfg.width, 3), dtype=np.uint8)
            canvas[: window_cfg.camera_height, : window_cfg.camera_width] = out
            window.show(canvas)

            should_quit, current = _handle_events(current)
            if should_quit:
                break
            clock.tick(window_cfg.target_fps)
