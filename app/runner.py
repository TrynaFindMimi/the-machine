import time

import cv2
import mediapipe as mp
import numpy as np
import pygame

from app.registry import TEST_NAMES, TESTS
from app.vision import make_landmarker, make_recognizer
from infrastructure.capture import Camera
from infrastructure.display import Window
from presentation.ui.effects import apply_cctv_effect
from presentation.ui.layout import draw_sidebar
from presentation.ui.theme import SIDEBAR_W

WINDOW_W = 1280
WINDOW_H = 720
CAMERA_W = WINDOW_W - SIDEBAR_W
CAMERA_H = WINDOW_H


def run(mode_idx: int = 0) -> None:
    landmarker = make_landmarker()
    recognizer = make_recognizer()
    cam = Camera(CAMERA_W, CAMERA_H)
    window = Window(WINDOW_W, WINDOW_H, "the-machine")
    clock = pygame.time.Clock()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            name = TEST_NAMES[mode_idx]
            if name == "position":
                results = recognizer.recognize_for_video(mp_img, ts)
            else:
                results = landmarker.detect_for_video(mp_img, ts)
            out, hand_count = TESTS[name](frame, results)
            fps = clock.get_fps()
            apply_cctv_effect(out)
            draw_sidebar(out, name, hand_count, fps)
            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
            canvas[:CAMERA_H, :CAMERA_W] = out
            window.show(canvas)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    if event.key == pygame.K_n:
                        mode_idx = (mode_idx + 1) % len(TEST_NAMES)
            clock.tick(30)
    finally:
        cam.release()
        landmarker.close()
        recognizer.close()
        pygame.quit()
