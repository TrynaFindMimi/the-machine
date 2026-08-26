import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from tests.grid_test import draw as grid_test
from tests.line_test import draw as line_test
from tests.hand_test import draw as hand_test
from utils.text import USAGE, WINDOW

MODEL_PATH = "models/hand_landmarker.task"

TESTS = {
    "hand": hand_test,
    "grid": grid_test,
    "line": line_test,
}
TEST_NAMES = list(TESTS)


def make_landmarker() -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def run(mode_idx: int = 0) -> None:
    landmarker = make_landmarker()
    vc = cv2.VideoCapture(0)

    pygame.init()
    pygame.display.set_caption(WINDOW)
    screen = None
    clock = pygame.time.Clock()

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        name = TEST_NAMES[mode_idx]
        out = TESTS[name](frame, results)

        h, w = out.shape[:2]
        if screen is None:
            screen = pygame.display.set_mode((w, h))
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_out, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                vc.release()
                landmarker.close()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    vc.release()
                    landmarker.close()
                    pygame.quit()
                    return
                if event.key == pygame.K_n:
                    mode_idx = (mode_idx + 1) % len(TEST_NAMES)

        clock.tick(30)

    vc.release()
    landmarker.close()
    pygame.quit()


def main() -> None:
    start = 0
    if len(sys.argv) > 1:
        if sys.argv[1] not in TESTS:
            names = ", ".join(TEST_NAMES)
            print(USAGE.format(names))
            raise SystemExit(1)
        start = TEST_NAMES.index(sys.argv[1])
    run(start)


if __name__ == "__main__":
    main()
