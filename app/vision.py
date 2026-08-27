from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

LANDMARKER_PATH = "models/hand_landmarker.task"
GESTURE_PATH = "models/gesture_recognizer.task"


def make_landmarker() -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=20,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def make_recognizer() -> GestureRecognizer:
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=GESTURE_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return GestureRecognizer.create_from_options(options)
