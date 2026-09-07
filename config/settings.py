from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class WindowSettings:
    width: int = 1280
    height: int = 720
    sidebar_width: int = 220
    title: str = "the-machine"
    target_fps: int = 30

    @property
    def camera_width(self) -> int:
        return self.width - self.sidebar_width

    @property
    def camera_height(self) -> int:
        return self.height


@dataclass(frozen=True, slots=True)
class VisionSettings:
    landmarker_path: str = "models/hand_landmarker.task"
    gesture_path: str = "models/gesture_recognizer.task"
    num_hands_landmarker: int = 20
    num_hands_gesture: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class AudioSettings:
    driver: str = "pulseaudio"
    device: str = "AirPods Max - Find My"
    frequency: int = 44100
    size: int = -16
    channels: int = 2
    buffer: int = 512
    volume: float = 1.0
    auto_find_bluetooth: bool = True


@dataclass(frozen=True, slots=True)
class PerceptronSettings:
    n_samples: int = 24
    margin: float = 0.03
    learning_rate: float = 0.05
    max_epochs: int = 3000
    fingers_together_thresh: float = 0.04
    epoch_budget: int = 200
    point_a_idx: int = 4
    point_b_idx: int = 8


WINDOW: Final = WindowSettings()
VISION: Final = VisionSettings()
AUDIO: Final = AudioSettings()
PERCEPTRON: Final = PerceptronSettings()
