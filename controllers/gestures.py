from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from controllers.hand import HandController

WINDOW_LEN: Final = 8
AGREE_REQUIRED: Final = 4
COOLDOWN_FRAMES: Final = 12


class MusicGestureController:
    def __init__(self, window: int = WINDOW_LEN) -> None:
        self.window_len = window
        self.hand = HandController()
        self.window: list[NDArray[np.float64]] = []
        self.agreement: int = 0
        self.last_count: int = -1
        self.last_action: int = -1
        self.cooldown: int = 0
        self.pending: str = ""

    def load_or_train(self) -> None:
        if not self.hand.load():
            self.hand = HandController.train_all(window=self.window_len)

    def reset(self) -> None:
        self.window = []
        self.agreement = 0
        self.last_count = -1
        self.last_action = -1
        self.cooldown = 0
        self.pending = ""

    def consume_pending_action(self) -> str:
        action = self.pending
        self.pending = ""
        return action

    def _apply_action(self, count: int) -> str:
        if count == self.last_action or self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - 1)
            return ""
        self.cooldown = COOLDOWN_FRAMES
        self.last_action = count
        action = self.hand.action(count)
        if action:
            self.pending = action
        return action

    def feed(self, feat: NDArray[np.float64]) -> tuple[int, str]:
        self.window.append(feat)
        if len(self.window) > self.window_len:
            self.window = self.window[-self.window_len:]
        if len(self.window) < self.window_len:
            return -1, ""
        window_np = np.stack(self.window, axis=0)
        count = self.hand.count(window_np)
        if count == self.last_count:
            self.agreement += 1
        else:
            self.agreement = 1
            self.last_count = count
        if self.agreement >= AGREE_REQUIRED:
            return count, self._apply_action(count)
        return count, ""