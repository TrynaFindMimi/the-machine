import time


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = instant if self.fps == 0 else (
                self.smoothing * self.fps + (1 - self.smoothing) * instant
            )
        return self.fps
