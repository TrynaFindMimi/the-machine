from __future__ import annotations

import pathlib
import time
from typing import Final

import numpy as np
from numpy.typing import NDArray

from config.settings import PERCEPTRON, VOLUME
from core.perceptron import Perceptron

COMMIT_COOLDOWN_SECONDS: Final = 3.0


def _build_dataset(
    near: float,
    far: float,
    n: int = 60,
    seed: int = 11,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for _ in range(n):
        xs.append([-float(rng.uniform(0.0, near)), 1.0])
        ys.append(-1)
        xs.append([float(rng.uniform(far, 1.0)), 1.0])
        ys.append(1)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


class VolumeController:
    def __init__(self) -> None:
        self.perceptron = Perceptron(
            lr=PERCEPTRON.learning_rate,
            max_epochs=PERCEPTRON.max_epochs,
            n_features=2,
        )
        self._ema: float | None = None
        self._last_joined: bool = False
        self._saved: float | None = None
        self._next_commit: float = 0.0
        self.pending: float | None = None

    def _vol_from_dist(self, d: float) -> float:
        w = self.perceptron.w
        z = float(w[0] * d + w[1])
        z_near = float(w[0] * VOLUME.near + w[1])
        z_far = float(w[0] * VOLUME.far + w[1])
        span = z_far - z_near
        if abs(span) < 1e-9:
            return 0.0
        return float(np.clip((z - z_near) / span, 0.0, 1.0))

    def load_or_train(self) -> None:
        path = pathlib.Path(VOLUME.model_path)
        if path.is_file():
            data = np.load(path)
            if "w" in data and data["w"].shape == (2,):
                self.perceptron.w = data["w"]
                return
        self._train()
        self.save()

    def _train(self) -> None:
        X, y = _build_dataset(VOLUME.near, VOLUME.far)
        self.perceptron.reset()
        self.perceptron.train(X, y)
        if self.perceptron.w[0] < 0:
            self.perceptron.w = -self.perceptron.w

    def save(self) -> None:
        path = pathlib.Path(VOLUME.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.perceptron.w)

    def feed(self, distance: float, joined: bool) -> tuple[float, float | None]:
        self._ema = distance if self._ema is None else VOLUME.smoothing * distance + (1.0 - VOLUME.smoothing) * self._ema
        if joined:
            was_joined = self._last_joined
            self._last_joined = True
            now = time.monotonic()
            if not was_joined and now >= self._next_commit:
                self._saved = self._vol_from_dist(self._ema)
                self.pending = self._saved
                self._next_commit = now + COMMIT_COOLDOWN_SECONDS
        else:
            self._last_joined = False
        if self._last_joined and self._saved is not None:
            preview = self._saved
        else:
            preview = self._vol_from_dist(self._ema)
        return preview, self._saved if self.pending is not None else None

    @property
    def joined(self) -> bool:
        return self._last_joined

    def consume_pending_volume(self) -> float | None:
        volume = self.pending
        self.pending = None
        return volume

    def reset(self) -> None:
        self._ema = None
        self._last_joined = False
        self._saved = None
        self._next_commit = 0.0
        self.pending = None