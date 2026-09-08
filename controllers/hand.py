from __future__ import annotations

import pathlib
from typing import Final

import numpy as np
from numpy.typing import NDArray

from config.strings import (
    MUSIC_ACTION_NEXT_SAGA,
    MUSIC_ACTION_NEXT_SONG,
    MUSIC_ACTION_PAUSE,
    MUSIC_ACTION_PLAY,
    MUSIC_ACTION_PREV_SAGA,
    MUSIC_ACTION_PREV_SONG,
)

_MODEL_DIR: Final = pathlib.Path("models")
_WINDOW: Final = 8


class FingerRNN:
    def __init__(self, hidden: int = 8, lr: float = 0.08, seed: int = 0) -> None:
        self.hidden = hidden
        self.lr = lr
        rng = np.random.default_rng(seed)
        h = hidden
        s = 0.6 / np.sqrt(h)
        self.Wx = rng.normal(0.0, s, (h, 1))
        self.Wh = rng.normal(0.0, s, (h, h))
        self.b = np.zeros(h)
        self.Wy = rng.normal(0.0, 0.5, (2, h))
        self.by = np.zeros(2)

    def predict(self, seq: NDArray[np.float64]) -> int:
        h = np.zeros(self.hidden)
        for t in range(seq.shape[0]):
            x = float(seq[t, 0])
            h = np.tanh(self.Wx[:, 0] * x + self.Wh @ h + self.b)
        logits = self.Wy @ h + self.by
        return int(np.argmax(logits))

    def train(self, xs: NDArray[np.float64], ys: NDArray[np.int64], epochs: int = 45) -> None:
        for _ in range(epochs):
            dWx = np.zeros_like(self.Wx)
            dWh = np.zeros_like(self.Wh)
            db = np.zeros_like(self.b)
            dWy = np.zeros_like(self.Wy)
            dby = np.zeros_like(self.by)

            for xi, yi in zip(xs, ys):
                hs = self._forward(xi)
                h_last = hs[-1]
                logits = self.Wy @ h_last + self.by
                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()
                grad_y = probs.copy()
                grad_y[yi] -= 1.0
                dWy += np.outer(grad_y, h_last)
                dby += grad_y
                dh = self.Wy.T @ grad_y
                h_prev = np.zeros(self.hidden)
                for t in range(xi.shape[0]):
                    x = float(xi[t, 0])
                    hh = hs[t]
                    dh_out = dh * (1.0 - hh * hh)
                    dWx[:, 0] += dh_out * x
                    dWh += np.outer(dh_out, h_prev)
                    db += dh_out
                    dh = self.Wh.T @ dh_out
                    h_prev = hh
            n = max(1, len(xs))
            for p, g in [
                (self.Wx, dWx),
                (self.Wh, dWh),
                (self.b, db),
                (self.Wy, dWy),
                (self.by, dby),
            ]:
                p -= self.lr * g / n

    def _forward(self, xi: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        hs: list[NDArray[np.float64]] = []
        h = np.zeros(self.hidden)
        for t in range(xi.shape[0]):
            x = float(xi[t, 0])
            h = np.tanh(self.Wx[:, 0] * x + self.Wh @ h + self.b)
            hs.append(h.copy())
        return hs


class HandController:
    def __init__(self) -> None:
        self.fingers: list[FingerRNN] = [FingerRNN(seed=idx) for idx in range(5)]

    def count(self, window: NDArray[np.float64]) -> int:
        c = 0
        for i, rnn in enumerate(self.fingers):
            seq = window[:, i : i + 1]
            if rnn.predict(seq) == 1:
                c += 1
        return c

    def action(self, count: int) -> str:
        if count == 0:
            return MUSIC_ACTION_PAUSE
        if count == 1:
            return MUSIC_ACTION_PREV_SONG
        if count == 2:
            return MUSIC_ACTION_NEXT_SONG
        if count == 3:
            return MUSIC_ACTION_PREV_SAGA
        if count == 4:
            return MUSIC_ACTION_NEXT_SAGA
        if count == 5:
            return MUSIC_ACTION_PLAY
        return ""

    def save(self) -> None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for i, rnn in enumerate(self.fingers):
            np.savez(
                _MODEL_DIR / f"finger_{i}.npz",
                Wx=rnn.Wx,
                Wh=rnn.Wh,
                b=rnn.b,
                Wy=rnn.Wy,
                by=rnn.by,
            )

    def load(self) -> bool:
        ok = True
        for i, rnn in enumerate(self.fingers):
            p = _MODEL_DIR / f"finger_{i}.npz"
            if not p.is_file():
                ok = False
                continue
            d = np.load(p)
            if "Wx" not in d:
                ok = False
                continue
            rnn.Wx = d["Wx"]
            rnn.Wh = d["Wh"]
            rnn.b = d["b"]
            rnn.Wy = d["Wy"]
            rnn.by = d["by"]
        return ok

    @staticmethod
    def train_all(window: int = _WINDOW) -> HandController:
        hc = HandController()
        rng = np.random.default_rng(7)
        for i, rnn in enumerate(hc.fingers):
            xs: list[NDArray[np.float64]] = []
            ys: list[int] = []
            for p, n in ((0.00, 25), (0.20, 25), (0.42, 25)):
                for _ in range(n):
                    xs.append(HandController._noisy_window(rng, window, p))
                    ys.append(0)
            for p, n in ((0.65, 25), (0.82, 25), (1.00, 25)):
                for _ in range(n):
                    xs.append(HandController._noisy_window(rng, window, p))
                    ys.append(1)
            rnn.train(np.stack(xs), np.asarray(ys, dtype=np.int64), epochs=45)
        hc.save()
        return hc

    @staticmethod
    def _noisy_window(rng: np.random.Generator, window: int, p: float) -> NDArray[np.float64]:
        base = np.where(rng.uniform(0.0, 1.0, (window, 1)) < p, 1.0, -1.0)
        return (base + rng.normal(0.0, 0.12, base.shape)).astype(np.float64)