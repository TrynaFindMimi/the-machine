from __future__ import annotations

import pathlib
from typing import Final

import numpy as np
from numpy.typing import NDArray

from config.strings import MUSIC_ACTION_PAUSE, MUSIC_ACTION_PLAY

_MODEL_DIR: Final = pathlib.Path("models")
_WINDOW: Final = 8


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


class FingerLSTM:
    def __init__(self, hidden: int = 8, lr: float = 0.08) -> None:
        self.hidden = hidden
        self.lr = lr
        h = hidden
        s = 0.6 / np.sqrt(h)
        self.Wi = np.random.normal(0.0, s, (h, 1))
        self.Ui = np.random.normal(0.0, s, (h, h))
        self.bi = np.zeros(h)
        self.Wf = np.random.normal(0.0, s, (h, 1))
        self.Uf = np.random.normal(0.0, s, (h, h))
        self.bf = np.ones(h)
        self.Wo = np.random.normal(0.0, s, (h, 1))
        self.Uo = np.random.normal(0.0, s, (h, h))
        self.bo = np.zeros(h)
        self.Wg = np.random.normal(0.0, s, (h, 1))
        self.Ug = np.random.normal(0.0, s, (h, h))
        self.bg = np.zeros(h)
        self.Wy = np.random.normal(0.0, 0.5, (2, h))
        self.by = np.zeros(2)

    def predict(self, seq: NDArray[np.float64]) -> int:
        h = np.zeros(self.hidden)
        c = np.zeros(self.hidden)
        for t in range(seq.shape[0]):
            x = float(seq[t, 0])
            i = _sigmoid(self.Wi[:, 0] * x + self.Ui @ h + self.bi)
            f = _sigmoid(self.Wf[:, 0] * x + self.Uf @ h + self.bf)
            o = _sigmoid(self.Wo[:, 0] * x + self.Uo @ h + self.bo)
            g = np.tanh(self.Wg[:, 0] * x + self.Ug @ h + self.bg)
            c = f * c + i * g
            h = o * np.tanh(c)
        logits = self.Wy @ h + self.by
        return int(np.argmax(logits))

    def train(self, xs: NDArray[np.float64], ys: NDArray[np.int64], epochs: int = 150) -> None:
        for _ in range(epochs):
            dWi = np.zeros_like(self.Wi)
            dUi = np.zeros_like(self.Ui)
            dbi = np.zeros_like(self.bi)
            dWf = np.zeros_like(self.Wf)
            dUf = np.zeros_like(self.Uf)
            dbf = np.zeros_like(self.bf)
            dWo = np.zeros_like(self.Wo)
            dUo = np.zeros_like(self.Uo)
            dbo = np.zeros_like(self.bo)
            dWg = np.zeros_like(self.Wg)
            dUg = np.zeros_like(self.Ug)
            dbg = np.zeros_like(self.bg)
            dWy = np.zeros_like(self.Wy)
            dby = np.zeros_like(self.by)

            for xi, yi in zip(xs, ys):
                hs, cs, iss, fs, os, gs = self._forward(xi)
                h_last = hs[-1]
                logits = self.Wy @ h_last + self.by
                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()
                grad_y = probs.copy()
                grad_y[yi] -= 1.0
                dWy += np.outer(grad_y, h_last)
                dby += grad_y
                dh = self.Wy.T @ grad_y
                dc = np.zeros_like(h_last)
                for t in reversed(range(xi.shape[0])):
                    x = float(xi[t, 0])
                    h = hs[t]
                    c = cs[t]
                    c_prev = cs[t - 1] if t > 0 else np.zeros_like(c)
                    h_prev = hs[t - 1] if t > 0 else np.zeros_like(h)
                    i = iss[t]
                    f = fs[t]
                    o = os[t]
                    g = gs[t]
                    tanh_c = np.tanh(c)
                    dh_total = dh
                    do = dh_total * tanh_c * o * (1.0 - o)
                    dc_total = dc + dh_total * o * (1.0 - tanh_c ** 2)
                    di = dc_total * g * i * (1.0 - i)
                    df = dc_total * c_prev * f * (1.0 - f)
                    dg = dc_total * i * (1.0 - g ** 2)
                    dWg[:, 0] += dg * x
                    dUg += np.outer(dg, h_prev)
                    dbg += dg
                    dWi[:, 0] += di * x
                    dUi += np.outer(di, h_prev)
                    dbi += di
                    dWf[:, 0] += df * x
                    dUf += np.outer(df, h_prev)
                    dbf += df
                    dWo[:, 0] += do * x
                    dUo += np.outer(do, h_prev)
                    dbo += do
                    dh = (
                        self.Ui.T @ di
                        + self.Uf.T @ df
                        + self.Uo.T @ do
                        + self.Ug.T @ dg
                    )
                    dc = dc_total * f
            n = max(1, len(xs))
            for p, g in [
                (self.Wi, dWi),
                (self.Ui, dUi),
                (self.bi, dbi),
                (self.Wf, dWf),
                (self.Uf, dUf),
                (self.bf, dbf),
                (self.Wo, dWo),
                (self.Uo, dUo),
                (self.bo, dbo),
                (self.Wg, dWg),
                (self.Ug, dUg),
                (self.bg, dbg),
                (self.Wy, dWy),
                (self.by, dby),
            ]:
                p -= self.lr * g / n


    def _forward(self, xi: NDArray[np.float64]):
        hs: list[NDArray[np.float64]] = []
        cs: list[NDArray[np.float64]] = []
        iss: list[NDArray[np.float64]] = []
        fs: list[NDArray[np.float64]] = []
        os: list[NDArray[np.float64]] = []
        gs: list[NDArray[np.float64]] = []
        h = np.zeros(self.hidden)
        c = np.zeros(self.hidden)
        for t in range(xi.shape[0]):
            x = float(xi[t, 0])
            i = _sigmoid(self.Wi[:, 0] * x + self.Ui @ h + self.bi)
            f = _sigmoid(self.Wf[:, 0] * x + self.Uf @ h + self.bf)
            o = _sigmoid(self.Wo[:, 0] * x + self.Uo @ h + self.bo)
            g = np.tanh(self.Wg[:, 0] * x + self.Ug @ h + self.bg)
            c = f * c + i * g
            h = o * np.tanh(c)
            hs.append(h.copy())
            cs.append(c.copy())
            iss.append(i)
            fs.append(f)
            os.append(o)
            gs.append(g)
        return hs, cs, iss, fs, os, gs


class HandController:
    def __init__(self) -> None:
        self.fingers: list[FingerLSTM] = [FingerLSTM() for _ in range(5)]

    def count(self, window: NDArray[np.float64]) -> int:
        c = 0
        for i, lstm in enumerate(self.fingers):
            seq = window[:, i : i + 1]
            if lstm.predict(seq) == 1:
                c += 1
        return c

    def action(self, count: int) -> str:
        if count == 0:
            return MUSIC_ACTION_PAUSE
        if count >= 4:
            return MUSIC_ACTION_PLAY
        return ""

    def save(self) -> None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for i, lstm in enumerate(self.fingers):
            np.savez(
                _MODEL_DIR / f"finger_{i}.npz",
                Wi=lstm.Wi,
                Ui=lstm.Ui,
                bi=lstm.bi,
                Wf=lstm.Wf,
                Uf=lstm.Uf,
                bf=lstm.bf,
                Wo=lstm.Wo,
                Uo=lstm.Uo,
                bo=lstm.bo,
                Wg=lstm.Wg,
                Ug=lstm.Ug,
                bg=lstm.bg,
                Wy=lstm.Wy,
                by=lstm.by,
            )

    def load(self) -> bool:
        ok = True
        for i, lstm in enumerate(self.fingers):
            p = _MODEL_DIR / f"finger_{i}.npz"
            if not p.is_file():
                ok = False
                continue
            d = np.load(p)
            if "Wi" not in d:
                ok = False
                continue
            lstm.Wi = d["Wi"]
            lstm.Ui = d["Ui"]
            lstm.bi = d["bi"]
            lstm.Wf = d["Wf"]
            lstm.Uf = d["Uf"]
            lstm.bf = d["bf"]
            lstm.Wo = d["Wo"]
            lstm.Uo = d["Uo"]
            lstm.bo = d["bo"]
            lstm.Wg = d["Wg"]
            lstm.Ug = d["Ug"]
            lstm.bg = d["bg"]
            lstm.Wy = d["Wy"]
            lstm.by = d["by"]
        return ok

    @staticmethod
    def train_all(window: int = _WINDOW) -> HandController:
        hc = HandController()
        rng = np.random.default_rng(7)
        for i, lstm in enumerate(hc.fingers):
            xs: list[NDArray[np.float64]] = []
            ys: list[int] = []
            for label in (0, 1):
                base = 1.0 if label == 1 else -1.0
                for _ in range(60):
                    seq = np.full((window, 1), base, dtype=np.float64)
                    seq += rng.normal(0.0, 0.15, seq.shape)
                    xs.append(seq)
                    ys.append(label)
            lstm.train(np.stack(xs), np.asarray(ys, dtype=np.int64), epochs=60)
        hc.save()
        return hc
