import math

import cv2
import numpy as np

BLUE = (255, 60, 20)

N_SAMPLES = 24
MARGIN = 0.03
LR = 0.05
MAX_EPOCHS = 3000
FINGERS_TOGETHER_THRESH = 0.04


class Perceptron:
    def __init__(self, lr: float = LR, max_epochs: int = MAX_EPOCHS):
        self.w = np.zeros(3, dtype=float)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)

    def predict(self, x) -> int:
        return 1 if float(np.dot(self.w, x)) >= 0 else -1

    def train(self, X, y) -> int:
        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != yi:
                    self.w += self.lr * yi * np.asarray(xi, dtype=float)
                    errors += 1
            if errors == 0:
                return epoch
        return self.max_epochs

    def train_budget(self, X, y, budget: int) -> int:
        trained = 0
        for _ in range(budget):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != yi:
                    self.w += self.lr * yi * np.asarray(xi, dtype=float)
                    errors += 1
            trained += 1
            if errors == 0:
                return trained
        return trained


def build_dataset(p5, p9):
    seg = p9 - p5
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        return None, None
    perp = np.array([-seg[1], seg[0]], dtype=float) / length

    t = np.linspace(0.0, 1.0, N_SAMPLES).reshape(-1, 1)
    pos = p5 + t * seg
    neg = pos + MARGIN * perp

    X = np.hstack([np.vstack([pos, neg]), np.ones((len(pos) + len(neg), 1))])
    y = np.array([1] * len(pos) + [-1] * len(neg))
    return X, y


def boundary_points(w_vec, w_px: int, h_px: int):
    a, b, c = w_vec
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        return None
    pts = []
    try:
        if abs(b) >= abs(a):
            for x in (0.0, 1.0):
                pts.append((x * w_px, (-c - a * x) / b * h_px))
        else:
            for y in (0.0, 1.0):
                pts.append(((-c - b * y) / a * w_px, y * h_px))
    except ZeroDivisionError:
        return None
    (x1, y1), (x2, y2) = pts
    if math.hypot(x2 - x1, y2 - y1) < 1e-6:
        return None
    return (int(x1), int(y1)), (int(x2), int(y2))


def draw_line(frame, p1, p2, color=BLUE, thickness: int = 2):
    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), color, thickness, cv2.LINE_AA)
    return (tuple(map(int, p1)), tuple(map(int, p2)))
