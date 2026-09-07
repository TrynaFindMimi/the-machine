from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TIP_IDS: tuple[int, ...] = (4, 8, 12, 16, 20)
PIP_IDS: tuple[int, ...] = (3, 6, 10, 14, 18)
_MCP_MIDDLE: int = 9
_WRIST: int = 0
_THUMB_TIP: int = 4
_RING_TIP: int = 16
_PINKY_TIP: int = 20


def _finger_feature(hand_landmarks, tip: int, pip: int, scale: float) -> float:
    t = hand_landmarks[tip]
    p = hand_landmarks[pip]
    if tip == 4:
        dx = float(t.x - p.x)
        dy = float(t.y - p.y)
        return float((dx * dx + dy * dy) ** 0.5) / scale if scale > 0 else 0.0
    return float(p.y - t.y) / scale if scale > 0 else 0.0


def hand_scale(hand_landmarks) -> float:
    wrist = hand_landmarks[_WRIST]
    mcp = hand_landmarks[_MCP_MIDDLE]
    return float(max(1e-6, abs(mcp.y - wrist.y) + abs(mcp.x - wrist.x)))


def _tip_distance(hand_landmarks, a: int, b: int, scale: float) -> float:
    pa = hand_landmarks[a]
    pb = hand_landmarks[b]
    dx = float(pa.x - pb.x)
    dy = float(pa.y - pb.y)
    return float((dx * dx + dy * dy) ** 0.5) / scale if scale > 0 else 0.0


def ring_thumb_distance(hand_landmarks, scale: float) -> float:
    return _tip_distance(hand_landmarks, _RING_TIP, _THUMB_TIP, scale)


def pinky_ring_distance(hand_landmarks, scale: float) -> float:
    return _tip_distance(hand_landmarks, _PINKY_TIP, _RING_TIP, scale)


def features_from_landmarks(hand_landmarks, w: int, h: int) -> NDArray[np.float64]:
    scale = hand_scale(hand_landmarks)
    features: list[float] = []
    for tip, pip in zip(TIP_IDS, PIP_IDS):
        v = _finger_feature(hand_landmarks, tip, pip, scale)
        thresh = 0.25 if tip == 4 else 0.15
        ext = v > thresh
        features.append(1.0 if ext else -1.0)
    features.append(_horizontal_hand_sign(hand_landmarks, scale))
    return np.asarray(features, dtype=np.float64)


def _horizontal_hand_sign(hand_landmarks, scale: float) -> float:
    wrist = hand_landmarks[_WRIST]
    return float((hand_landmarks[_MCP_MIDDLE].y - wrist.y)) / scale if scale > 0 else 0.0


def count_fingers_geometric(hand_landmarks) -> int:
    scale = hand_scale(hand_landmarks)
    count = 0
    for tip, pip in zip(TIP_IDS, PIP_IDS):
        thresh = 0.25 if tip == 4 else 0.15
        if _finger_feature(hand_landmarks, tip, pip, scale) > thresh:
            count += 1
    return count