from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

TIP_IDS: tuple[int, ...] = (4, 8, 12, 16, 20)
PIP_IDS: tuple[int, ...] = (3, 6, 10, 14, 18)
_MCP_MIDDLE: int = 9
_WRIST: int = 0
_THUMB_TIP: int = 4
_THUMB_IP: int = 3
_INDEX_MCP: int = 5
_INDEX_TIP: int = 8
_PINKY_TIP: int = 20

THUMB_EXTEND_THRESH: Final = 0.02
FINGER_EXTEND_THRESH: Final = 0.15


def _finger_feature(hand_landmarks, tip: int, pip: int, scale: float) -> float:
    t = hand_landmarks[tip]
    p = hand_landmarks[pip]
    return float(p.y - t.y) / scale if scale > 0 else 0.0


def _thumb_feature(hand_landmarks, scale: float) -> float:
    tip = hand_landmarks[_THUMB_TIP]
    ip = hand_landmarks[_THUMB_IP]
    idx_mcp = hand_landmarks[_INDEX_MCP]
    if scale <= 0:
        return 0.0
    d_tip = _point_distance(tip, idx_mcp) / scale
    d_ip = _point_distance(ip, idx_mcp) / scale
    return d_tip - d_ip


def hand_scale(hand_landmarks) -> float:
    wrist = hand_landmarks[_WRIST]
    mcp = hand_landmarks[_MCP_MIDDLE]
    return float(max(1e-6, abs(mcp.y - wrist.y) + abs(mcp.x - wrist.x)))


def _point_distance(pa, pb) -> float:
    dx = float(pa.x - pb.x)
    dy = float(pa.y - pb.y)
    return float((dx * dx + dy * dy) ** 0.5)


def _tip_distance(hand_landmarks, a: int, b: int, scale: float) -> float:
    return float(_point_distance(hand_landmarks[a], hand_landmarks[b])) / scale if scale > 0 else 0.0


def index_thumb_distance(hand_landmarks, scale: float) -> float:
    return _tip_distance(hand_landmarks, _INDEX_TIP, _THUMB_TIP, scale)


def pinky_thumb_distance(hand_landmarks, scale: float) -> float:
    return _tip_distance(hand_landmarks, _PINKY_TIP, _THUMB_TIP, scale)


def features_from_landmarks(hand_landmarks, w: int, h: int) -> NDArray[np.float64]:
    scale = hand_scale(hand_landmarks)
    features: list[float] = []
    for tip, pip in zip(TIP_IDS, PIP_IDS):
        if tip == _THUMB_TIP:
            v = _thumb_feature(hand_landmarks, scale)
            thresh = THUMB_EXTEND_THRESH
        else:
            v = _finger_feature(hand_landmarks, tip, pip, scale)
            thresh = FINGER_EXTEND_THRESH
        features.append(1.0 if v > thresh else -1.0)
    features.append(_horizontal_hand_sign(hand_landmarks, scale))
    return np.asarray(features, dtype=np.float64)


def _horizontal_hand_sign(hand_landmarks, scale: float) -> float:
    wrist = hand_landmarks[_WRIST]
    return float((hand_landmarks[_MCP_MIDDLE].y - wrist.y)) / scale if scale > 0 else 0.0


def count_fingers_geometric(hand_landmarks) -> int:
    scale = hand_scale(hand_landmarks)
    count = 0
    for tip, pip in zip(TIP_IDS, PIP_IDS):
        if tip == _THUMB_TIP:
            ext = _thumb_feature(hand_landmarks, scale) > THUMB_EXTEND_THRESH
        else:
            ext = _finger_feature(hand_landmarks, tip, pip, scale) > FINGER_EXTEND_THRESH
        if ext:
            count += 1
    return count