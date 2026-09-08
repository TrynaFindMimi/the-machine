from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

TIP_IDS: tuple[int, ...] = (4, 8, 12, 16, 20)
PIP_IDS: tuple[int, ...] = (3, 6, 10, 14, 18)
MCP_IDS: tuple[int, ...] = (2, 5, 9, 13, 17)
_MCP_MIDDLE: int = 9
_WRIST: int = 0
_THUMB_TIP: int = 4
_INDEX_MCP: int = 5
_INDEX_TIP: int = 8
_PINKY_TIP: int = 20

EXTEND_THRESHOLDS: Final[tuple[float, ...]] = (0.50, 0.45, 0.45, 0.35, 0.45)
THUMB_EXTEND_MIN_DIST: Final = 0.35


def _finger_extension(hand_landmarks, mcp: int, pip: int, tip: int) -> float:
    l_mcp = hand_landmarks[mcp]
    l_pip = hand_landmarks[pip]
    l_tip = hand_landmarks[tip]
    v1 = (float(l_pip.x - l_mcp.x), float(l_pip.y - l_mcp.y))
    v2 = (float(l_tip.x - l_pip.x), float(l_tip.y - l_pip.y))
    n1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
    n2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    return float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))


def _thumb_extended(hand_landmarks) -> bool:
    straight = _finger_extension(hand_landmarks, 2, 3, 4)
    if straight <= EXTEND_THRESHOLDS[0]:
        return False
    idx_len = _point_distance(hand_landmarks[5], hand_landmarks[6])
    if idx_len <= 1e-6:
        return False
    d_tip = _point_distance(hand_landmarks[_THUMB_TIP], hand_landmarks[_INDEX_MCP])
    d_ip = _point_distance(hand_landmarks[3], hand_landmarks[_INDEX_MCP])
    return (d_tip - d_ip) / idx_len > 0.08


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
    features: list[float] = []
    for i, (mcp, pip, tip, thresh) in enumerate(zip(MCP_IDS, PIP_IDS, TIP_IDS, EXTEND_THRESHOLDS)):
        if i == 0:
            ext = _thumb_extended(hand_landmarks)
        else:
            ext = _finger_extension(hand_landmarks, mcp, pip, tip) > thresh
        features.append(1.0 if ext else -1.0)
    features.append(_horizontal_hand_sign(hand_landmarks))
    return np.asarray(features, dtype=np.float64)


def _horizontal_hand_sign(hand_landmarks, scale: float | None = None) -> float:
    wrist = hand_landmarks[_WRIST]
    if scale is None:
        scale = hand_scale(hand_landmarks)
    return float((hand_landmarks[_MCP_MIDDLE].y - wrist.y)) / scale if scale > 0 else 0.0


def count_fingers_geometric(hand_landmarks) -> int:
    count = 0
    for i, (mcp, pip, tip, thresh) in enumerate(zip(MCP_IDS, PIP_IDS, TIP_IDS, EXTEND_THRESHOLDS)):
        if i == 0:
            ext = _thumb_extended(hand_landmarks)
        else:
            ext = _finger_extension(hand_landmarks, mcp, pip, tip) > thresh
        if ext:
            count += 1
    return count