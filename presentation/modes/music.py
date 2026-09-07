from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BLACK, WHITE
from config.strings import (
    MUSIC_ACTION_FMT,
    MUSIC_COUNT_FMT,
    MUSIC_FINGERS_FMT,
    MUSIC_HAND_L,
    MUSIC_HAND_R,
    MUSIC_SAGA_FMT,
    MUSIC_SONG_FMT,
    MUSIC_STATE_FMT,
    MUSIC_TITLE,
    MUSIC_UNKNOWN,
    MUSIC_VOLUME_FMT,
    MUSIC_VOLUME_SAVED,
)
from controllers.gestures import MusicGestureController
from controllers.volume import VolumeController
from core.fingers import (
    features_from_landmarks,
    hand_scale,
    pinky_ring_distance,
    ring_thumb_distance,
)
from core.handedness import LEFT, RIGHT, get_handedness
from core.results import count_hands, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

_gesture: MusicGestureController | None = None
_volume: VolumeController | None = None
_saga_name: str = MUSIC_UNKNOWN
_song_name: str = MUSIC_UNKNOWN
_track_text: str = MUSIC_UNKNOWN
_player_state: str = MUSIC_UNKNOWN


def _get_gesture() -> MusicGestureController:
    global _gesture
    if _gesture is not None:
        return _gesture
    gc = MusicGestureController()
    gc.load_or_train()
    _gesture = gc
    return _gesture


def _get_volume() -> VolumeController:
    global _volume
    if _volume is not None:
        return _volume
    vc = VolumeController()
    vc.load_or_train()
    _volume = vc
    return _volume


def _put_text_box(
    frame: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    font_scale: float,
    thickness: int,
    color: tuple[int, int, int] = WHITE,
    bg: tuple[int, int, int] = BLACK,
    pad_x: int = 6,
    pad_y: int = 4,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    x1, y1 = x - pad_x, y - th - pad_y
    x2, y2 = x + tw + pad_x, y + baseline + pad_y // 2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def reset_state() -> None:
    _get_gesture().reset()
    _get_volume().reset()


def set_context(saga_name: str, song_name: str, track_text: str, player_state: str) -> None:
    global _saga_name, _song_name, _track_text, _player_state
    _saga_name = saga_name
    _song_name = song_name
    _track_text = track_text
    _player_state = player_state


def consume_pending_action() -> str:
    return _get_gesture().consume_pending_action()


def consume_pending_volume() -> float | None:
    return _get_volume().consume_pending_volume()


def _bbox_tuple(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _draw_volume_ui(
    frame: NDArray[np.uint8],
    preview: float,
    pending: float | None,
    bbox: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    vc = _get_volume()
    if pending is not None:
        text = MUSIC_VOLUME_SAVED.format(round(preview * 100))
    else:
        text = MUSIC_VOLUME_FMT.format(round(preview * 100))
    _put_text_box(frame, text, (max(0, x1), max(18, y2 + 8)), 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    bar_w = 80
    bar_h = 6
    bx, by = max(0, x1), max(18, y2 + 30)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), BLACK, -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * preview), by + bar_h), WHITE, -1)


def draw(frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]:
    _put_text_box(frame, spaced(MUSIC_TITLE), (10, 22), 0.70, 2, WHITE, BLACK)
    h, w = frame.shape[:2]
    hand_count = count_hands(results)
    landmarks = results.hand_landmarks or []
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
        reset_state()
        return frame, hand_count

    right_idx = next(
        (i for i in range(len(landmarks)) if get_handedness(results, i) == RIGHT),
        None,
    )
    left_idx = next(
        (i for i in range(len(landmarks)) if get_handedness(results, i) == LEFT),
        None,
    )
    if right_idx is None and left_idx is None:
        reset_state()
        return frame, hand_count

    if left_idx is not None:
        left_lm = landmarks[left_idx]
        left_pts = to_pixel_points(left_lm, w, h)
        draw_bbox(frame, left_pts, WHITE, thickness=2)
        draw_skeleton(frame, left_pts)
        vc = _get_volume()
        scale = hand_scale(left_lm)
        distance = ring_thumb_distance(left_lm, scale)
        joined = pinky_ring_distance(left_lm, scale) < 0.06
        preview, pending = vc.feed(distance, joined)
        _draw_volume_ui(frame, preview, pending, _bbox_tuple(left_pts))
        lx = min(p[0] for p in left_pts)
        ly = min(p[1] for p in left_pts)
        _put_text_box(frame, MUSIC_HAND_L, (max(0, lx), max(18, ly - 30)), 0.6, 2, WHITE, BLACK, pad_x=5, pad_y=3)

    if right_idx is not None:
        hand_landmarks = landmarks[right_idx]
        pts = to_pixel_points(hand_landmarks, w, h)
        draw_bbox(frame, pts, WHITE, thickness=2)
        draw_skeleton(frame, pts)

        feat = features_from_landmarks(hand_landmarks, w, h)
        count, action = _get_gesture().feed(feat)
        rx, ry = min(p[0] for p in pts) - 10, min(p[1] for p in pts) - 10
        _put_text_box(frame, MUSIC_HAND_R, (max(0, rx), max(18, ry - 10)), 0.6, 2, WHITE, BLACK, pad_x=5, pad_y=3)
    else:
        _get_gesture().reset()
        count, action = -1, ""

    _put_text_box(frame, MUSIC_SAGA_FMT.format(_saga_name), (10, 70), 0.55, 2, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_SONG_FMT.format(_song_name), (10, 100), 0.45, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_COUNT_FMT.format(_track_text), (10, 124), 0.45, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_STATE_FMT.format(_player_state), (10, 148), 0.55, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_FINGERS_FMT.format(count if count >= 0 else MUSIC_UNKNOWN), (10, 172), 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_ACTION_FMT.format(action if action else MUSIC_UNKNOWN), (10, 196), 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3)

    return frame, hand_count
