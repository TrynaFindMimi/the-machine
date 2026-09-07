from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from config.palette import BLACK, WHITE
from config.strings import (
    MUSIC_ACTION_FMT,
    MUSIC_COUNT_FMT,
    MUSIC_FINGERS_FMT,
    MUSIC_HAND_R,
    MUSIC_SAGA_FMT,
    MUSIC_SONG_FMT,
    MUSIC_STATE_FMT,
    MUSIC_TITLE,
    MUSIC_UNKNOWN,
)
from controllers.gestures import MusicGestureController
from core.fingers import features_from_landmarks
from core.handedness import RIGHT, get_handedness
from core.results import count_hands, to_pixel_points
from presentation.ui.drawing import draw_bbox, draw_skeleton, spaced
from presentation.ui.effects import draw_viewfinder_crosshair
from presentation.ui.theme import FONT

_gesture: MusicGestureController | None = None
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


def set_context(saga_name: str, song_name: str, track_text: str, player_state: str) -> None:
    global _saga_name, _song_name, _track_text, _player_state
    _saga_name = saga_name
    _song_name = song_name
    _track_text = track_text
    _player_state = player_state


def consume_pending_action() -> str:
    return _get_gesture().consume_pending_action()


def draw(frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]:
    _put_text_box(frame, spaced(MUSIC_TITLE), (10, 22), 0.70, 2, WHITE, BLACK)
    h, w = frame.shape[:2]
    hand_count = count_hands(results)
    if hand_count == 0:
        draw_viewfinder_crosshair(frame, WHITE)
        reset_state()
        return frame, hand_count

    landmarks = results.hand_landmarks or []
    right_idx = next(
        (i for i in range(len(landmarks)) if get_handedness(results, i) == RIGHT),
        None,
    )

    if right_idx is None:
        reset_state()
        return frame, hand_count

    hand_landmarks = landmarks[right_idx]
    pts = to_pixel_points(hand_landmarks, w, h)
    draw_bbox(frame, pts, WHITE, thickness=2)
    draw_skeleton(frame, pts)

    feat = features_from_landmarks(hand_landmarks, w, h)
    count, action = _get_gesture().feed(feat)

    _put_text_box(frame, MUSIC_SAGA_FMT.format(_saga_name), (10, 70), 0.55, 2, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_SONG_FMT.format(_song_name), (10, 100), 0.45, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_COUNT_FMT.format(_track_text), (10, 124), 0.45, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_STATE_FMT.format(_player_state), (10, 148), 0.55, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_FINGERS_FMT.format(count if count >= 0 else MUSIC_UNKNOWN), (10, 172), 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3)
    _put_text_box(frame, MUSIC_ACTION_FMT.format(action if action else MUSIC_UNKNOWN), (10, 196), 0.5, 1, WHITE, BLACK, pad_x=5, pad_y=3)

    x1, y1 = (min(p[0] for p in pts) - 10, min(p[1] for p in pts) - 10)
    _put_text_box(frame, MUSIC_HAND_R, (max(0, x1), max(18, y1 - 10)), 0.6, 2, WHITE, BLACK, pad_x=5, pad_y=3)
    return frame, hand_count
