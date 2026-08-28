"""
core/handedness.py — dominio puro para distinguir mano izquierda/derecha.

MediaPipe devuelve `handedness` como categoría "Left"/"Right" desde el
punto de vista del modelo. Como `infrastructure/capture.Camera.read()` hace
`cv2.flip(frame, 1)` (espejo), la etiqueta está invertida para el usuario.
Este módulo centraliza esa corrección y evita que `core/results.py` mezcle
responsabilidades (conversión geométrica vs. semántica de lateralidad).

Capa: core → sin dependencias de cv2/pygame/presentation. Solo stdlib.
Importado por `presentation/modes/*`.
"""

from __future__ import annotations

from typing import Final, Protocol, Sequence


class _Category(Protocol):
    category_name: str


class _Results(Protocol):
    handedness: Sequence[Sequence[_Category]] | None


LEFT: Final = "Left"
RIGHT: Final = "Right"
UNKNOWN: Final = "?"


def normalize_handedness(raw_label: str) -> str:
    """Corrige el flip horizontal: Left del modelo → Right del usuario y viceversa."""
    if raw_label == "Left":
        return RIGHT
    if raw_label == "Right":
        return LEFT
    return UNKNOWN


def get_handedness(results: _Results, idx: int) -> str:
    """Extrae y normaliza la lateralidad de `results.handedness[idx]`."""
    if not results.handedness or idx >= len(results.handedness):
        return UNKNOWN
    cats = results.handedness[idx]
    if not cats:
        return UNKNOWN
    raw = cats[0].category_name
    return normalize_handedness(str(raw))


def is_left(results: _Results, idx: int) -> bool:
    return get_handedness(results, idx) == LEFT


def is_right(results: _Results, idx: int) -> bool:
    return get_handedness(results, idx) == RIGHT
