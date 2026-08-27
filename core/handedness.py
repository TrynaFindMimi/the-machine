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

LEFT = "Left"
RIGHT = "Right"
UNKNOWN = "?"


def normalize_handedness(raw_label: str) -> str:
    """Corrige el flip horizontal: Left del modelo → Right del usuario y viceversa."""
    if raw_label == "Left":
        return RIGHT
    if raw_label == "Right":
        return LEFT
    return UNKNOWN


def get_handedness(results, idx: int) -> str:
    """Extrae y normaliza la lateralidad de `results.handedness[idx]`."""
    if not results.handedness or idx >= len(results.handedness):
        return UNKNOWN
    cats = results.handedness[idx]
    if not cats:
        return UNKNOWN
    raw = cats[0].category_name
    return normalize_handedness(raw)


def is_left(results, idx: int) -> bool:
    return get_handedness(results, idx) == LEFT


def is_right(results, idx: int) -> bool:
    return get_handedness(results, idx) == RIGHT
