# Modos de presentacion (presentation/modes/)

Cada modo expone `draw(frame, results) -> (frame, hand_count)`.

## hand (`presentation/modes/hand.py`)

Skeleton fino `1px` + landmarks `2px` numerados `0.3` + bounding box viewfinder blanco + label handedness corregido por flip. Crosshair si `hand_count==0`.

## line (`presentation/modes/line.py`)

Perceptron en vivo por mano (P_A=4 thumb tip, P_B=8 index tip). `build_dataset` + `Perceptron.train_budget(budget=200)` cada frame. Linea `BLUE` `1px` entre P5→P9. `FINGERS_TOGETHER_THRESH=0.04` resetea perceptrón. Soporta 2 manos, `_perc[0]/[1]` persistentes.

## position (`presentation/modes/position.py`)

`GestureRecognizer` con `num_hands=2` (ambas manos). Paleta predominante `BLACK`/`WHITE` (sin `GESTURE_COLORS`): `draw_bbox`/`draw_skeleton`/`draw_landmarks` todo en `WHITE 2px` sin borde negro. Tipografía grande y legible con caja negra: título `0.70/2`, gesto `0.65/2` + `Left/Right` `0.60/2` + HUD `0.60/2` vía `_put_text_box()` (rect `BLACK` + texto `WHITE`). Muestra `gesture + confianza %` por mano (máx 2) y usa `core/handedness.get_handedness` (flip corregido) + `core/results.get_gesture`.

## Ciclo

`hand → line → position → hand` con `n`. Inicio: `python main.py [hand|line|position]`. Registry en `app/registry.py`.
