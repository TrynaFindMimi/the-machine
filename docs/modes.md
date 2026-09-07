# Modos de presentacion (presentation/modes/)

Cada modo expone `draw(frame, results) -> (frame, hand_count)`.

## hand (`presentation/modes/hand.py`)

Skeleton fino `1px` + landmarks `2px` numerados `0.3` + bounding box viewfinder blanco + label handedness corregido por flip. Crosshair si `hand_count==0`.

## line (`presentation/modes/line.py`)

Perceptron en vivo por mano (P_A=4 thumb tip, P_B=8 index tip). `build_dataset` + `Perceptron.train_budget(budget=200)` cada frame. Linea `BLUE` `1px` entre P5→P9. `FINGERS_TOGETHER_THRESH=0.04` resetea perceptrón. Soporta 2 manos, `_perc[0]/[1]` persistentes.

## position (`presentation/modes/position.py`)

`GestureRecognizer` con `num_hands=2` (ambas manos). Paleta predominante `BLACK`/`WHITE` (sin `GESTURE_COLORS`): `draw_bbox`/`draw_skeleton`/`draw_landmarks` todo en `WHITE 2px` sin borde negro. Tipografía grande y legible con caja negra: título `0.70/2`, gesto `0.65/2` + `Left/Right` `0.60/2` + HUD `0.60/2` vía `_put_text_box()` (rect `BLACK` + texto `WHITE`). Muestra `gesture + confianza %` por mano (máx 2) y usa `core/handedness.get_handedness` (flip corregido) + `core/results.get_gesture`.

## music (`presentation/modes/music.py`)

Reproductor de EPIC: The Musical por sagas. Solo procesa la mano derecha (`core/handedness.get_handedness` + flip corregido). UI de tags negros estilo position vía `_put_text_box`: título `0.70/2`, SAGA `0.55/2`, SONG `0.45/1`, TRACK `0.45/1`, STATE `0.55/1`, FINGERS `0.5/1`, ACTION `0.5/1`, etiqueta `Right` junto al bbox.

Pipeline por frame:
- `core/fingers.features_from_landmarks` → 5 features binarias (±1): pulgar por distancia euclideana tip↔pip/scale (umbral `0.25`), resto por delta `y` (`0.15`) + 6ª feature hand-sign (no usada para contar).
- `controllers/hand.HandController.count(window 8x6)` → 5 `FingerLSTM` (uno por dedo) → suma dedos detectados.
- `controllers/gestures.MusicGestureController.feed(feat)` → ventana `len=8`, agreement `>=4` frames, cooldown `12`, mapeo `0→PAUSE`, `>=4→PLAY` vía `HandController.action`.
- `consume_pending_action()` → el runner despacha `player.play()/pause()`.

Gestos: puño (0 dedos) = PAUSE, mano abierta (>=4) = PLAY. `FingerLSTM` (hidden=8, 60 epochs, sin dropout) entrenado con ruido gaussiano en `models/finger_*.npz` (`HandController.train_all`); si no existen, se reentrenan al iniciar.

## Ciclo

`hand → line → position → music → hand` con `n`. Inicio: `python main.py [hand|line|position|music]`. Registry en `app/registry.py`.
