# Arquitectura del proyecto

## Capas

```
main.py              → fachada mínima (args → app.runner.run)
app/                 → orquestación: runner + vision + registry
presentation/        → UI: modes/ (hand/line/position/music) + ui/ (theme/layout/drawing/effects)
controllers/         → gesto→accion: hand.py (FingerRNN x5, HandController), gestures.py (MusicGestureController), volume.py (VolumeController, Perceptron)
core/                → dominio puro: fingers, perceptron, results, gestures, handedness, playlist (sin cv2/pygame)
config/              → configuración: palette, strings, settings (hojas)
common/              → transversal: fps
infrastructure/      → adapters: capture (cv2.VideoCapture), display (pygame), player (pygame.mixer)
models/              → modelos .task + finger_*.npz (RNN por dedo) + volume.npz (Perceptron)
music/               → sagas mp3 (EPIC: The Musical)
```

### Flujo de datos

```
Webcam → infrastructure/capture.Camera.read() [flip+resize 1060x720]
    ↓
app/vision.py → HandLandmarker (hand/line/music) | GestureRecognizer (position, 2 manos)
    ↓
presentation/modes/*.draw(frame, results) → (frame, hand_count)  [bbox viewfinder + landmarks 2px + crosshair, paleta B/W]
    ↓
presentation/ui/effects.apply_cctv_effect() [scanlines+viñeta+tinte verde]
    ↓
presentation/ui/layout → draw_sidebar (MODE/HANDS/FPS)  [B/W: WHITE/GRAY sobre BLACK]
    ↓
infrastructure/display.Window.show(canvas 1280x720) [BGR→RGB→pygame]
```

Flujo del modo music (por frame, bifurcacion sobre el anterior):

```
[mano derecha] presentation/modes/music.draw(right hand landmarks)
    ↓ core/fingers.features_from_landmarks → 5 binarias (±1, coseno del angulo de flexion MCP→PIP→TIP; umbrales por dedo: pulgar 0.50 / indice 0.45 / medio 0.45 / anular 0.35 / meñique 0.45) + hand-sign
    ↓ controllers/gestures.MusicGestureController.feed → window(8), agreement(4), settle(0.45s), rate limit(1.5s)
    ↓ controllers/hand.HandController.count → 5x FingerRNN → count → action(0=PAUSE,1=PREV SONG,2=NEXT SONG,3=PREV SAGA,4=NEXT SAGA,5=PLAY)
    ↓ .pending queda como accion
app/runner.consume_pending_action() → player.play()/pause()/prev|next song|saga

[mano izquierda] presentation/modes/music.draw(left hand landmarks)
    ↓ core/fingers.index_thumb_distance / pinky_thumb_distance (normalizadas por hand_scale)
    ↓ controllers/volume.VolumeController.feed → Perceptron (2 features) → nivel 0..1; commit al unir meñique↔pulgar, re-arm al separar
    ↓ .pending queda como volumen
app/runner.consume_pending_volume() → player.set_volume()
    ↓ pygame.mixer.music (devicename desde config/settings.AUDIO) → sink PulseAudio/bluetooth
```

### Dependencias entre capas

```
config/palette.py, config/strings.py, config/settings.py, common/fps.py     ← hojas
core/fingers.py → config (nada) | core/perceptron.py → (nada) | core/results.py → (nada) | core/handedness.py → (nada) | core/gestures.py → config/palette | core/playlist.py → (nada)
controllers/hand.py → config/strings + numpy | controllers/gestures.py → controllers/hand | controllers/volume.py → core/perceptron + config/settings
presentation/ui/theme.py → (nada) | presentation/ui/drawing.py → config/palette | presentation/ui/effects.py → config/palette
presentation/ui/layout.py → config/palette + config/strings + presentation/ui/theme  # solo draw_sidebar (B/W)
presentation/modes/*.py → controllers/* + core/* + config/* + common/fps + presentation/ui/*   # music → controllers/gestures + controllers/volume + core/fingers
app/registry.py → presentation/modes/* | app/vision.py → mediapipe | app/runner.py → app/* + infrastructure + presentation/ui/* + config/strings
infrastructure/capture.py → cv2 | infrastructure/display.py → pygame+cv2 | infrastructure/player.py → pygame + config/settings + core/playlist
main.py → app/registry + app/runner + config/strings
```

Reglas: `core` nunca importa `presentation`; `config/common` nunca importan capas superiores; `presentation` no importa `app/infrastructure`. El modo music comunica acciones y volumen con el runner a traves de `MusicGestureController.pending` y `VolumeController.pending` (`consume_pending_action`/`consume_pending_volume`), sin que `presentation` toque pygame. `core/handedness.py` aísla la corrección de flip (Left↔Right) para que `core/results.py` solo haga conversión geométrica/gestos. UI minimalista B/W sin header/footer (solo sidebar).

Para el informe tecnico completo (objetivos, tecnicas de ML, deteccion de dedos, fine-tuning de umbrales, limitaciones y trabajo futuro) ver [INFO.md](informe.md).

### Estilo visual

CCTV minimalista monocromo: scanlines cada 4px (`alpha 0.12`), viñeta gaussiana, tipografía `FONT_HERSHEY_SIMPLEX` `0.3-0.5` scale (position/music `0.60-0.70` bold con caja `BLACK` para legibilidad), bounding box `1-2px` + esquinas `12px`. Paleta predominante `BLACK`/`WHITE`/`GRAY`; `position` sin borde negro en esqueleto/landmarks (solo `WHITE 2px`) y sin color por gesto. Sidebar `SIDEBAR_BG`/`SIDEBAR_BORDER` en grises sin header/footer. `music` reutiliza `_put_text_box` (rect `BLACK` + texto `WHITE`).
