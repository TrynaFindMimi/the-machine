# the-machine

Deteccion de manos y gestos en tiempo real usando MediaPipe y OpenCV + estilo CCTV minimalista. Incluye un reproductor de musica (EPIC: The Musical) controlado por gestos de la mano derecha mediante 5 LSTMs (uno por dedo).

> La musica incluida es obra de Jorge Hernan (EPIC: The Musical) y se distribuye con fines academicos, sin animo de lucro. Ver [CREDITS.md](CREDITS.md).

## Requisitos

- Python 3.10+
- Webcam

## Instalacion

### 1. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 2. Instalar dependencias

```bash
pip install opencv-python mediapipe numpy pygame
```

| Libreria | Que hace |
|----------|----------|
| **opencv-python** | Captura video, dibujo CCTV |
| **mediapipe** | 21 landmarks + 8 gestos |
| **numpy** | Perceptron y coordenadas |
| **pygame** | Ventana 1280x720 |

### 3. Descargar modelos

```bash
cd models/
wget "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
wget "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
```

Los pesos LSTM por dedo (`models/finger_*.npz`) ya estan versionados; si faltan, el modo music los reentrena al iniciarse.

### 4. Ejecutar

```bash
python main.py              # inicia en hand
python main.py position     # inicia en gestos (2 manos)
python main.py music        # inicia en music player
```

| Tecla | Accion |
|-------|--------|
| `n` | Siguiente modo |
| `q` | Salir |

## Modo music

`app/runner.py` instancia `infrastructure/player.MusicPlayer` (pygame.mixer) y cada frame inyecta el contexto (saga/song/track/state) al modo `presentation/modes/music.py`, que dibuja la UI y expone la accion pendiente del controlador de gestos.

| Gestos (mano derecha) | Accion |
|-------|--------|
| Mano cerrada (0 dedos) | Pausar |
| 1 dedo | Cancion anterior |
| 2 dedos | Siguiente cancion |
| 3 dedos | Saga anterior |
| 4 dedos | Siguiente saga |
| Mano abierta (5 dedos) | Reproducir |

| Mano izquierda | Accion |
|-------|--------|
| Pinza anular↔pulgar (distancia) | Ajustar volumen (perceptron) |
| Anadir meñique a la pinza | Guardar/aplicar volumen |

El volumen de la mano izquierda usa un `Perceptron` (2 features: distancia anular→pulgar normalizada + bias) entrenado como clasificador cerca/lejos; su salida continua se mapea a 0..1. El nivel se aplica al reproductor solo cuando el meñique se une a la pinza, y se rearma al separarlo.

### Como se elige el dispositivo de audio

El dispositivo de salida se configura en `config/settings.py` (`AudioSettings`): driver `pulseaudio` y devicename con el nombre del sink que expone SDL (por ejemplo `AirPods Max - Find My`). El mixer se re-inicializa con ese dispositivo en cada `play()`.

```bash
# listar los dispositivos de audio que ve SDL
venv/bin/python -c "import os; os.environ['SDL_AUDIODRIVER']='pulseaudio'; import pygame; pygame.mixer.init(); from pygame import _sdl2; print(_sdl2.audio.get_audio_device_names(True))"
```

### Como se decide el gesto (capas)

```
core/fingers.features_from_landmarks → 5 binarias (±1, pulgar euclidea 0.25 / resto 0.15) + hand-sign
controllers/hand.HandController → 5x FingerLSTM (secuencia de 8 features por dedo) → count()
controllers/gestures.MusicGestureController → ventana, agreement (4), cooldown (12), mapeo count→accion
app/runner → consume_pending_action → player.play()/pause()/prev/next song|saga
```

Volumen (mano izquierda):

```
core/fingers.ring_thumb_distance / pinky_ring_distance → distancias normalizadas
controllers/volume.VolumeController → Perceptron (2 features) mapea distancia→0..1, commit al cerrar pinza+meñique, re-arm al separar
app/runner → consume_pending_volume → player.set_volume()
```

## Modos

**hand** — Esqueleto fino + bounding box viewfinder + label Left/Right. Crosshair si no hay mano.

**line** — Perceptron en vivo entre P5 (pulgar, idx 4) → P9 (índice, idx 8). Linea azul fina + 2 manos con perceptrones independientes.

**position** — GestureRecognizer (2 manos, paleta B/W). Gestos: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. Bounding box/skeleton/landmarks en `WHITE` sin borde negro + fuente grande legible con caja `BLACK` (`_put_text_box`) + `%` confianza por mano.

**music** — Reproductor de EPIC por sagas (folders `NN Title.mp3`). UI con caja negra (`_put_text_box`): SAGA, SONG, TRACK, STATE, FINGERS (LSTM) y ACTION. Tags negros estilo `position`. Mano derecha controla reproduccion/navegacion; mano izquierda controla volumen (distancia anular↔pulgar, commit por meñique).

Estilo global: overlay CCTV monocromo (scanlines, viñeta), sidebar B/W con modo/hands/FPS.

## Arquitectura en Capas

```
main.py              → fachada (parsea args)
app/                 → orquestación (runner, vision, registry)
presentation/        → UI (modes/ + ui/theme|layout|drawing|effects)
controllers/         → gesto→accion (hand: HandController LSTM, gestures: MusicGestureController)
core/                → dominio puro (finger_features, perceptron, results, handedness, gestures, playlist)
config/              → configuración (palette, strings, settings)
common/              → transversal (fps)
infrastructure/      → adapters (capture, display, player)
models/              → .task preentrenados + finger_*.npz (LSTM)
music/               → sagas (mp3)
docs/                → documentación por capa
```

Regla: `presentation → controllers/core → config/common`, `app → presentation/controllers/infrastructure`, `main → app/config`. `presentation` no importa `infrastructure`; la accion del gesto se comunica via `MusicGestureController.pending` (u `consume_pending_action()`). Ver `docs/architecture.md`.

## Estructura de archivos

```
the-machine/
  main.py
  app/
    runner.py        # loop, try/finally, CCTV effects, music context + action dispatch
    vision.py        # make_landmarker / make_recognizer
    registry.py      # TESTS dict
  presentation/
    modes/hand.py, line.py, position.py, music.py
    ui/theme.py, layout.py, drawing.py, effects.py
  controllers/
    hand.py          # FingerLSTM x5 + HandController (count/action/train/save/load)
    gestures.py      # MusicGestureController (ventana, agreement, cooldown, pending)
    volume.py        # VolumeController (Perceptron 2-f, commit por meñique, re-arm)
  core/
    fingers.py       # features_from_landmarks + ring_thumb_distance + pinky_ring_distance
    perceptron.py, results.py, handedness.py, gestures.py
    playlist.py      # scan_sagas(base)
  config/
    palette.py, strings.py, settings.py
  common/fps.py
  infrastructure/
    capture.py, display.py, player.py   # MusicPlayer over pygame.mixer
  models/
    hand_landmarker.task, gesture_recognizer.task, finger_0..4.npz, volume.npz
  music/
    01 The Troy Saga/ ... 09 The Ithaca Saga/
  docs/
    architecture.md, modes.md, layers.md, mediapipe.md, perceptron.md, opencv.md, pygame.md
  CREDITS.md
```
