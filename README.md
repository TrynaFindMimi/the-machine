# the-machine

Deteccion de manos y gestos en tiempo real usando MediaPipe y OpenCV, con una interfaz de estilo "Grecia antigua / EPIC: The Musical" (paleta basalto/bronce/dorado/marmol y motivos griegos). Incluye un reproductor de musica (EPIC: The Musical) controlado por gestos de la mano derecha: un Perceptron para el pulgar y 4 RNNs (una por dedo restante), y volumen por la mano izquierda (Perceptron).

> La musica incluida es obra de Jorge Hernan (EPIC: The Musical) y se distribuye con fines academicos, sin animo de lucro. Ver [CREDITS.md](CREDITS.md).

## Requisitos

- Python 3.10+
- Webcam

## Instalacion

> Lanzamiento rapido por OS: usa `run-windows.ps1` (PowerShell) o `run-linux.sh` (bash). Crean el entorno `venv313`, instalan dependencias y arrancan `main.py`.

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
| **opencv-python** | Captura video, preprocesado y dibujo de la UI |
| **mediapipe** | 21 landmarks + 8 gestos |
| **numpy** | Perceptron y coordenadas |
| **pygame** | Ventana 1280x720 |

### 3. Descargar modelos

```bash
cd models/
wget "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
wget "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
```

Los pesos RNN por dedo (`models/finger_*.npz`) ya estan versionados; si faltan, el modo music los reentrena al iniciarse.

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
| Mano cerrada (0 dedos) | Saga anterior |
| 1 dedo | Cancion anterior |
| 2 dedos | Pausar |
| 3 dedos | Reproducir |
| 4 dedos | Siguiente cancion |
| Mano abierta (5 dedos) | Siguiente saga |

| Mano izquierda | Accion |
|-------|--------|
| Pinza indice↔pulgar (distancia) | Ajustar volumen (perceptron) |
| Anadir meñique a la pinza (junta meñique↔pulgar) | Guardar/aplicar volumen |

El volumen de la mano izquierda usa un `Perceptron` (2 features: distancia indice→pulgar normalizada + bias) entrenado como clasificador cerca/lejos; su salida continua se mapea a 0..1 (dedos juntos = 0, separados = 100). El nivel se aplica al reproductor solo cuando el meñique se une al pulgar (pinza meñique↔pulgar), y se rearma al separarlo. Se aplica un cooldown a los gestos de la mano derecha para que un comando no se repita mientras se mantiene el gesto.

### Como se elige el dispositivo de audio

El dispositivo de salida se configura en `config/settings.py` (`AudioSettings`): driver `wasapi` en Windows / `pulseaudio` en Linux (override con `THE_MACHINE_AUDIO_DRIVER`) y `device` opcional para elegir salida concreta (override con `THE_MACHINE_AUDIO_DEVICE`). `infrastructure/player.py` setea `SDL_AUDIODRIVER` desde `AUDIO.driver` y re-inicializa `pygame.mixer` en cada `play()`, usando `AUDIO.device` como `devicename` (si es `None`, SDL usa el dispositivo por defecto del sistema). Cadena de fallback si el driver configurado falla: 2) driver por defecto de SDL (sin `SDL_AUDIODRIVER`); 3) `pygame.mixer.init()` final sin driver ni device. Si todo falla (p. ej. entorno sin audio), el modo `music` sigue funcionando como UI sin sonido y avisa por consola (`audio_ok=False`).

```bash
# override del driver/dispositivo por entorno (todas las variables opcionales)
THE_MACHINE_AUDIO_DRIVER=alsa THE_MACHINE_AUDIO_DEVICE="USB Headset" python main.py
THE_MACHINE_CAMERA_INDEX=1 python main.py   # si tu webcam no es el indice 0
```

```bash
# listar los dispositivos de audio que ve SDL (Linux/pulseaudio; en Windows usa wasapi)
venv/bin/python -c "import os; os.environ['SDL_AUDIODRIVER']='pulseaudio'; import pygame; pygame.mixer.init(); from pygame import _sdl2; print(_sdl2.audio.get_audio_device_names(True))"
```

### Como se decide el gesto (capas)

```
core/fingers.features_from_landmarks → 5 binarias (±1): indice/medio/anular/meñique por coseno del angulo de flexion MCP→PIP→TIP (umbrales 0.45/0.45/0.35/0.45) + pulgar por Perceptron 3-features (dist a puntos de la palma {5,7,9,13,17} + dentro del cuadrilatero 5-17-1-9 + bias) + hand-sign
controllers/thumb.ThumbPerceptron → pesos models/thumb.npz; decisión por frame del pulgar
controllers/hand.HandController → 4x FingerRNN (secuencia de 8 features por dedo) + mayoría del pulgar sobre la ventana → count()
controllers/gestures.MusicGestureController → ventana, agreement (4), settle (0.45s), rate limit (1.5s), mapeo count→accion
app/runner → consume_pending_action → player.play()/pause()/prev/next song|saga
```

La extension de cada dedo (excepto pulgar) se mide como el coseno del angulo de flexion en la articulacion PIP (segmentos `MCP→PIP` y `PIP→TIP`), sin importar la orientacion de la mano; el anular recibe umbral mas permisivo (0.35). El **pulgar** ya no usa RNN: un Perceptron binario decide con 3 features respecto a la palma — distancia minima de la punta (P4) a los puntos del cuadrilatero de la palma `{5,7,9,13,17}` normalizada por `hand_scale`, un flag "punta dentro del cuadrilatero (5,17,1,9)" y el bias — entrenado con polos sinteticos "doblado sobre la palma / pegado al indice" vs "extendido lejos", con suavizado por mayoria en la ventana de 8 frames. Detalle completo en [INFO.md](docs/informe.md).

Volumen (mano izquierda):

```
core/fingers.index_thumb_distance → distancia indice↔pulgar normalizada por hand_scale
controllers/volume.VolumeController → Perceptron (2 features) mapea distancia→0..1 en vivo cada frame
app/runner → consume_pending_volume → player.set_volume()
```

## Modos

**hand** — Esqueleto fino + bounding box viewfinder + label Left/Right. Crosshair si no hay mano.

**line** — Perceptron en vivo entre P5 (pulgar, idx 4) → P9 (índice, idx 8). Linea azul fina + 2 manos con perceptrones independientes.

**position** — GestureRecognizer (2 manos, paleta B/W). Gestos: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. Bounding box/skeleton/landmarks en `WHITE` sin borde negro + fuente grande legible con caja `BLACK` (`_put_text_box`) + `%` confianza por mano.

**music** — Reproductor de EPIC por sagas (folders `NN Title.mp3`). UI con caja negra (`_put_text_box`): SAGA, SONG, TRACK, STATE, FINGERS (4 RNN + Perceptron de pulgar) y ACTION. Tags negros estilo `position`. Mano derecha controla reproduccion/navegacion (con settle + cooldown anti-repeticion); mano izquierda controla volumen solo con indice+pulgar: juntos→bajo, separados→alto, en vivo.

Estilo global: temática griega. Paleta `config/palette.py` con basalto (fondo), bronce (bordes/relieves), dorado (acentos), marfil/piedra (texto) y oliva (live). `presentation/ui/layout.draw_sidebar` dibuja un panel BASALT con `draw_fret_band` (banda de friso) y lista de modos/HANDS/FPS/CONTROLS, siempre pegado al borde derecho del canvas (`canvas[:, width-sidebar_width:]`), fuera del área de la cámara. El helper `apply_cctv_effect` existe pero queda como no-op (efecto CCTV desactivado); `presentation/ui/greek.py` aporta los motivos (panel, friso, frontón, laurel, texto centrado). Los modos dibujan esqueleto/bbox en `WHITE` sobre `BLACK` (crosshair si no hay mano).

## Arquitectura en Capas

```
main.py              → fachada (parsea args)
app/                 → orquestación (runner, vision, registry)
presentation/        → UI (modes/ + ui/theme|layout|drawing|effects|greek)
controllers/         → gesto→accion (hand: 4x HandController RNN, thumb: ThumbPerceptron, gestures: MusicGestureController)
core/                → dominio puro (finger_features, perceptron, results, handedness, gestures, playlist)
config/              → configuración (palette, strings, settings)
common/              → transversal (fps)
infrastructure/      → adapters (capture, display, player)
models/              → .task preentrenados + finger_1..4.npz (RNN) + thumb.npz / volume.npz (Perceptron)
music/               → sagas (mp3)
docs/                → documentación por capa
```

Regla: `presentation → controllers/core → config/common`, `app → presentation/controllers/infrastructure`, `main → app/config`. `presentation` no importa `infrastructure`; la accion del gesto se comunica via `MusicGestureController.pending` (u `consume_pending_action()`). Ver `docs/architecture.md` e [INFO.md](docs/informe.md) para el informe detallado de arquitectura y proyecto.

## Estructura de archivos

```
the-machine/
  main.py
  app/
    runner.py        # loop, try/finally, sidebar griega + contexto music + dispatch de acciones
    vision.py        # make_landmarker / make_recognizer
    registry.py      # TESTS dict
  presentation/
    modes/hand.py, line.py, position.py, music.py
    ui/theme.py, layout.py, drawing.py, effects.py, greek.py
  controllers/
    hand.py          # FingerRNN x4 + HandController (count por mayoria de pulgar + RNNs)
    thumb.py         # ThumbPerceptron (3 features de palma, models/thumb.npz)
    gestures.py      # MusicGestureController (ventana, agreement, cooldown, pending)
    volume.py        # VolumeController (Perceptron 2-f, commit por meñique↔pulgar, re-arm)
  core/
    fingers.py       # features_from_landmarks + index_thumb_distance + pinky_thumb_distance
    perceptron.py, results.py, handedness.py, gestures.py
    playlist.py      # scan_sagas(base)
  config/
    palette.py, strings.py, settings.py
  common/fps.py
  infrastructure/
    capture.py, display.py, player.py   # MusicPlayer over pygame.mixer
  models/
    hand_landmarker.task, gesture_recognizer.task, finger_1..4.npz, thumb.npz, volume.npz
  music/
    01 The Troy Saga/ ... 09 The Ithaca Saga/
  docs/
    informe.md, architecture.md, modes.md, layers.md, mediapipe.md, perceptron.md, opencv.md, pygame.md, utils.md
  CREDITS.md
```
