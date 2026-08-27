# the-machine

Deteccion de manos y gestos en tiempo real usando MediaPipe y OpenCV + estilo CCTV minimalista.

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

### 4. Ejecutar

```bash
python main.py              # inicia en hand
python main.py position     # inicia en gestos (2 manos)
```

| Tecla | Accion |
|-------|--------|
| `n` | Siguiente modo |
| `q` | Salir |

## Modos

**hand** — Esqueleto fino + bounding box viewfinder + label Left/Right. Crosshair si no hay mano.

**line** — Perceptron en vivo entre P5 (pulgar, idx 4) → P9 (índice, idx 8). Linea azul fina + 2 manos con perceptrones independientes.

**position** — GestureRecognizer (2 manos, paleta B/W). Gestos: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. Bounding box/skeleton/landmarks en `WHITE` sin borde negro + fuente grande legible con caja `BLACK` (`_put_text_box`) + `%` confianza por mano.

Estilo global: overlay CCTV monocromo (scanlines, viñeta), sidebar B/W con modo/hands/FPS.

## Arquitectura en Capas

```
main.py              → fachada (parsea args)
app/                 → orquestación (runner, vision, registry)
presentation/        → UI (modes/ + ui/theme|layout|drawing|effects)
core/                → dominio puro (perceptron, results, handedness, gestures)
config/              → configuración (palette, strings)
common/              → transversal (fps)
infrastructure/      → adapters (capture, display)
models/              → .task preentrenados
docs/                → documentación por capa
```

Regla: `presentation → core → config/common`, `app → presentation/core/infrastructure`, `main → app/config`. `core/handedness.py` aísla corrección `Left↔Right` (flip espejo) separada de `core/results.py`. Ver `docs/architecture.md`.

## Estructura de archivos

```
the-machine/
  main.py
  app/
    runner.py        # loop, try/finally, CCTV effects
    vision.py        # make_landmarker / make_recognizer
    registry.py      # TESTS dict
  presentation/
    modes/hand.py, line.py, position.py
    ui/theme.py, layout.py, drawing.py, effects.py
   core/
     perceptron.py, results.py, handedness.py, gestures.py
  config/
    palette.py, strings.py
  common/fps.py
  infrastructure/
    capture.py, display.py
  models/
    hand_landmarker.task, gesture_recognizer.task
  docs/
    architecture.md, modes.md, layers.md, mediapipe.md, perceptron.md, opencv.md, pygame.md
```
