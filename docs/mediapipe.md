# Uso de MediaPipe HandLandmarker

## Que es

MediaPipe es el framework de ML de Google para deteccion de landmarks de mano en video en tiempo real. Detecta 21 puntos por mano (landmarks) con sus coordenadas normalizadas (0..1).

## Inicializacion (main.py)

```python
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionTaskRunningMode.VIDEO,   # modo video (no imagen fija)
    num_hands=2,                                 # detectar hasta 2 manos
    min_hand_detection_confidence=0.5,           # umbral de deteccion
    min_tracking_confidence=0.5,                 # umbral de tracking entre frames
)
landmarker = HandLandmarker.create_from_options(options)
```

## Deteccion por frame

MediaPipe necesita RGB, no BGR (que es el formato nativo de OpenCV):

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
results = landmarker.detect_for_video(mp_img, timestamp_ms)
```

`results.hand_landmarks` es una lista de listas. Cada sublista tiene 21 landmarks de una mano. Cada landmark tiene `.x`, `.y` (normalizados 0..1) y `.z`.

## Conversion a pixeles

```python
h, w = frame.shape[:2]
pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
```

Las coordenadas `x,y` se multiplican por el tamano del frame para obtener coordenadas de pixel.

## Modelo

El archivo `models/hand_landmarker.task` es el modelo preentrenado de MediaPipe. Sin este archivo el landmarker no funciona.

## Landmarks de mano (21 puntos)

| Indice | Punto               |
|--------|---------------------|
| 0      | Muneca (wrist)      |
| 1-4    | Pulgar (thumb)      |
| 5-8    | Indice (index)      |
| 9-12   | Medio (middle)      |
| 13-16  | Anular (ring)       |
| 17-20  | Meñique (pinky)    |

En el proyecto se usan P5 (indice 4, yema del pulgar) y P9 (indice 8, yema del indice) para el perceptron.
