# Hand Recognition

Deteccion y conteo de manos en tiempo real usando MediaPipe y OpenCV.

## Requisitos

- Python 3.10+
- Webcam

## Instalacion

### 1. Crear entorno virtual

```bash
python -m venv venv
```

Activar el entorno:

```bash
# Linux / Mac
source venv/bin/activate

# Windows (CMD)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```bash
pip install opencv-python mediapipe
```

| Libreria | Que hace |
|----------|----------|
| **opencv-python** | Captura video de la webcam, dibuja en frames, muestra la ventana |
| **mediapipe** | Framework de ML de Google para deteccion de manos (y otros landmark) |

### 3. Descargar el modelo (.task)

MediaPipe Tasks API necesita un archivo `.task` con el modelo pre-entrenado.
Descargalo una sola vez:

```bash
# Linux / Mac
wget "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

> **Que es el archivo `.task`?**
> Es un modelo de TensorFlow Lite empaquetado para la Tasks API de MediaPipe.
> Contiene los pesos y la arquitectura de una red neuronal entrenada para
> detectar manos y localizar 21 puntos (landmarks) por mano: muñeca, nudillos,
> falanges y yemas de los dedos. El modelo `float16` usa menor precision
> (16 bits en vez de 32) para ser mas rapido y liviano sin perder precision
> notable. Se descarga de:
> https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

### 4. Ejecutar

```bash
python finalcount.py
```

Presiona `q` para salir.

## Como funciona el codigo

### Pipeline general

```
Webcam -> OpenCV captura frame -> MediaPipe detecta manos -> Se dibujan los landmarks -> Se muestra el frame
```

### Explicacion por partes

**1. Configuracion del HandLandmarker**

Se crea un `HandLandmarkerOptions` que le dice a MediaPipe:
- Que modelo usar (`hand_landmarker.task`)
- Modo de ejecucion: `VIDEO` (un frame a la vez, con timestamp)
- Cuantas manos detectar al mismo tiempo (hasta 6)
- Confianza minima para detectar y rastrear manos (0.5)

Luego se crea el `HandLandmarker` con esas opciones.

**2. Captura de video**

OpenCV abre la webcam (`VideoCapture(0)`) y entra en un loop leyendo frames.

**3. Preprocesamiento de cada frame**

- Se voltea el frame horizontalmente (efecto espejo, mas natural)
- Se convierte de BGR (formato de OpenCV) a RGB (formato que espera MediaPipe)
- Se envuelve en un `mp.Image` con formato SRGB

**4. Deteccion de manos**

`detect_for_video(mp_img, timestamp)` corre el modelo de IA sobre la imagen.
Retorna una lista de manos detectadas, cada una con 21 landmarks (x, y, z normalizados de 0 a 1).

**5. Dibujado**

Para cada mano detectada se dibujan:
- Puntos blancos en cada landmark (nudillos, yemas, muñeca)
- Lineas blancas conectando los landmarks segun la anatomia de la mano
- Texto con el numero de manos detectadas

## Estructura de archivos

```
hand-recognition/
  hand_landmarker.task   # modelo pre-entrenado (7.5 MB)
  finalcount.py          # script principal
  venv/                  # entorno virtual
  README.md              # este archivo
```
