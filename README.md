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
pip install opencv-python mediapipe numpy
```

| Libreria | Versión probada | Que hace | Documentación |
|----------|-----------------|----------|---------------|
| **opencv-python** | 4.8+ | Captura video de la webcam, dibuja en frames, muestra la ventana (`cv2.VideoCapture`, `cv2.line`, `cv2.circle`, `cv2.putText`) | https://docs.opencv.org/4.x/ |
| **mediapipe** | 0.10+ | Framework de ML de Google para deteccion de manos (21 landmarks por mano) | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |
| **numpy** | 1.24+ | Algebra lineal para el perceptrón y manejo de coordenadas normalizadas | https://numpy.org/doc/ |

Compilación verificada:
```bash
venv/bin/python -m py_compile main.py utils/*.py tests/*.py && echo "PY_COMPILES OK"
venv/bin/python -c "import main; main.make_landmarker().close()"
```

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
python main.py
```

Tambien se puede arrancar directo en un modo:

```bash
python main.py grid   # o line, o hand
```

Controles:

| Tecla | Accion |
|-------|--------|
| `n` | Cambia al siguiente modo (hand -> grid -> line -> ...) |
| `q` | Salir |

## Modos

**grid** — Rejilla de coordenadas sobre el video con los ejes etiquetados.
Cada uno de los 21 landmarks de cada mano se dibuja con su coordenada
pixel `(x, y)` al costado. Un HUD muestra la cantidad de manos detectadas
y la posicion de la muneca.

**line** — Un perceptron (neurona con pesos `a, b, c`) se entrena en vivo
cada frame para separar puntos sobre el tramo P5->P9 (nudillo indice ->
nudillo medio) de puntos desplazados perpendicularmente. La frontera de
decision aprendida se dibuja como una linea azul que une P5 y P9. El HUD
muestra epocas de entrenamiento y pesos.

**hand** — Visual estilo Persona: fondo oscurecido, esqueleto blanco con
glow rojo, nudillos blancos, yemas de dedos en rojo con anillo blanco,
barra de titulo y acento diagonal. Pensado para verse bien antes de ser
util.

## Como funciona el codigo

### Pipeline general

```
Webcam -> OpenCV captura frame -> MediaPipe detecta manos -> El modo activo dibuja -> Se muestra el frame
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

**5. Modo activo**

`main.py` despacha el frame y los resultados al `draw()` del modo activo
(`tests/grid_test.py`, `tests/line_test.py` o `tests/hand_test.py`). Cada
modo es una funcion pura que recibe el frame y los resultados y devuelve
el frame dibujado. Con la tecla `n` se cicla entre modos sin reiniciar
la captura ni el modelo.

## Estructura de archivos

```
the-machine/
  main.py                # entrada: maneja el VideoCapture, el landmarker y el cambio de modo (tecla n)
  models/
    hand_landmarker.task # modelo pre-entrenado (7.5 MB)
    hand_recognition.py  # script monolitico original (referencia)
  tests/
    grid_test.py         # rejilla + coordenadas x,y de las manos
    line_test.py         # perceptron que une P5 y P9 con linea azul
    hand_test.py         # render estilo Persona de las manos
  utils/
    fps.py               # contador de FPS reutilizable (EMA)
  README.md              # este archivo
```
