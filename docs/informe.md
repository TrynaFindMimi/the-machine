# Informe tecnico — the-machine

Deteccion de manos y gestos en tiempo real con MediaPipe y OpenCV, con estilo visual CCTV monocromo minimalista. Incluye un reproductor de musica (EPIC: The Musical) controlado por la mano derecha (5 RNNs, una por dedo) y volumen controlado por la mano izquierda (Perceptron).

---

## 1. Resumen ejecutivo

`the-machine` es un proyecto de inteligencia artificial que combina vision por computadora con modelos de aprendizaje automatico pequenos y entrenados "en casa" (sin frameworks de deep learning) sobre una base de landmarks de MediaPipe:

1. **MediaPipe HandLandmarker** detecta hasta 2 manos y devuelve **21 landmarks** por mano en tiempo real.
2. Una capa de **features geometricas** (`core/fingers.py`) convierte los landmarks en 5 senales binarias (±1) que indican si cada dedo esta extendido.
3. Cinco **RNN puras en NumPy** (`controllers/hand.py`), una por dedo, suavizan la secuencia temporal y producen el conteo `0..5`.
4. El conteo se mapea a una accion del reproductor (pause, prev/next song, prev/next saga, play) con anti-repeticion.
5. Una segunda rama usa un **Perceptron** para el control de volumen con la mano izquierda.

Todo el pipeline se ejecuta por frame (~30 FPS) y la UI se pinta con OpenCV sobre una ventana de pygame con overlay CCTV.

---

## 2. Contexto y objetivos

- **Contexto**: asignatura de Inteligencia Artificial (universidad).
- **Objetivo general**: construir un sistema de control por gestos de manos en tiempo real que, ademas, sirva de via de estudio practico de los modelos de IA clásicos (perceptron, RNN) implementados desde cero.
- **Objetivos especificos**:
  1. Detectar manos y landmarks con MediaPipe (`hand`, `line`, `position`).
  2. Contar dedos levantados de forma robusta para navegar un reproductor de musica (`music`).
  3. Controlar volumen con gestos de pinza de la mano izquierda.
  4. Mantener una arquitectura en capas limpia, documentada y con responsabilidades separadas.

La musica incluida (EPIC: The Musical) tiene fines academicos; ver `CREDITS.md`.

---

## 3. Tecnologias

| Libreria | Rol |
|----------|-----|
| **opencv-python** | Captura de video, preprocesado, dibujo de la UI y efectos CCTV |
| **mediapipe** | `HandLandmarker` (21 landmarks) y `GestureRecognizer` (8 gestos preentrenados) |
| **numpy** | Implementacion de RNN y Perceptron, vectores de features |
| **pygame** | Ventana de salida y mixer de audio (`pygame.mixer`) |

Otros: `dataclasses` para configuracion tipada (`config/settings.py`), `contextlib.ExitStack` para limpieza de recursos (`app/runner.py`).

---

## 4. Arquitectura en capas

```
main.py              → fachada minima (parsea args → app.runner.run)
app/                 → orquestacion: registry, vision, runner
presentation/        → UI: modes/ (hand|line|position|music) + ui/ (theme|layout|drawing|effects)
controllers/         → gesto→accion: hand.py (FingerRNN x5), gestures.py (MusicGestureController), volume.py (VolumeController)
core/                → dominio puro: fingers, perceptron, results, gestures, handedness, playlist
config/              → configuracion: palette, strings, settings
common/              → transversal: fps
infrastructure/      → adapters: capture, display, player
models/              → .task de MediaPipe + finger_*.npz (RNN) + volume.npz (Perceptron)
music/               → sagas mp3 (EPIC: The Musical)
docs/                → documentacion por capa e informe tecnico
```

### Reglas de dependencia

- `core` nunca importa `presentation` ni `infrastructure` (dominio puro, sin cv2/pygame).
- `config` y `common` son hojas: nada importa hacia abajo de ellas.
- `presentation` no importa `app` ni `infrastructure`; el modo music no toca pygame directamente.
- La comunicacion gesto→accion es *diferida*: el controlador deja valores en `.pending` y el runner los consume (`consume_pending_action` / `consume_pending_volume`).
- `core/handedness.py` aísla la correccion del flip de camara (Left↔Right), de modo que el clasificador de gestos de MediaPipe no se ve afectado por el espejo.

### Por que esta separacion

- Las features de dedo, el perceptron y la playlist son logicamente puros y testeables sin camara (se pueden validar con landmarks sinteticos, como se hizo para calibrar los umbrales; ver seccion 8).
- Los adaptadores (camara, ventana, sonido) son intercambiables.
- Anadir un modo nuevo = crear `presentation/modes/<x>.py` con `draw()`, registrarlo en `app/registry.py` y darle titulo en `config/strings.py`.

---

## 5. Flujo de datos por frame

```
Webcam → infrastructure/capture.Camera.read()   [flip + resize 1060x720]
    ↓ BGR→RGB + resize de inferencia (60%)
    ↓ app/vision.py → HandLandmarker (hand|line|music) | GestureRecognizer (position)
    ↓ presentation/modes/<modo>.draw(frame, results) → (frame, hand_count)
    ↓ presentation/ui/effects.apply_cctv_effect()      [scanlines + viñeta + tinte]
    ↓ presentation/ui/layout.draw_sidebar()            [MODE / HANDS / FPS]
    ↓ infrastructure/display.Window.show(canvas 1280x720)
```

### Rama music (por frame)

Mano derecha (acciones):

```
features (5 binarias + hand-sign)
  → MusicGestureController.feed   [ventana 8, agreement 4, settle 0.45s]
  → HandController.count          [5x FingerRNN → suma]
  → HandController.action(count)  [0=PAUSE 1=PREV SONG 2=NEXT SONG 3=PREV SAGA 4=NEXT SAGA 5=PLAY]
  → .pending → app/runner → player.play()/pause()/prev|next song|saga
```

Mano izquierda (volumen):

```
index_thumb_distance / pinky_thumb_distance (normalizadas por hand_scale)
  → VolumeController.feed         [Perceptron 2 features → 0..1]
  → commit al unir meñique↔pulgar, re-arm al separar (cooldown 3s)
  → .pending → app/runner → player.set_volume()
```

---

## 6. Componentes por capa

| Capa | Modulo | Responsabilidad |
|------|--------|-----------------|
| **Fachada** | `main.py` | Parsear `argv`, validar modo, delegar en `app.runner.run`. |
| **Aplicacion** | `app/runner.py` | Bucle principal, limpieza con `try/finally` (ExitStack), efecto CCTV, sidebar, dispatch de acciones y volumen del modo music. |
| **Aplicacion** | `app/vision.py` | Crear `HandLandmarker` / `GestureRecognizer` con la configuracion de `config/settings.py`. |
| **Aplicacion** | `app/registry.py` | Mapa `TESTS` (modo → función `draw`) y orden `TEST_NAMES`. |
| **Presentacion** | `presentation/modes/*.py` | Un `draw(frame, results)` por modo. `music.py` no conoce `infrastructure`. |
| **Presentacion** | `presentation/ui/*.py` | Tema, layout (sidebar), dibujo (bbox, skeleton, texto con caja) y efectos (CCTV). |
| **Controladores** | `controllers/hand.py` | `FingerRNN` (RNN simple/Elman en NumPy + BPTT), `HandController` (count/action/train/save/load). |
| **Controladores** | `controllers/gestures.py` | `MusicGestureController`: ventana temporal, agreement, settle, rate limit, `pending`. |
| **Controladores** | `controllers/volume.py` | `VolumeController`: perceptron volumen, commit/re-arm, EMA. |
| **Dominio** | `core/fingers.py` | Features de dedos, `hand_scale`, distancias indice/pinky↔pulgar. |
| **Dominio** | `core/perceptron.py` | Perceptron binario (modo line). |
| **Dominio** | `core/results.py` | `count_hands`, `to_pixel_points`, `get_gesture`. |
| **Dominio** | `core/handedness.py` | Correccion Left↔Right por el espejo de camara. |
| **Dominio** | `core/playlist.py` | `scan_sagas` → lista de pistas por carpeta `NN Title.mp3`. |
| **Config/Transversal** | `config/`, `common/` | Constantes, textos, dataclasses de settings, contador FPS. |
| **Infraestructura** | `infrastructure/capture.py` | `Camera` sobre cv2.VideoCapture (flip + resize). |
| **Infraestructura** | `infrastructure/display.py` | `Window` sobre pygame (BGR→RGB). |
| **Infraestructura** | `infrastructure/player.py` | `MusicPlayer` sobre pygame.mixer con seleccion de dispositivo (PulseAudio/bluetooth). |

---

## 7. Modelos y tecnicas de IA

### 7.1 MediaPipe de base (modelo preentrenado)

- `models/hand_landmarker.task`: 21 landmarks por mano (0=muñeca, 1-4 pulgar, 5-8 indice, 9-12 medio, 13-16 anular, 17-20 meñique). Usado en `hand`, `line` y `music` en modo VIDEO.
- `models/gesture_recognizer.task`: clasifica 8 gestos (None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou). Usado solo en `position`.

MediaPipe aporta la percepcion; todo lo demas (conteo, acciones, volumen) es propio.

### 7.2 Extraccion de features de dedos (`core/fingers.py`)

Cada frame y mano produce **6 features**:

- **5 binarias por dedo** (±1): `1` si el dedo esta extendido, `-1` si esta flexionado.
- **1 "hand-sign"** (`_horizontal_hand_sign`): orientacion aproximada muñeca→MCP medio, normalizada; no participa en el conteo.

### 7.3 FingerRNN (una por dedo)

- RNN simple de tipo **Elman** implementada **manualmente en NumPy** (oculta diminuta, hidden=8) con recurrencia `h = tanh(Wx·x + Wh·h + b)` y salida softmax de 2 clases (plegado=0 / extendido=1).
- Entrada: la feature binaria del dedo a lo largo de una **ventana de 8 frames**.
- Entrenamiento supervisado sintetico (`HandController.train_all`): 45 epochs con **mezclas sinteticas por grupo** (clase plegado concentrada en `(0.00, 0.20, 0.42)`, clase extendido en `(0.65, 0.82, 1.00)` de frames extendidos dentro de la ventana) mas windows constantes ±1 con noise gaussiano 0.05. La semeilla de inicializacion es fija por dedo (`seed=idx` en `HandController`), por lo que el resultado es **reproducible**. La RNN actua como **filtro temporal**: ignora parpadeos de 1-2 frames y estabiliza el conteo (flip point por dedo entre 6/8 y 8/8).
- Pesos versionados en `models/finger_0..4.npz` (claves `Wx, Wh, b, Wy, by`); si faltan, se reentrenan al iniciar el modo music.
- `MusicGestureController` exige **agreement de 4 frames** con el mismo conteo y un **settle de 0.45s** sin cambios, ademas de **no repetir** la accion mientras el gesto se mantiene (`_last_fired`) y un **rate limit de 1.5s** entre acciones, evitando tanto comandos repetidos como el salto de canciones al abrir/cerrar la mano.

### 7.4 Perceptron de volumen (mano izquierda — modo music)

- 2 features: `w0*distancia + w1` (bias), con distancia normalizada `index_thumb_distance`.
- Entrenado como clasificador cerca/lejos (dedos juntos → 0, separados → 100), 3000 epochs, pesos en `models/volume.npz`.
- La salida continua se mapea a `0..1` interpolando entre los extremos `near=0.10` y `far=0.45` (`config/settings.VolumeSettings`).
- Un **EMA** (smoothing 0.35) filtra temblores.
- El valor se **aplica** (`commit`) solo cuando el meñique se une al pulgar (`pinky_thumb_distance < 0.06`), con cooldown de 3s; se rearma al separarlos.

### 7.5 Perceptron del modo line (entrenamiento en vivo)

- `core/perceptron.Perceptering` entrena un perceptron binario en vivo entre la punta del pulgar (P4) y la del indice (P8): 24 puntos sobre el segmento + 24 desplazados perpendicular `MARGIN=0.03`, presupuesto de 200 epocas/frame. Se resetea si los dedos estan juntos (`FINGERS_TOGETHER_THRESH=0.04`).

---

## 8. Problema: la deteccion de "3 dedos" y el fine-tuning

### 8.1 Sintoma

En el modo music, el gesto de **3 dedos** (indice + medio + anular extendidos, pulgar y meñique recogidos) se contaba mal: con frecuencia daba **2** o **4** en vez de **3**, y en consecuencia disparaba la saga anterior/incorrecta.

### 8.2 Causa raiz (diagnostico)

La version previa de `core/fingers.py` definia la extension de cada dedo como un **delta en Y** normalizado por la escala global de la mano:

```
no-pulgar:  feature = (pip.y - tip.y) / hand_scale   umbral 0.15
pulgar:     feature = dist(tip→mcp_indice) - dist(ip→mcp_indice)   umbral 0.02
```

Esta formulacion presentaba tres problemas:

1. **Dependencia de la orientacion**: medir "cuanto mas arriba esta la punta que el PIP" solo funciona con la mano aproximadamente vertical. Con la mano inclinada o rotada, dedos claramente extendidos pueden dar un valor por debajo del umbral.
2. **El anular es el dedo mas debil**: anular y meñique comparten los tendones extensores. Al recoger el meñique (como en el gesto de 3), el anular apenas puede levantar la punta por encima de su PIP; el umbral absoluto de 0.15*escala quedaba fuera de su alcance fisiológico → conteo de 2. **Esta era la causa mas comun.**
3. **Falsos positivos del pulgar**: la feature del pulgar (distancia al MCP del indice) es muy sensible: con el pulgar recogido sobre la palma, la diferencia `dist(tip→mcp_indice) - dist(ip→mcp_indice)` podia superar el umbral de 0.02 → conteo de 4. **Esta era la segunda causa.**

Ademas, el umbral era **global** (igual para los 5 dedos), sin tener en cuenta que cada dedo tiene un rango de extension distinto.

### 8.3 Solucion adoptada: angulo de flexion (coseno) por dedo

Se reemplazo la medida absoluta por una **medida de flexion local e independiente de la orientacion**:

Para cada dedo se toman los tres landmarks consecutivos `MCP → PIP → TIP` y se calcula el **coseno del angulo entre los segmentos** `v1 = MCP→PIP` y `v2 = PIP→TIP`:

```
cos = (v1·v2) / (|v1| · |v2|)
```

- Dedo **extendido** (recto): v2 alinea con v1 → `cos ≈ 1`.
- Dedo **flexionado** (curvado hacia la palma): v2 se invierte respecto a v1 → `cos ≤ 0`.

Ventajas:

- **Independiente de la rotacion de la mano**: la rectitud del dedo se mide en su propio marco de referencia, no contra el eje Y de la imagen.
- **Normalizada por construccion** (es un coseno): no depende de la escala global; cada dedo se evalúa respecto a su propia longitud.
- **Robusta al gesto de 3**: el anular, aunque solo se extienda ligeramente en angulo, sigue estando *recto* y por tanto queda por encima del umbral; el pulgar y meñique recogidos quedan *curvados* y no se cuentan.

Se definieron umbrales **por dedo** (fine-tuning de las posiciones):

| Dedo | Landmarks (MCP→PIP→TIP) | Umbral de coseno | Nota |
|------|--------------------------|------------------|------|
| **Pulgar** | 2 → 3 → 4 | **0.50** | El mas exigente: solo cuenta si esta claramente salido/recto; evita falsos positivos al recogerlo sobre la palma |
| **Índice** | 5 → 6 → 8 | **0.45** | |
| **Medio** | 9 → 10 → 12 | **0.45** | |
| **Anular** | 13 → 14 → 16 | **0.35** | Permisivo: limitacion anatomica por los tendones compartidos con el meñique en el gesto de 3 |
| **Meñique** | 17 → 18 → 20 | **0.45** | Exigente: evita que el meñique medio doblado cuente como extendido |

La feature binaria sigue siendo `+1` si `cos > umbral` y `-1` en caso contrario, con lo que se conserva intacto el contrato con las `FingerRNN` (que reciben las mismas secuencias ±1 en la ventana de 8 frames). `count_fingers_geometric` se mantiene consistente con el mismo criterio.

### 8.4 Validacion

Se validó con landmarks sinteticos construidos a mano (objetos con `.x`/`.y`) sin usar la camara:

| Gesto (dedos extendidos) | Conteo geometrico | Features |
|--------------------------|-------------------|----------|
| `[]` (puño) | 0 | `[-1,-1,-1,-1,-1]` |
| `[indice]` | 1 | `[-1, 1,-1,-1,-1]` |
| `[indice, medio]` | 2 | `[-1, 1, 1,-1,-1]` |
| `[indice, medio, anular]` (3) | **3** | `[-1, 1, 1, 1,-1]` |
| `[indice..meñique]` | 4 | `[-1, 1, 1, 1, 1]` |
| `[todos]` (palma abierta) | 5 | `[ 1, 1, 1, 1, 1]` |

Y se probaron variantes del gesto de 3 variando la curvatura de pulgar y meñique (2.0 a 2.6 rad): el conteo se mantuvo en **3** en todas ellas (antes, con el criterio por delta-Y, fallaba). Tambien se verifico que una mano con dedos rectos cuenta 5 en cualquier rotacion → independencia de orientacion confirmada.

Finalmente se ejecutó el pipeline completo (`features → MusicGestureController → HandController.count`, con los pesos `finger_*.npz` cargados) mostrando conteos correctos **0..5**.

### 8.5 Siguientes bugs: puño cerrado y salto de cancion al pausar/play

Dos problemas observados tras el fix del "3" (seccion 8.2-8.4) y su solucion:

1. **El puño cerrado contaba 1**: con el criterio nuevo, un pulgar que se pliega "envuelto" sobre el puño quedaba casi recto (cos ≈ 0.9) y se sumaba como dedo extendido. Fix en `_thumb_extended` (doble condicion): además de rectitud (`cos > 0.50`) se exige que la punta del pulgar quede **mas cerca** del nudillo del indice que la articulacion IP del propio pulgar, `(d_tip - d_ip)/idx_len > 0.08`, con `d_tip = dist(P4, MCP5)`, `d_ip = dist(P3, MCP5)`, `idx_len = dist(MCP5, PIP6)`. Con el pulgar pegado al indice el puño vuelve a leer **0** (validado con geometria sintetica: puño→0, gestos 1..5→1..5, palma→5).
2. **Pausar y poner play saltaba la cancion**: al abrir la mano desde el puño (0 → 5) el pipeline pasaba por 1,2,3,4 y cada conteo sostenido ≥4 frames disparaba PREV/NEXT SONG/SAGA. El gesto 0 en pausa larga también podia re-despachar PAUSE. Fix en `MusicGestureController` (`controllers/gestures.py`):
   - **settle**: la accion solo se despacha si el conteo lleva **0.45s sin cambiar** (`SETTLE_SECONDS`); los conteos intermedios del barrido duran ~5 frames (0.17s) → se filtran.
   - **no re-peticion** (`_last_fired`): un gesto mantenido dispara una unica accion; al volver al mismo conteo tras haber hecho otro gesto, se puede disparar de nuevo (si pasa el settle y el rate limit).
   - **rate limit de 1.5s** entre acciones.

Validacion temporal (landmarks sinteticos a 30 fps): puño 1s → palma → solo `PAUSE, PLAY` (sin saltos); "3" sostenido → una unica `PREV SAGA`; 5→0→5 rapido → `PLAY, PAUSE, PLAY`; 2→3→2 → `NEXT SONG, PREV SAGA, NEXT SONG`.

### 8.6 Cambio de LSTM a RNN simple

En deteccion de dedos se sustituyo la celda `FingerLSTM` por una **RNN simple (Elman)** en `controllers/hand.py` (`Wx, Wh, b, Wy, by`, hidden=8, BPTT):

- Solo 2 matrices de pesos por capa recurrente (sin puertas), suficiente para un clasificador de ventana 8 con features binarias ±1; se entrena mas rapido y con menos riesgo de sobreajuste.
- Inicializacion por semilla fija por dedo (`seed=idx`) y datos sinteticos por grupo (seccion 7.3, config 45 epochs): los flip points por dedo quedan en **6/8..8/8** (reproducibles entre ejecuciones).
- Los pesos `models/finger_0..4.npz` se regeneraron con las claves `Wx, Wh, b, Wy, by`; `FingerLSTM` fue eliminado del codigo y de la documentacion.

---

## 9. Ejecucion y uso

```bash
python main.py              # inicia en hand
python main.py position     # inicia en gestos (2 manos)
python main.py music        # inicia en music player
```

| Tecla | Accion |
|-------|--------|
| `n` | Siguiente modo (hand → line → position → music → hand) |
| `q` | Salir |

### Gestos (mano derecha, modo music)

| Conteo | Accion |
|--------|--------|
| 0 | Pause |
| 1 | Cancion anterior |
| 2 | Siguiente cancion |
| 3 | Saga anterior |
| 4 | Siguiente saga |
| 5 | Play |

### Mano izquierda (modo music)

| Gesto | Accion |
|-------|--------|
| Pinza indice↔pulgar (distancia) | Ajustar volumen (perceptron) |
| Anadir meñique a la pinza | Guardar/aplicar volumen |

---

## 10. Estructura de archivos

```
the-machine/
  main.py
  app/            runner.py, vision.py, registry.py
  presentation/   modes/*.py, ui/*.py
  controllers/    hand.py, gestures.py, volume.py
  core/           fingers.py, perceptron.py, results.py, gestures.py, handedness.py, playlist.py
  config/         palette.py, strings.py, settings.py
  common/         fps.py
  infrastructure/ capture.py, display.py, player.py
  models/         hand_landmarker.task, gesture_recognizer.task, finger_0..4.npz, volume.npz
  music/          01 The Troy Saga/ ... 09 The Ithaca Saga/
  docs/           informe.md, architecture.md, modes.md, layers.md, mediapipe.md, perceptron.md, opencv.md, pygame.md
  CREDITS.md
```

---

## 11. Limitaciones conocidas

1. **Ambiguedad del pulgar en gestos de 1-2 dedos**: si el pulgar se mantiene salido y recto al hacer "1" o "2", cuenta como extendido (conteo +1). Es una limitacion ergonomica clasica del conteo por landmarks; la solucion es recoger el pulgar sobre la palma. (Limite opuesto del problema del puño cerrado: al envolver el pulgar sobre el puño, el `_thumb_extended` exige `d_tip` < `d_ip`, y un pulgar que NO toque el indice vuelve a leerse como +1 → puño "0" devuelve 1.)
2. **Deteccion solo si la mano esta razonablemente visible**: la calidad de los landmarks depende de la confianza de MediaPipe (`min_hand_detection_confidence=0.5`).
3. **Las RNNs se entrenan con datos sinteticos**: el modelo no ve datos reales de camara; el agreement + settle + rate limit compensan el ruido, pero un dataset real de gestos mejoraria la robustez.
4. **Inferencia en CPU**: el downscale (60%) reduce carga pero degrada landmarks en manos lejanas.
5. **Dispositivo de audio**: depende de PulseAudio/SDL; sin auriculares bluetooth cae al dispositivo por defecto.

---

## 12. Trabajo futuro

- Recolectar dataset real de las 6 posturas (0..5 dedos) y reentrenar las `FingerRNN` sobre features reales (en lugar de solo ruido sintetico).
- Calibrar los umbrales por dedo con observaciones empiricas multiusuario (hoy se calibraron con geometria y pruebas sinteticas).
- Añadir tracking de identidad de mano para evitar intercambios Left/Right.
- Mover la inferencia a GPU/TFLite para eliminar el downscale.
- Sustituir la ventana fija y el agreement por una FSM de estados de gesto.
- Exportar el informe de diagnostico (seccion 8) como metrica de precision por gesto.

---

## 13. Creditos

- **EPIC: The Musical** — musica de Jorge Hernan (https://epicthemusical.com), incluida con fines academicos y sin animo de lucro. Ver `CREDITS.md`.
- **MediaPipe Hands / Tasks** — modelo de landmarks y gestos de Google.
- **OpenCV** — vision y renderizado; **pygame** — ventana y audio.