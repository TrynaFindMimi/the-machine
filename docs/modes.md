# Modos de presentacion (tests/)

Cada archivo en `tests/` es un "modo" de visualizacion. No son tests unitarios.

## Estructura comun

Cada modo expone una funcion `draw(frame, results)` que:
1. Recibe el frame BGR de OpenCV y los resultados de MediaPipe
2. Dibuja sobre el frame (in-place)
3. Retorna el frame anotado

```python
def draw(frame, results) -> np.ndarray:
    ...
    return frame
```

## Modos disponibles

### hand (tests/hand_test.py)

Modo basico. Dibuja esqueleto + landmarks numerados de todas las manos detectadas.

```
Webcam → MediaPipe → draw_skeleton + draw_landmarks → Pygame
```

### grid (tests/grid_test.py)

Muestra esqueleto + landmarks con etiqueta de coordenadas (x,y) por cada punto. Sin ejes ni lineas de referencia.

```
Webcam → MediaPipe → draw_skeleton + draw_grid_landmarks → Pygame
```

### line (tests/line_test.py)

Entrena un perceptron por mano en tiempo real y dibuja la frontera de decision como linea azul fina entre el pulgar (P5) y el indice (P9). Soporta 2 manos simultaneamente, cada una con su propio perceptron persistente.

```
Webcam → MediaPipe → por cada mano (hasta 2):
    → draw_skeleton + draw_landmarks
    → build_dataset(p5, p9)
    → Perceptron.train_budget()
    → draw_line(azul, thickness=1)
    → Pygame
```

## Cambio de modo

Se cambia con la tecla `n` durante la ejecucion. El ciclo es: hand → grid → line → hand...

## Modo inicial

Se puede especificar por linea de comandos:

```bash
python main.py hand    # inicia en modo hand
python main.py grid    # inicia en modo grid
python main.py line    # inicia en modo line
```
