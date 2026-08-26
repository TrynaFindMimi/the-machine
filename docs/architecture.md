# Arquitectura del proyecto

## Capas

```
main.py                    → Punto de entrada, loop principal, orquestacion
tests/                     → Modos de presentacion (cada uno expone draw())
utils/                     → Logica compartida y utilidades de dibujo
models/                    → Modelo preentrenado y script legacy
```

### Flujo de datos

```
Webcam (cv2.VideoCapture)
    │
    ▼
main.py: captura frame → flip espejo → MediaPipe detecta manos
    │
    ▼
tests/{mode}_test.py: draw(frame, results) → frame anotado
    │
    ▼
utils/: dibujo de manos, coordenadas, perceptron, linea
    │
    ▼
Pygame: surface ← frame BGR→RGB → pantalla
```

### Dependencias entre capas

- `tests/` importa desde `utils/` (nunca al reves)
- `main.py` importa `tests/` y `utils/text.py`
- `utils/line.py` no importa de otros `utils/` (es hoja)
- `utils/hand.py` importa `utils/line.py`
- `utils/grid.py` importa `utils/hand.py`
- `utils/style.py` importa `utils/text.py`
