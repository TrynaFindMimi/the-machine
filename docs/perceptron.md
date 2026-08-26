# Perceptron — Modo Line

## Que hace

Entrena un perceptron binario en tiempo real para calcular la frontera de decision entre dos puntos de la mano: la yema del pulgar (P5, indice 4) y la yema del indice (P9, indice 8). Soporta las 2 manos simultaneamente.

## Flujo

```
results.hand_landmarks (hasta 2 manos)
    │
    ├─ Mano 0 → build_dataset(p5, p9) → _perc[0].train_budget() → draw_line(azul)
    │
    └─ Mano 1 → build_dataset(p5, p9) → _perc[1].train_budget() → draw_line(azul)
```

## Clase Perceptron

```python
class Perceptron:
    def __init__(self, lr=0.05, max_epochs=3000):
        self.w = np.zeros(3)       # pesos [w_x, w_y, bias]

    def predict(self, x) -> int:   # x = [x, y, 1]
        return 1 if dot(w, x) >= 0 else -1

    def train(self, X, y) -> int:
        # entrena hasta convergencia completa

    def train_budget(self, X, y, budget) -> int:
        # entrena como maximo 'budget' epocas por frame
```

## Persistencia entre frames

Cada mano tiene su propio perceptron en `_perc: list[Perceptron | None] = [None, None]`:
- `_perc[0]` → perceptron de la mano izquierda
- `_perc[1]` → perceptron de la mano derecha
- Se crea la primera vez que se detecta cada mano
- Se reutiliza en frames siguientes (el aprendizaje se acumula)
- Se resetea individualmente cuando esa mano desaparece o junta los dedos

Con `train_budget(X, y, 200)`:
- Frame 1: 200 epocas, puede no converger
- Frame 2: ya tiene pesos del frame anterior, converge mas rapido
- Frame N: converge en ~1-5 epocas porque ya aprendio la frontera

## build_dataset

Genera un dataset linealmente separable:
- **Positivos (+1):** puntos a lo largo del segmento p5→p9
- **Negativos (-1):** puntos desplazados perpendicularmente por `MARGIN=0.03`

## Umbral de dedos juntos

```python
FINGERS_TOGETHER_THRESH = 0.04
```

Si la distancia normalizada entre P5 y P9 es menor a 0.04, no se dibuja la linea.

## Visualizacion

- **Linea azul** (`BLUE = (255, 60, 20)` en BGR, `thickness=1`) entre P5 y P9
- **Esqueleto blanco** y **landmarks numerados** de la mano
- Sin circulos ni marcadores adicionales en los puntos de la linea
