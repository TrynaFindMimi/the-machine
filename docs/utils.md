# Archivos utils/ — Descripcion y uso

## utils/text.py

Constantes de texto centralizadas. Todos los strings UI importados desde aqui.

```python
from utils.text import WINDOW, USAGE, HAND_TITLE, GRID_TITLE, LINE_TITLE, FPS_FMT
from utils.text import ROOT_PROMPT, KEYBIND_HINT
```

## utils/hand.py

Dibujo de manos (esqueleto y landmarks). Funciones principales:

```python
from utils.hand import draw_hands          # dibuja manos completas (skeleton + landmarks)
from utils.hand import draw_skeleton       # solo el esqueleto (conexiones, thickness=1)
from utils.hand import draw_landmarks      # solo los puntos numerados
from utils.hand import draw_label          # etiqueta de texto al lado de un punto
from utils.hand import spaced              # convierte "hola" → "H O L A"
from utils.hand import WHITE, FONT         # constantes de estilo
```

## utils/grid.py

Etiquetas de coordenadas para el modo grid. Solo dibuja circles blancos con etiqueta (x,y) por cada landmark.

```python
from utils.grid import draw_grid_landmarks    # dibuja landmarks con etiqueta (x,y)
```

## utils/line.py

Perceptron y utilidades geometricas para la linea de decision.

```python
from utils.line import Perceptron             # clase perceptron binario (train + train_budget)
from utils.line import build_dataset          # genera datos de entrenamiento desde P5 y P9
from utils.line import boundary_points        # calcula interseccion frontera↔frame
from utils.line import draw_line              # dibuja linea entre dos puntos
from utils.line import BLUE, FINGERS_TOGETHER_THRESH
```

## utils/fps.py

Contador de FPS con suavizado por media movil exponencial.

```python
from utils.fps import FPSCounter

meter = FPSCounter()
fps = meter.tick()   # retorna FPS actual (float)
```

## utils/style.py

Tema visual "root" (terminal verde/ambar). Incluye header, footer, scanlines y vignette.

```python
from utils.style import apply_root_overlay     # oscurece frame + scanlines + vignette
from utils.style import draw_root_header       # barra superior con prompt root
from utils.style import draw_root_footer       # barra inferior con keybindings
```

Nota: `style.py` esta preparado pero no esta integrado en ningun modo activo actualmente.
