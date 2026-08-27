# Capas y módulos

| Capa | Directorio | Módulos | Imports permitidos | Descripción |
|------|------------|---------|--------------------|-------------|
| Config | `config/` | `palette.py`, `strings.py` | nada | Constantes BGR, textos UI, `MODES_ORDER` |
| Transversal | `common/` | `fps.py` | stdlib | `FPSCounter` EMA |
| Dominio | `core/` | `perceptron.py`, `results.py`, `handedness.py`, `gestures.py` | `config`, `common`, `numpy` | Lógica pura sin cv2/pygame. `handedness.py` aísla corrección flip `get_handedness`/`normalize_handedness`; `results.py` solo `get_gesture`/`to_pixel_points`/`count_hands` |
| Presentación UI | `presentation/ui/` | `theme.py`, `drawing.py`, `effects.py`, `layout.py` | `config`, `core` | Dibujo CCTV minimalista |
| Presentación Modes | `presentation/modes/` | `hand.py`, `line.py`, `position.py` | `core`, `config`, `common`, `presentation/ui` | Un `draw()` por modo |
| Aplicación | `app/` | `registry.py`, `vision.py`, `runner.py` | todos los anteriores + `infrastructure` | Orquesta captura→visión→modo→UI |
| Infraestructura | `infrastructure/` | `capture.py`, `display.py` | `config` | Adapters cv2/pygame |
| Fachada | `main.py` | — | `app`, `config` | Solo parsea args |

Añadir nuevo modo: crear `presentation/modes/nuevo.py` con `draw()`, registrar en `app/registry.py`, añadir título en `config/strings.py`.
