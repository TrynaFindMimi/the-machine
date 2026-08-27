# Arquitectura del proyecto

## Capas

```
main.py              → fachada mínima (args → app.runner.run)
app/                 → orquestación: runner + vision + registry
presentation/        → UI: modes/ (hand/line/position) + ui/ (theme/layout/drawing/effects)
core/                → dominio puro: perceptron, results, gestures, handedness (sin cv2/pygame)
config/              → configuración: palette, strings (hojas)
common/              → transversal: fps
infrastructure/      → adapters: capture (cv2.VideoCapture), display (pygame)
models/              → modelos .task + legacy
```

### Flujo de datos

```
Webcam → infrastructure/capture.Camera.read() [flip+resize 1060x720]
    ↓
app/vision.py → HandLandmarker (hand/line) | GestureRecognizer (position, 2 manos)
    ↓
presentation/modes/*.draw(frame, results) → (frame, hand_count)  [bbox viewfinder + landmarks 2px + crosshair, paleta B/W]
    ↓
presentation/ui/effects.apply_cctv_effect() [scanlines+viñeta+tinte verde]
    ↓
presentation/ui/layout → draw_sidebar (MODE/HANDS/FPS)  [B/W: WHITE/GRAY sobre BLACK]
    ↓
infrastructure/display.Window.show(canvas 1280x720) [BGR→RGB→pygame]
```

### Dependencias entre capas

```
config/palette.py, config/strings.py, common/fps.py          ← hojas
core/perceptron.py → common (nada) | core/results.py → (nada) | core/handedness.py → (nada) | core/gestures.py → config/palette
presentation/ui/theme.py → (nada) | presentation/ui/drawing.py → config/palette | presentation/ui/effects.py → config/palette
presentation/ui/layout.py → config/palette + config/strings + presentation/ui/theme  # solo draw_sidebar (B/W)
presentation/modes/*.py → core/* + config/* + common/fps + presentation/ui/*   # hand/position → core/handedness + core/results
app/registry.py → presentation/modes/* | app/vision.py → mediapipe | app/runner.py → app/* + infrastructure + presentation/ui/*
infrastructure/capture.py → cv2 | infrastructure/display.py → pygame+cv2
main.py → app/registry + app/runner + config/strings
```

Reglas: `core` nunca importa `presentation`; `config/common` nunca importan capas superiores; `presentation` no importa `app/infrastructure`. `core/handedness.py` aísla la corrección de flip (Left↔Right) para que `core/results.py` solo haga conversión geométrica/gestos. UI minimalista B/W sin header/footer (solo sidebar).

### Estilo visual

CCTV minimalista monocromo: scanlines cada 4px (`alpha 0.12`), viñeta gaussiana, tipografía `FONT_HERSHEY_SIMPLEX` `0.3-0.5` scale (position `0.60-0.70` bold con caja `BLACK` para legibilidad), bounding box `1-2px` + esquinas `12px`. Paleta predominante `BLACK`/`WHITE`/`GRAY`; `position` sin borde negro en esqueleto/landmarks (solo `WHITE 2px`) y sin color por gesto. Sidebar `SIDEBAR_BG`/`SIDEBAR_BORDER` en grises sin header/footer.
