# Capas y módulos

| Capa | Directorio | Módulos | Imports permitidos | Descripción |
|------|------------|---------|--------------------|-------------|
| Config | `config/` | `palette.py`, `strings.py`, `settings.py` | nada | Constantes BGR, textos UI, `MODES_ORDER`, `AudioSettings` |
| Transversal | `common/` | `fps.py` | stdlib | `FPSCounter` EMA |
| Dominio | `core/` | `perceptron.py`, `results.py`, `handedness.py`, `gestures.py`, `fingers.py`, `playlist.py` | `config`, `common`, `numpy` | Lógica pura sin cv2/pygame. `handedness.py` aísla corrección flip `get_handedness`/`normalize_handedness`; `results.py` solo `get_gesture`/`to_pixel_points`/`count_hands`; `fingers.py` features binarias por dedo + `ring_thumb_distance`/`pinky_ring_distance`/`hand_scale`; `playlist.py` `scan_sagas` |
| Controladores | `controllers/` | `hand.py`, `gestures.py`, `volume.py` | `config`, `core`, `numpy` | `FingerLSTM` x5 + `HandController` (count/action/train/save/load); `MusicGestureController` (ventana, agreement, cooldown, acción pendiente); `VolumeController` (Perceptron 2-f, commit por meñique, re-arm) |
| Presentación UI | `presentation/ui/` | `theme.py`, `drawing.py`, `effects.py`, `layout.py` | `config`, `core` | Dibujo CCTV minimalista |
| Presentación Modes | `presentation/modes/` | `hand.py`, `line.py`, `position.py`, `music.py` | `controllers`, `core`, `config`, `common`, `presentation/ui` | Un `draw()` por modo. `music.py` NO importa `infrastructure`; expone `set_context`/`consume_pending_action`/`consume_pending_volume` |
| Aplicación | `app/` | `registry.py`, `vision.py`, `runner.py` | todos los anteriores + `infrastructure` | Orquesta captura→visión→modo→UI; crea `MusicPlayer`, inyecta contexto music, despacha acciones |
| Infraestructura | `infrastructure/` | `capture.py`, `display.py`, `player.py` | `config`, `core/playlist` | Adapters cv2/pygame. `player.py` zquets `MusicPlayer` sobre `pygame.mixer` con `AUDIO` de settings |
| Fachada | `main.py` | — | `app`, `config` | Solo parsea args |

Añadir nuevo modo: crear `presentation/modes/nuevo.py` con `draw()`, registrar en `app/registry.py`, añadir título en `config/strings.py`.

Regla de capas para el modo music: `presentation/modes/music.py` no conoce `infrastructure` — la decisión gesto→acción vive en `controllers/gestures.MusicGestureController` (`.pending`) y en `controllers/volume.VolumeController` (`.pending`), y el runner (`app/runner.py`) las consume (`consume_pending_action`/`consume_pending_volume`) y aplica sobre `MusicPlayer`. `infrastructure/player.py` lee el dispositivo de audio de `config/settings.AUDIO`.
