"""presentation/ui/theme.py — constantes de tipografía y layout UI."""

from __future__ import annotations

from typing import Final

import cv2

from config.settings import WINDOW

FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
# Re-export desde config para no romper imports, pero fuente canónica es config.settings.WINDOW
SIDEBAR_W: Final = WINDOW.sidebar_width
