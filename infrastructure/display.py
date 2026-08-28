"""infrastructure/display.py — adapter de ventana (pygame+cv2)."""

from __future__ import annotations

import cv2
import numpy as np
import pygame
from numpy.typing import NDArray


class Window:
    """Ventana pygame que muestra un canvas BGR."""

    def __init__(self, w: int, h: int, title: str) -> None:
        if w <= 0 or h <= 0:
            raise ValueError("dimensiones de ventana deben ser > 0")
        self.w: int = int(w)
        self.h: int = int(h)
        self.title: str = title
        # pygame.init es idempotente; se llama una vez
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((w, h))

    def show(self, canvas: NDArray[np.uint8]) -> None:
        rgb_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # pygame espera (width, height, 3) en RGB tras transpose
        surface = pygame.surfarray.make_surface(np.transpose(rgb_out, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def poll(self) -> list[pygame.event.Event]:
        return pygame.event.get()

    def close(self) -> None:
        pygame.quit()

    # Context manager
    def __enter__(self) -> Window:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
