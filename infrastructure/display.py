import cv2
import numpy as np
import pygame


class Window:
    def __init__(self, w: int, h: int, title: str):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((w, h))
        self.w = w
        self.h = h

    def show(self, canvas: np.ndarray) -> None:
        rgb_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_out, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def poll(self):
        return pygame.event.get()
