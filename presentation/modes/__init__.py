"""presentation/modes — contrato de modo.

Cada modo expone `draw(frame, results) -> (frame, hand_count)`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ModeHandler(Protocol):
    def __call__(self, frame: NDArray[np.uint8], results) -> tuple[NDArray[np.uint8], int]: ...
