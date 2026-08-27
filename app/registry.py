from presentation.modes import hand as hand_mode
from presentation.modes import line as line_mode
from presentation.modes import position as position_mode

TESTS = {
    "hand": hand_mode.draw,
    "line": line_mode.draw,
    "position": position_mode.draw,
}
TEST_NAMES = list(TESTS)
