from config.palette import CYAN, GREEN, RED

GESTURE_COLORS = {
    "None": RED,
    "Closed_Fist": (0, 140, 255),
    "Open_Palm": GREEN,
    "Pointing_Up": CYAN,
    "Thumb_Down": (0, 0, 255),
    "Thumb_Up": (0, 255, 100),
    "Victory": (255, 200, 0),
    "ILoveYou": (200, 100, 255),
}

GESTURES = list(GESTURE_COLORS.keys())
