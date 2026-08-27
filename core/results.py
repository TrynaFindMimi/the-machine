def get_gesture(results, idx):
    if not results.gestures or idx >= len(results.gestures):
        return "None", 0.0
    cats = results.gestures[idx]
    if not cats:
        return "None", 0.0
    return cats[0].category_name, cats[0].score


def to_pixel_points(hand_landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]


def count_hands(results):
    return len(results.hand_landmarks) if results.hand_landmarks else 0
