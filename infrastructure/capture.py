import cv2


class Camera:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.vc = cv2.VideoCapture(0)
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self):
        ret, frame = self.vc.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        return cv2.resize(frame, (self.w, self.h))

    def release(self):
        self.vc.release()
