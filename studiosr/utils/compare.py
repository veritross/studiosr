import os
import time

import cv2
import numpy as np


def clip(x, x_min, x_max):
    return min(max(x, x_min), x_max)


class MouseHandler:
    def __init__(self, width, height, crop_size=64):
        self.w = width
        self.h = height
        self.x = self.w // 2
        self.y = self.h // 2
        self.s = crop_size

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.x = x % self.w
            self.y = y % self.h

    def get_crop_rect(self):
        x, y, s = self.x, self.y, self.s
        h, w = self.h, self.w

        x1 = x - s // 2
        y1 = y - s // 2

        x1 = clip(x1, 0, w - s)
        y1 = clip(y1, 0, h - s)

        x2 = x1 + s
        y2 = y1 + s

        return x1, y1, x2, y2

    def modify_crop_size(self, diff):
        min_crop_size = 8
        max_crop_size = min(self.w, self.h)
        s = clip(self.s + diff, min_crop_size, max_crop_size)
        self.s = s


def compare(images: list, crop_size: int = 64, zoom_size: int = 256):
    cv2.namedWindow("image-compare")
    cv2.namedWindow("image-crops")
    mh = MouseHandler(images[0].shape[1], images[0].shape[0], crop_size)
    cv2.setMouseCallback("image-compare", mh.mouse_event, None)

    while True:
        x1, y1, x2, y2 = mh.get_crop_rect()
        views = []
        crops = []
        for image in images:
            view = image.copy()
            crops.append(
                cv2.resize(
                    image[y1:y2, x1:x2],
                    (zoom_size, zoom_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            )

            cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
            views.append(view)

        views = np.concatenate(views, 1)
        crops = np.concatenate(crops, 1)
        cv2.imshow("image-compare", views)
        cv2.imshow("image-crops", crops)
        key = cv2.waitKey(30)

        if key == 27:
            break
        elif key == ord("c") or key == ord("C"):
            capture_dir = "./captures"
            os.makedirs(capture_dir, exist_ok=True)
            capture_path = os.path.join(capture_dir, f"{time.time_ns()}.png")
            cv2.imwrite(capture_path, crops)
            print("Capture Image ->", capture_path)
        elif key == ord("a") or key == ord("A"):
            mh.modify_crop_size(-4)
        elif key == ord("s") or key == ord("S"):
            mh.modify_crop_size(+4)
        elif key == ord("z") or key == ord("Z"):
            zoom_size = clip(zoom_size - 4, 32, 512)
        elif key == ord("x") or key == ord("X"):
            zoom_size = clip(zoom_size + 4, 32, 512)
