import os
from urllib import request

import cv2
import numpy as np
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_weights(src: str, dst: str) -> None:
    def show_progress(block_num: int, block_size: int, total_size: int) -> None:
        size = block_num * block_size / 1000000
        progress = block_num * block_size / total_size * 100
        print("  %4.1f / 100.0  --  %6.1f MB" % (progress, size), end="\r")

    if not os.path.exists(dst):
        request.urlretrieve(src, dst, show_progress)


def imread(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imwrite(path: str, image: np.ndarray) -> bool:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = cv2.imwrite(path, image)
    return result
