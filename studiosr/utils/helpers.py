import logging
import os
from typing import List, Optional
from urllib import request

import cv2
import numpy as np
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def download(src: str, dst: str) -> None:
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


class Logger:
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        use_console=False,
    ) -> None:
        self.logger = logging.getLogger("custom_logger")
        self.logger.setLevel(log_level)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter("%(asctime)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        if use_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def log(self, level: int, message: str) -> None:
        self.logger.log(level, message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


def get_image_extensions() -> List[str]:
    return [
        ".bmp",
        ".dib",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".jp2",
        ".png",
        ".webp",
        ".avif",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pxm",
        ".pnm",
        ".pfm",
        ".sr",
        ".ras",
        ".tiff",
        ".tif",
        ".exr",
        ".hdr",
        ".pic",
    ]


def get_image_files(root: str) -> List[str]:
    image_files = []
    for (root, dirs, files) in os.walk(root):
        for f in files:
            extension = os.path.splitext(f)[1].lower()
            if extension in get_image_extensions():
                image_files.append(f)
    print(image_files)
    print(root)
    return image_files
