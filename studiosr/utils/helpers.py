import logging
import os
import tempfile
import zipfile
from typing import List, Optional

import cv2
import gdown
import numpy as np
import requests
import torch
import torch.nn as nn
from tqdm import tqdm


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def download(src: str, dst: str) -> None:
    response = requests.get(src, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(dst, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def gdown_and_extract(id: str, save_dir: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "tmp.zip")
        gdown.download(id=id, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)


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
        use_console: bool = False,
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
    return [".bmp", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".tiff", ".tif"]


def get_image_files(root: str) -> List[str]:
    image_files = []
    for (root, dirs, files) in os.walk(root):
        for f in files:
            extension = os.path.splitext(f)[1].lower()
            if extension in get_image_extensions():
                image_files.append(f)
    return sorted(image_files)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
