import os
from urllib import request

import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    size = block_num * block_size / 1000000
    progress = block_num * block_size / total_size * 100
    print("  %4.1f / 100.0  --  %6.1f MB" % (progress, size), end="\r")


def download_weights(src: str, dst: str) -> None:
    if not os.path.exists(dst):
        request.urlretrieve(src, dst, show_progress)
