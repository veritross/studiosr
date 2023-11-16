import os
from urllib import request

import numpy as np
import torch
import torch.nn as nn


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    size = block_num * block_size / 1000000
    progress = block_num * block_size / total_size * 100
    print("  %4.1f / 100.0  --  %6.1f MB" % (progress, size), end="\r")


def download_weights(src: str, dst: str) -> None:
    if not os.path.exists(dst):
        request.urlretrieve(src, dst, show_progress)


class BaseModule(nn.Module):
    @torch.no_grad()
    def inference(self, image: np.ndarray) -> np.ndarray:
        scale = 255.0 if self.img_range == 1.0 else 1.0
        device = next(self.parameters()).get_device()
        device = torch.device("cpu") if device < 0 else device
        x = image.transpose(2, 0, 1).astype(np.float32) / scale
        x = torch.from_numpy(x).unsqueeze(0)
        x = x.to(device)
        output = self.forward(x) * scale
        y = output.squeeze().cpu().numpy().round().clip(0, 255)
        return y.astype(np.uint8).transpose(1, 2, 0)
