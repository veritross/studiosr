import random
from typing import Callable, List, Tuple, Union

import numpy as np
import torch


def paired_random_crop(
    lq: np.ndarray,
    gt: np.ndarray,
    size: int = 48,
    scale: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, c = lq.shape
    xs = random.randint(0, w - size)
    ys = random.randint(0, h - size)
    xe = xs + size
    ye = ys + size
    lq = lq[ys:ye, xs:xe]
    gt = gt[ys * scale : ye * scale, xs * scale : xe * scale]
    return lq, gt


def paired_random_fliplr(
    lq: np.ndarray,
    gt: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        lq = np.fliplr(lq)
        gt = np.fliplr(gt)
    return lq, gt


def paired_random_flipud(
    lq: np.ndarray,
    gt: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        lq = np.flipud(lq)
        gt = np.flipud(gt)
    return lq, gt


def paired_random_rot90(
    lq: np.ndarray,
    gt: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        lq = np.rot90(lq)
        gt = np.rot90(gt)
    return lq, gt


def array2tensor(array: np.ndarray) -> torch.Tensor:
    array = array.transpose(2, 0, 1)
    array = array.astype(np.float32) / 255
    tensor = torch.from_numpy(array)
    return tensor


class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        lq: Union[np.ndarray, torch.Tensor],
        gt: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        for transform in self.transforms:
            lq, gt = transform(lq, gt)
        return lq, gt


class RandomCrop:
    def __init__(self, size: int = 48, scale: int = 4) -> None:
        self.size = size
        self.scale = scale

    def __call__(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return paired_random_crop(lq, gt, self.size, self.scale)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return paired_random_fliplr(lq, gt, self.p)


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return paired_random_flipud(lq, gt, self.p)


class RandomRotation90:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return paired_random_rot90(lq, gt, self.p)


class ToTensor:
    def __call__(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        lq = array2tensor(lq)
        gt = array2tensor(gt)
        return lq, gt
