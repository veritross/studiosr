import random

import numpy as np
import torch


def paired_random_crop(lq, gt, size=48, scale=4):
    h, w, c = lq.shape
    xs = random.randint(0, w - size)
    ys = random.randint(0, h - size)
    xe = xs + size
    ye = ys + size
    lq = lq[ys:ye, xs:xe]
    gt = gt[ys * scale : ye * scale, xs * scale : xe * scale]
    return lq, gt


def paired_random_fliplr(lq, gt, p=0.5):
    if random.random() < p:
        lq = np.fliplr(lq)
        gt = np.fliplr(gt)
    return lq, gt


def paired_random_flipud(lq, gt, p=0.5):
    if random.random() < p:
        lq = np.flipud(lq)
        gt = np.flipud(gt)
    return lq, gt


def paired_random_rot90(lq, gt, p=0.5):
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
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lq, gt):
        for transform in self.transforms:
            lq, gt = transform(lq, gt)
        return lq, gt


class RandomCrop:
    def __init__(self, size=48, scale=4):
        self.size = size
        self.scale = scale

    def __call__(self, lq, gt):
        return paired_random_crop(lq, gt, self.size, self.scale)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lq, gt):
        return paired_random_fliplr(lq, gt, self.p)


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lq, gt):
        return paired_random_flipud(lq, gt, self.p)


class RandomRotation90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lq, gt):
        return paired_random_rot90(lq, gt, self.p)


class ToTensor:
    def __call__(self, lq, gt):
        lq = array2tensor(lq)
        gt = array2tensor(gt)
        return lq, gt
