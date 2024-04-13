from typing import Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim


def is_rgb(im: np.ndarray) -> bool:
    return len(im.shape) == 3 and im.shape[-1] == 3


def to_y(image: np.ndarray) -> np.ndarray:
    if not is_rgb(image):
        return image
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    y = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    return y


def crop_img_to_equal(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff_x = abs(im1.shape[0] - im2.shape[0])
    diff_y = abs(im1.shape[1] - im2.shape[1])
    if im1.shape[0] > im2.shape[0]:
        im1 = im1[:-(diff_x), :]
    elif im1.shape[0] < im2.shape[0]:
        im2 = im2[:-(diff_x), :]

    if im1.shape[1] > im2.shape[1]:
        im1 = im1[:, :-(diff_y)]
    elif im1.shape[1] < im2.shape[1]:
        im2 = im2[:, :-(diff_y)]

    return im1, im2


def compute_psnr(im1: np.ndarray, im2: np.ndarray, y_only: bool = False, crop_border: int = 0) -> np.float64:
    im1, im2 = crop_img_to_equal(im1, im2)
    if crop_border:
        im1 = im1[crop_border:-crop_border, crop_border:-crop_border]
        im2 = im2[crop_border:-crop_border, crop_border:-crop_border]
    if y_only:
        im1, im2 = to_y(im1), to_y(im2)
    elif im1.dtype != np.uint8:
        im1, im2 = im1 * 255.0, im2 * 255.0
    error = np.mean((im1.astype(np.float32) - im2.astype(np.float32)) ** 2)
    if error == 0:
        return np.inf
    p = 20 * np.log10(255.0 / np.sqrt(error))
    return p


def compute_ssim(im1: np.ndarray, im2: np.ndarray, y_only: bool = False, crop_border: int = 0) -> np.float64:
    im1, im2 = crop_img_to_equal(im1, im2)
    if crop_border:
        im1 = im1[crop_border:-crop_border, crop_border:-crop_border]
        im2 = im2[crop_border:-crop_border, crop_border:-crop_border]
    if y_only:
        im1, im2 = to_y(im1), to_y(im2)
    channel_axis = 2 if is_rgb(im1) else None
    s = ssim(
        im1,
        im2,
        K1=0.01,
        K2=0.03,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        channel_axis=channel_axis,
        data_range=255,
    )
    return s
