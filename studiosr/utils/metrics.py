import numpy as np
from skimage.metrics import structural_similarity as ssim


def is_rgb(im):
    return len(im.shape) == 3 and im.shape[-1] == 3


def to_y(image):
    if not is_rgb(image):
        return image
    if image.dtype == np.uint8:
        image = image / 255.0
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    y = y.clip(0, 255).round().astype(np.uint8)
    return y


def crop_img_to_equal(im1, im2):
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


def compute_psnr(im1, im2, y_only=False, crop_border=0):
    im1, im2 = crop_img_to_equal(im1, im2)
    if crop_border:
        im1 = im1[crop_border:-crop_border, crop_border:-crop_border]
        im2 = im2[crop_border:-crop_border, crop_border:-crop_border]
    if y_only:
        im1, im2 = to_y(im1), to_y(im2)
    if im1.dtype == np.uint8:
        data_range = 255.0
    else:
        data_range = 1.0
    error = np.mean((im1.astype(np.float64) - im2.astype(np.float64)) ** 2)
    if error == 0:
        return np.inf
    p = 10 * np.log10((data_range**2) / error)
    return p


def compute_ssim(im1, im2, y_only=False, crop_border=0):
    im1, im2 = crop_img_to_equal(im1, im2)
    if crop_border:
        im1 = im1[crop_border:-crop_border, crop_border:-crop_border]
        im2 = im2[crop_border:-crop_border, crop_border:-crop_border]
    if y_only:
        im1, im2 = to_y(im1), to_y(im2)
    if is_rgb(im1):
        channel_axis = 2
    else:
        channel_axis = None
    s = ssim(
        im1,
        im2,
        K1=0.01,
        K2=0.03,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        channel_axis=channel_axis,
    )
    return s
