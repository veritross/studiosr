import numpy as np
import pytest

from studiosr import compute_psnr

image_shapes_for_test = [
    (16, 16, 3),
    (32, 32, 3),
    (64, 64, 3),
    (128, 128, 3),
    (256, 256, 3),
]


@pytest.mark.parametrize("image_shape", image_shapes_for_test)
def test_compute_psnr_zero(image_shape):
    image1 = np.zeros(image_shape, np.uint8)
    image2 = np.ones(image_shape, np.uint8) * 255
    psnr = compute_psnr(image1, image2)
    assert psnr == 0.0


@pytest.mark.parametrize("image_shape", image_shapes_for_test)
def test_compute_psnr_inf(image_shape):
    image1 = np.random.randint(0, 255, size=image_shape).astype(np.uint8)
    image2 = image1.copy()
    psnr = compute_psnr(image1, image2)
    assert np.isinf(psnr)


@pytest.mark.parametrize("image_shape", image_shapes_for_test)
def test_compute_psnr_dtype(image_shape):
    image1 = np.random.randint(0, 255, size=image_shape).astype(np.uint8)
    image2 = np.random.randint(0, 255, size=image_shape).astype(np.uint8)

    psnr1 = compute_psnr(image1, image2)
    psnr2 = compute_psnr(image1 / 255.0, image2 / 255.0)
    assert abs(psnr1 - psnr2) < 1e-12


@pytest.mark.parametrize("image_shape", image_shapes_for_test)
def test_compute_psnr_compatibility(image_shape):
    from skimage.metrics import peak_signal_noise_ratio

    image1 = np.random.randint(0, 255, size=image_shape).astype(np.uint8)
    image2 = np.random.randint(0, 255, size=image_shape).astype(np.uint8)

    psnr1 = compute_psnr(image1, image2)
    psnr2 = peak_signal_noise_ratio(image1, image2)
    assert psnr1 == psnr2
