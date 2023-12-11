import pytest
import torch

from studiosr.models import HAT


@pytest.mark.parametrize(
    ["scale", "input_shape", "output_shape"],
    [
        (2, (1, 3, 12, 12), (1, 3, 24, 24)),
        (3, (1, 3, 12, 12), (1, 3, 36, 36)),
        (4, (1, 3, 12, 12), (1, 3, 48, 48)),
        (8, (1, 3, 12, 12), (1, 3, 96, 96)),
    ],
)
def test_shape_of_hat(scale, input_shape, output_shape):
    n_colors = input_shape[1]
    model = HAT(scale=scale, n_colors=n_colors)
    x = torch.randn(*input_shape)
    y = model(x)
    assert y.shape == output_shape
