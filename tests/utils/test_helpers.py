import pytest
import torch.nn as nn

from studiosr.utils import count_parameters


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, bias",
    (
        [3, 64, 1, True],
        [64, 3, 1, True],
        [3, 64, 3, True],
        [64, 3, 3, True],
        [3, 64, 1, False],
        [64, 3, 1, False],
        [3, 64, 3, False],
        [64, 3, 3, False],
    ),
)
def test_count_parameters_conv2d(in_channels, out_channels, kernel_size, bias):
    model = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
    num_parameters = in_channels * out_channels * (kernel_size**2)
    num_parameters = num_parameters + out_channels if bias else num_parameters
    assert count_parameters(model) == num_parameters
