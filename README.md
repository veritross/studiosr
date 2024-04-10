# StudioSR
StudioSR is a PyTorch library providing implementations of training and evaluation of super-resolution models. StudioSR aims to offer an identical playground for modern super-resolution models so that researchers can readily compare and analyze a new idea. (inspired by [PyTorch-StudioGan](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN))


## Installation

### From [PyPI](https://pypi.org/project/studiosr/)
```bash
pip install studiosr
```

### From source (Editable)
```bash
git clone https://github.com/veritross/studiosr.git
cd studiosr
python3 -m pip install -e .
```


## Documentation
Documentation along with a quick start guide can be found in the [docs/](./docs/) directory.

### Quick Example

```bash
$ python -m studiosr --image image.png --scale 4 --model swinir
```

```python
from studiosr.models import SwinIR
from studiosr.utils import imread, imwrite

model = SwinIR.from_pretrained(scale=4).eval()
image = imread("image.png")
upscaled = model.inference(image)
imwrite("upscaled.png", upscaled)
```


## Benchmark
- The evaluation metric is [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).
- "DIV2K_mini" is a subset of DIV2K validation data.
- You can check the full benchmark [here](./docs/benchmark.md).

| Method | Scale | Set5   | Set14  | BSD100 | Urban100 | Manga109 | DIV2K_mini | Train |
| ------ | ----- | ------ | ------ | ------ | -------- | -------- | ---------- | -------- |
| VDSR   | x 4   | 31.860 | 28.424 | 27.431 | 25.729   | 29.973   | 34.188     | DF2K     |
| EDSR   | x 4   | 32.640 | 28.913 | 27.785 | 26.801   | 31.318   | 34.873     | DF2K     |
| RCAN   | x 4   | 32.602 | 28.825 | 27.739 | 26.736   | 31.127   | 34.916     | DIV2K    |
| HAN    | x 4   | 32.567 | 28.864 | 27.771 | 26.767   | 31.364   | 34.906     | DIV2K    |
| SwinIR | x 4   | 32.894 | 29.066 | 27.912 | 27.448   | 31.947   | 35.151     | DF2K     |
| HAT    | x 4   | 32.960 | 29.206 | 27.974 | 27.953   | 32.409   | 35.358     | DF2K     |

| Method | Scale | Set5   | Set14  | BSD100 | Urban100 | Manga109 | DIV2K_mini | Train |
| ------ | ----- | ------ | ------ | ------ | -------- | -------- | ---------- | -------- |
| VDSR   | x 3   | 34.124 | 30.155 | 28.990 | 27.806   | 33.109   | 30.338     | DF2K     |
| EDSR   | x 3   | 34.733 | 30.633 | 29.315 | 29.015   | 34.491   | 31.015     | DF2K     |
| RCAN   | x 3   | 34.707 | 30.600 | 29.297 | 29.005   | 34.340   | 31.033     | DIV2K    |
| HAN    | x 3   | 34.707 | 30.610 | 29.299 | 29.020   | 34.368   | 31.041     | DIV2K    |
| SwinIR | x 3   | 34.890 | 30.905 | 29.457 | 29.755   | 35.029   | 31.292     | DF2K     |
| HAT    | x 3   | 34.990 | 31.042 | 29.522 | 30.227   | 35.444   | 31.444     | DF2K     |

| Method | Scale | Set5   | Set14  | BSD100 | Urban100 | Manga109 | DIV2K_mini | Train |
| ------ | ----- | ------ | ------ | ------ | -------- | -------- | ---------- | -------- |
| VDSR   | x 2   | 37.819 | 33.447 | 32.102 | 31.725   | 38.308   | 28.331     | DF2K     |
| EDSR   | x 2   | 38.177 | 34.139 | 32.396 | 33.168   | 39.407   | 28.973     | DF2K     |
| RCAN   | x 2   | 38.167 | 34.080 | 32.376 | 33.160   | 39.310   | 28.932     | DIV2K    |
| HAN    | x 2   | 38.153 | 34.092 | 32.370 | 33.152   | 39.307   | 28.977     | DIV2K    |
| SwinIR | x 2   | 38.292 | 34.371 | 32.515 | 33.788   | 39.773   | 29.233     | DF2K     |
| HAT    | x 2   | 38.471 | 34.798 | 32.590 | 34.401   | 40.102   | 29.357     | DF2K     |

## License
StudioSR is an open-source library under the **MIT license**. 
