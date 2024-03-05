# StudioSR
StudioSR is a Pytorch library providing implementations of training and evaluation of super-resolution models. StudioSR aims to offer an identical playground for modern super-resolution models so that researchers can readily compare and analyze a new idea. (inspired by [PyTorch-StudioGan](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN))


## Installation

### From PyPI
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
- "+" indicates the result of self-ensemble.

| Method  | Scale | Set5   | Set14  | BSD100 | Urban100 | Manga109 | Training Dataset |
| ------- | ----- | ------ | ------ | ------ | -------- | -------- | ---------------- |
| VDSR    | x 4   | 31.860 | 28.424 | 27.431 | 25.729   | 29.973   | DF2K             |
| VDSR+   | x 4   | 31.950 | 28.491 | 27.471 | 25.809   | 30.182   | DF2K             |
| EDSR    | x 4   | 32.452 | 28.790 | 27.718 | 26.635   | 30.985   | DIV2K            |
| EDSR+   | x 4   | 32.612 | 28.925 | 27.798 | 26.859   | 31.398   | DIV2K            |
| RCAN    | x 4   | 32.602 | 28.825 | 27.739 | 26.736   | 31.127   | DIV2K            |
| RCAN+   | x 4   | 32.702 | 28.940 | 27.821 | 27.020   | 31.563   | DIV2K            |
| HAN     | x 4   | 32.567 | 28.864 | 27.771 | 26.767   | 31.364   | DIV2K            |
| HAN+    | x 4   | 32.689 | 28.940 | 27.820 | 26.935   | 31.687   | DIV2K            |
| SwinIR  | x 4   | 32.894 | 29.066 | 27.912 | 27.448   | 31.947   | DF2K             |
| SwinIR+ | x 4   | 32.899 | 29.117 | 27.942 | 27.564   | 32.147   | DF2K             |
| HAT     | x 4   | 32.960 | 29.206 | 27.974 | 27.953   | 32.409   | DF2K             |
| HAT+    | x 4   | 33.075 | 29.253 | 28.015 | 28.087   | 32.600   | DF2K             |

| Method  | Scale | Set5   | Set14  | BSD100 | Urban100 | Manga109 | Training Dataset |
| ------- | ----- | ------ | ------ | ------ | -------- | -------- | ---------------- |
| VDSR    | x 3   | 34.124 | 30.155 | 28.990 | 27.806   | 33.109   | DF2K             |
| VDSR+   | x 3   | 34.227 | 30.217 | 29.029 | 27.896   | 33.353   | DF2K             |
| EDSR    | x 3   | 34.617 | 30.510 | 29.258 | 28.809   | 34.116   | DIV2K            |
| EDSR+   | x 3   | 34.739 | 30.652 | 29.327 | 29.029   | 34.470   | DIV2K            |
| RCAN    | x 3   | 34.707 | 30.600 | 29.297 | 29.005   | 34.340   | DIV2K            |
| RCAN+   | x 3   | 34.803 | 30.703 | 29.362 | 29.229   | 34.658   | DIV2K            |
| HAN     | x 3   | 34.707 | 30.610 | 29.299 | 29.020   | 34.368   | DIV2K            |
| HAN+    | x 3   | 34.802 | 30.708 | 29.367 | 29.240   | 34.676   | DIV2K            |
| SwinIR  | x 3   | 34.890 | 30.905 | 29.457 | 29.755   | 35.029   | DF2K             |
| SwinIR+ | x 3   | 34.971 | 30.960 | 29.479 | 29.887   | 35.166   | DF2K             |
| HAT     | x 3   | 34.990 | 31.042 | 29.522 | 30.227   | 35.444   | DF2K             |
| HAT+    | x 3   | 35.070 | 31.092 | 29.550 | 30.326   | 35.571   | DF2K             |

| Method | Scale | Set5   | Set14  | BSD100 | Urban100 | Manga109 | Training Dataset |
| ------ | ----- | ------ | ------ | ------ | -------- | -------- | ---------------- |
| VDSR   | x 2   | 37.819 | 33.447 | 32.102 | 31.725   | 38.308   | DF2K             |
| VDSR+  | x 2   | 37.891 | 33.528 | 32.142 | 31.836   | 38.544   | DF2K             |
| EDSR   | x 2   | 38.096 | 33.900 | 32.341 | 32.948   | 39.065   | DIV2K            |
| EDSR+  | x 2   | 38.184 | 34.003 | 32.387 | 33.129   | 39.247   | DIV2K            |
| RCAN   | x 2   | 38.167 | 34.080 | 32.376 | 33.160   | 39.310   | DIV2K            |
| RCAN+  | x 2   | 38.222 | 34.155 | 32.419 | 33.388   | 39.474   | DIV2K            |
| HAN    | x 2   | 38.153 | 34.092 | 32.370 | 33.152   | 39.307   | DIV2K            |
| HAN+   | x 2   | 38.210 | 34.164 | 32.417 | 33.383   | 39.479   | DIV2K            |


## License
StudioSR is an open-source library under the **MIT license**. 
