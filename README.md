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


## License
StudioSR is an open-source library under the **MIT license**. 
