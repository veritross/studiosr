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

### Train
```python
from studiosr import Evaluator, Trainer
from studiosr.data import DIV2K
from studiosr.models import SwinIR

dataset_dir="path/to/dataset_dir",
scale = 4
size = 64
dataset = DIV2K(
    dataset_dir=dataset_dir,
    scale=scale,
    size=size,
    transform=True, # data augmentations
    to_tensor=True,
    download=True, # if you don't have the dataset
)
evaluator = Evaluator(scale=scale)

model = SwinIR(scale=scale)
trainer = Trainer(model, dataset, evaluator)
trainer.run()
```

### Evaluate
```python
from studiosr import Evaluator
from studiosr.models import SwinIR
from studiosr.utils import get_device

scale = 2  # 2, 3, 4
dataset = "Set5"  # Set5, Set14, BSD100, Urban100, Manga109
device = get_device()
model = SwinIR.from_pretrained(scale=scale).eval().to(device)
evaluator = Evaluator(dataset, scale=scale)
psnr, ssim = evaluator(model.inference)
```


## Benchmark
- The evaluation metric is [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).
- You can check the full benchmark [here](./docs/benchmark.md).

| Method | Scale | Training Dataset | Set5   | Set14  | BSD100 | Urban100 |
| ------ | ----- | ---------------- | ------ | ------ | ------ | -------- |
| EDSR   | x 4   | DIV2K            | 32.485 | 28.814 | 27.721 | 26.646   |
| RCAN   | x 4   | DIV2K            | 32.639 | 28.851 | 27.744 | 26.745   |
| SwinIR | x 4   | DF2K             | 32.916 | 29.087 | 27.919 | 27.453   |
| HAT    | x 4   | DF2K             | 33.055 | 29.235 | 27.988 | 27.945   |

| Method | Scale | Training Dataset | Set5   | Set14  | BSD100 | Urban100 |
| ------ | ----- | ---------------- | ------ | ------ | ------ | -------- |
| EDSR   | x 3   | DIV2K            | 34.680 | 30.533 | 29.263 | 28.812   |
| RCAN   | x 3   | DIV2K            | 34.758 | 30.627 | 29.302 | 29.009   |
| SwinIR | x 3   | DF2K             | 34.974 | 30.929 | 29.456 | 29.752   |
| HAT    | x 3   | DF2K             | 35.097 | 31.074 | 29.525 | 30.206   |

| Method | Scale | Training Dataset | Set5   | Set14  | BSD100 | Urban100 |
| ------ | ----- | ---------------- | ------ | ------ | ------ | -------- |
| EDSR   | x 2   | DIV2K            | 38.193 | 33.948 | 32.352 | 32.967   |
| RCAN   | x 2   | DIV2K            | 38.271 | 34.126 | 32.390 | 33.176   |
| SwinIR | x 2   | DF2K             | 38.415 | 34.458 | 32.526 | 33.812   |
| HAT    | x 2   | DF2K             | 38.605 | 34.845 | 32.590 | 34.418   |

## License
StudioSR is an open-source library under the **MIT license**. 
