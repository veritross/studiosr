## Examples

### Pipeline Architecture
<p align="center">
  <img width="800" src="../assets/Pipeline_arch.svg">
</p>

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

# Train with the model's training configuration.
model = SwinIR(scale=scale)
config = model.get_training_config()
trainer = Trainer(model, dataset, evaluator, **config)
trainer.run()
```

### Evaluate
```python
import torch

from studiosr import Evaluator
from studiosr.models import SwinIR
from studiosr.utils import get_device

scale = 2  # 2, 3, 4
dataset = "Set5"  # Set5, Set14, BSD100, Urban100, Manga109
device = get_device()
model = SwinIR.from_pretrained(scale=scale).eval().to(device)
evaluator = Evaluator(dataset, scale=scale)
psnr, ssim = evaluator(model.inference)

# Evaluation with self-ensemble
psnr, ssim = evaluator(model.inference_with_self_ensemble)
```

### Benchmark
```python
import torch

from studiosr import Evaluator
from studiosr.models import RCAN, HAN, SwinIR, HAT
from studiosr.utils import get_device

for model_class in [RCAN, HAN, SwinIR, HAT]:
    for scale in [2, 3, 4]:
        device = get_device()
        model = model_class.from_pretrained(scale=scale).eval().to(device)
        print(f"Benchmark -> {model_class.__name__}")
        Evaluator.benchmark(model.inference, scale=scale)
```
