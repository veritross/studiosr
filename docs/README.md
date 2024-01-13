## Examples

### Pipeline Architecture
<p align="center">
  <img width="800" src="../assets/Pipeline_arch.svg">
</p>

### Train
```python
from studiosr import Evaluator, Trainer
from studiosr.data import PairedImageDataset
from studiosr.models import SRCNN

scale = 4
size = 64
batch_size = 32

gt_path = "/data/DIV2K_train_HR"
lq_path = "/data/DIV2K_train_LR_bicubic/X4"
dataset = PairedImageDataset(gt_path, lq_path, size, scale, True, True)
evaluator = Evaluator(scale=scale)

model = SRCNN(scale=scale)
trainer = Trainer(model, dataset, evaluator, batch_size=batch_size)
trainer.run()

# Train with the model's training configuration.
model = EDSR(scale=scale)
config = model.get_training_config()
trainer = Trainer(model, dataset, evaluator, **config)
trainer.run()
```

### Evaluate
```python
import torch

from studiosr import Evaluator
from studiosr.models import SwinIR

scale = 2  # 2, 3, 4
dataset = "Set5"  # Set5, Set14, BSD100, Urban100, Manga109
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SwinIR.from_pretrained(scale=scale).eval().to(device)
evaluator = Evaluator(dataset, scale=scale)
psnr, ssim = evaluator(model.inference)
```

### Benchmark
```python
import torch

from studiosr import Evaluator
from studiosr.models import RCAN, HAN, SwinIR, HAT

for model_class in [RCAN, HAN, SwinIR, HAT]:
    for scale in [2, 3, 4]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_class.from_pretrained(scale=scale).eval().to(device)
        print(f"Benchmark -> {model_class.__name__}")
        Evaluator.benchmark(model.inference, scale=scale)
```