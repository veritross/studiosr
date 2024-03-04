import os
import platform
from typing import Callable, List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from studiosr.data import DataHandler
from studiosr.engine import Evaluator
from studiosr.models.common import BaseModule
from studiosr.utils import Logger, get_device


class Trainer:
    """
    A class for training a neural network model with various configurations.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_dataset (Dataset): The dataset used for training.
        evaluator (Evaluator, optional): An evaluator for model performance (default is None).
        batch_size (int, optional): Batch size for training (default is 32).
        tbWriter (SummaryWriter, optional): TensorBoard SummaryWriter for logging (default is None).

    Note:
        This class is designed for training a neural network model using PyTorch.
    """

    def __init__(
        self,
        model: BaseModule,
        train_dataset: Dataset,
        evaluator: Evaluator = None,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 0.0002,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        max_iters: int = 500000,
        gamma: float = 0.5,
        milestones: List[int] = [250000, 400000, 450000, 475000],
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.L1Loss(),
        eval_interval: int = 100,
        ckpt_path: str = "checkpoints",
        bfloat16: bool = True,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.dataset = train_dataset
        self.evaluator = evaluator

        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.ckpt_path = ckpt_path
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.device = get_device()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and bfloat16 else torch.float32
        self.seed = seed

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        self.criterion = loss_function

    def run(self) -> None:
        device, dtype = self.device, self.dtype
        print(f"device: {device}  dtype: {dtype}")
        ctx = torch.autocast(device_type=device, dtype=dtype)

        data_handler = DataHandler(self.dataset, self.batch_size, self.num_workers)
        data_handler.set_seed(self.seed)

        model = self.model.to(device)
        if data_handler.ddp_enabled:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[device], output_device=device)

        if data_handler.is_main_process:
            logger = Logger(os.path.join(self.ckpt_path, "train.log"))

        best_psnr = 0.0
        model = model.train()
        while data_handler.iterations < self.max_iters:
            x, y = data_handler.get_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                out = model(x)
                loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            iterations = data_handler.iterations
            print(f" Iterations = {iterations:<8}", end="\r")
            if iterations % self.eval_interval == 0 and data_handler.is_main_process:
                psnr, ssim = self.evaluate()
                log = f" Iterations = {iterations:<8}  PSNR: {psnr:6.3f} SSIM: {ssim:6.4f}"
                logger.info(log)
                if best_psnr <= psnr:
                    print(log, end="\r")
                    best_psnr = psnr
                    self.save("best.pth")

        data_handler.close()

    def evaluate(self) -> List[float]:
        psnr, ssim = 0.0, 0.0
        if self.evaluator:
            self.model = self.model.eval()
            psnr, ssim = self.evaluator.run(self.model.inference)
            self.model = self.model.train()
        return psnr, ssim

    def save(self, file_name: str) -> str:
        os.makedirs(self.ckpt_path, exist_ok=True)
        save_path = os.path.join(self.ckpt_path, file_name)
        torch.save(self.model.state_dict(), save_path)
        return save_path
