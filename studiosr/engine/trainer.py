import os
import platform
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from studiosr.data import DataIterator
from studiosr.engine import Evaluator


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
        model: nn.Module,
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
        log_interval: int = 1000,
        ckpt_path: str = "checkpoints",
        bfloat16: bool = True,
        seed: int = 0,
    ):

        self.model = model
        self.dataset = train_dataset
        self.evaluator = evaluator

        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.max_iters = max_iters
        self.log_interval = log_interval
        self.ckpt_path = ckpt_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bfloat16 = bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else False
        self.seed = seed

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        self.criterion = nn.L1Loss()

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        device = self.device
        dtype = torch.bfloat16 if self.bfloat16 else torch.float32
        print(f"device: {device}  dtype: {dtype}")
        ctx = torch.autocast(device_type=device, dtype=dtype)
        ddp = int(os.environ.get("RANK", -1)) != -1
        if ddp:
            backend = "nccl"
            dist.init_process_group(backend=backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=ddp_world_size,
                rank=ddp_rank,
                shuffle=True,
            )
            seed_offset = ddp_rank
        else:
            ddp_world_size = 1
            master_process = True
            sampler = None
            seed_offset = 0
        torch.manual_seed(self.seed + seed_offset)

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size // ddp_world_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            pin_memory=True,
        )
        data_iter = DataIterator(dataloader)

        model = self.model
        model.to(device)
        if ddp:
            model = DDP(model, device_ids=[device], output_device=device)
        raw_model = model.module if ddp else model

        if master_process:
            os.makedirs(self.ckpt_path, exist_ok=True)

        model = model.train()
        self.iter_num = 0
        while True:
            self.iter_num += 1
            if self.iter_num > self.max_iters:
                break
            x, y = data_iter.get_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                out = model(x)
                loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            print(f" Iterations = {self.iter_num:<8}", end="\r")
            if self.iter_num % self.log_interval == 0 and master_process:
                if self.evaluator:
                    raw_model = raw_model.eval()
                    psnr, ssim = self.evaluator.run(raw_model.inference, log=False)
                    raw_model = raw_model.train()
                    print(f" Iterations = {self.iter_num:<8} PSNR {psnr}")
                torch.save(
                    raw_model.state_dict(),
                    os.path.join(self.ckpt_path, "ckpt_%d.pth" % (self.iter_num)),
                )

        if ddp:
            dist.destroy_process_group()
