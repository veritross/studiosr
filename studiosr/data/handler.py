import os
import random
from typing import Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DataIterator:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.iterations = 0

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_batch()

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        self.iterations += 1
        return batch

    @property
    def epochs(self) -> float:
        return self.iterations / len(self.dataloader)

    def set_iterations(self, iterations: int) -> None:
        self.iterations = iterations


class DataHandler:
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ddp_rank = int(os.environ.get("RANK", -1))
        self.ddp_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.ddp_world_size = int(os.environ.get("WORLD_SIZE", -1))
        self.ddp_enabled = self.ddp_rank != -1
        if self.ddp_enabled:
            backend = "nccl"
            dist.init_process_group(backend=backend)
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.is_main_process = self.ddp_rank == 0
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.ddp_world_size,
                rank=self.ddp_rank,
                shuffle=True,
            )
        else:
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.is_main_process = True
            self.sampler = None

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size // self.ddp_world_size,
            num_workers=self.num_workers,
            sampler=self.sampler,
            shuffle=self.sampler is None,
            drop_last=True,
            pin_memory=True,
        )
        self.data_iterator = DataIterator(dataloader)

    @property
    def iterations(self) -> int:
        return self.data_iterator.iterations

    @property
    def epochs(self) -> float:
        return self.data_iterator.epochs

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_iterator.get_batch()

    def set_seed(self, seed: int) -> None:
        random.seed(seed + self.ddp_rank)
        torch.manual_seed(seed + self.ddp_rank)

    def set_iterations(self, iterations: int) -> None:
        self.data_iterator.set_iterations(iterations)

    def close(self) -> None:
        if self.ddp_enabled:
            dist.destroy_process_group()
