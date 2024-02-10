import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from studiosr.data import transforms as T
from studiosr.utils import gdown_and_extract, get_image_files, imread


class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset class for handling paired image data used in image super-resolution tasks.

    Args:
        gt_path (str): The path to the directory containing ground-truth high-resolution images.
        lq_path (str): The path to the directory containing low-quality (input) images.
        size (int, optional): The size of patches to extract from images (default is 48).
        scale (int, optional): The scaling factor for super-resolution (default is 4).
        transform (bool, optional): Apply data augmentation transformations (default is False).
        to_tensor (bool, optional): Convert images to PyTorch tensors (default is False).

    Note:
        This dataset class is designed for image super-resolution tasks, where it pairs low-quality
        input images (lq) with corresponding high-quality ground-truth images (gt). It supports
        data augmentation and converting images to PyTorch tensors, making it suitable for training
        deep learning models.
    """

    def __init__(
        self,
        gt_path: str,
        lq_path: str,
        size: int = 48,
        scale: int = 4,
        transform: bool = False,
        to_tensor: bool = False,
    ) -> None:
        self.gt_path = gt_path
        self.lq_path = lq_path
        self.files = sorted(get_image_files(gt_path))
        self.size = size
        self.scale = scale
        self.transform = transform
        self.to_tensor = to_tensor

        if self.transform:
            self.transform = T.Compose(
                [
                    T.RandomCrop(self.size, self.scale),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation90(),
                ]
            )
        if self.to_tensor:
            self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq, gt = self.get_image_pair(idx)
        if self.transform:
            lq, gt = self.transform(lq, gt)
        if self.to_tensor:
            lq, gt = self.to_tensor(lq, gt)
        return lq, gt

    def get_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file = self.files[idx]
        lq_path = os.path.join(self.lq_path, file)
        gt_path = os.path.join(self.gt_path, file)
        lq = imread(lq_path)
        gt = imread(gt_path)
        return lq, gt


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
        torch.manual_seed(seed + self.ddp_rank)

    def close(self) -> None:
        if self.ddp_enabled:
            dist.destroy_process_group()


def extract_subimages(
    input_dir: str,
    output_dir: str,
    crop_size: int,
    step: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = get_image_files(input_dir)
    for f in tqdm(files):
        name = os.path.splitext(f)[0]
        name = name.replace("x2", "").replace("x3", "").replace("x4", "")
        open_path = os.path.join(input_dir, f)
        image = cv2.imread(open_path)

        h, w = image.shape[0:2]
        y_range = np.arange(0, h - crop_size + 1, step)
        if h - (y_range[-1] + crop_size) > 0:
            y_range = np.append(y_range, h - crop_size)
        x_range = np.arange(0, w - crop_size + 1, step)
        if w - (x_range[-1] + crop_size) > 0:
            x_range = np.append(x_range, w - crop_size)

        index = 0
        for y in y_range:
            for x in x_range:
                index += 1
                cropped = image[y : y + crop_size, x : x + crop_size]
                save_path = os.path.join(output_dir, name + f"_{index:03d}.png")
                cv2.imwrite(save_path, cropped)


class DIV2K(PairedImageDataset):
    def __init__(
        self,
        dataset_dir: str,
        size: int = 48,
        scale: int = 4,
        transform: bool = False,
        to_tensor: bool = False,
        download: bool = False,
    ):
        if download:
            self.download(dataset_dir=dataset_dir)
            self.prepare(dataset_dir=dataset_dir)
        dataset_path = os.path.join(dataset_dir, "DIV2K/sub")
        gt_path = os.path.join(dataset_path, "DIV2K_train_HR")
        lq_path = os.path.join(dataset_path, f"DIV2K_train_LR_bicubic/X{scale}")
        super().__init__(
            gt_path=gt_path,
            lq_path=lq_path,
            size=size,
            scale=scale,
            transform=transform,
            to_tensor=to_tensor,
        )

    @classmethod
    def download(cls, dataset_dir: str) -> None:
        id = "1rhaiGcXoivv5pJKIf7Wy1QJHZ-tgiyB4"
        gdown_and_extract(id=id, save_dir=dataset_dir)

    @classmethod
    def prepare(cls, dataset_dir: str) -> None:
        dataset_dir = os.path.join(dataset_dir, "DIV2K")
        output_dir = os.path.join(dataset_dir, "sub")
        extract_subimages(
            input_dir=os.path.join(dataset_dir, "DIV2K_train_HR"),
            output_dir=os.path.join(output_dir, "DIV2K_train_HR"),
            crop_size=480,
            step=240,
        )
        extract_subimages(
            input_dir=os.path.join(dataset_dir, "DIV2K_train_LR_bicubic/X2"),
            output_dir=os.path.join(output_dir, "DIV2K_train_LR_bicubic/X2"),
            crop_size=240,
            step=120,
        )
        extract_subimages(
            input_dir=os.path.join(dataset_dir, "DIV2K_train_LR_bicubic/X3"),
            output_dir=os.path.join(output_dir, "DIV2K_train_LR_bicubic/X3"),
            crop_size=160,
            step=80,
        )
        extract_subimages(
            input_dir=os.path.join(dataset_dir, "DIV2K_train_LR_bicubic/X4"),
            output_dir=os.path.join(output_dir, "DIV2K_train_LR_bicubic/X4"),
            crop_size=120,
            step=60,
        )
