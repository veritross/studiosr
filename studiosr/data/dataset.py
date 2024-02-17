import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
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
        self.files = get_image_files(gt_path)
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


def prepare_dataset(dataset_dir: str, dataset_name: str, postfix: str = ""):
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    sub_dir = os.path.join(dataset_dir, "sub")
    packs = [
        dict(dir_name=f"{dataset_name}{postfix}_HR", crop_size=480, step=240),
        dict(dir_name=f"{dataset_name}{postfix}_LR_bicubic/X2", crop_size=240, step=120),
        dict(dir_name=f"{dataset_name}{postfix}_LR_bicubic/X3", crop_size=160, step=80),
        dict(dir_name=f"{dataset_name}{postfix}_LR_bicubic/X4", crop_size=120, step=60),
    ]

    for pack in packs:
        dir_name, crop_size, step = pack["dir_name"], pack["crop_size"], pack["step"]
        input_dir = os.path.join(dataset_dir, dir_name)
        output_dir = os.path.join(sub_dir, dir_name)
        if not os.path.exists(output_dir):
            extract_subimages(input_dir=input_dir, output_dir=output_dir, crop_size=crop_size, step=step)


class DIV2K(PairedImageDataset):
    dataset_name = "DIV2K"

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
        dataset_path = os.path.join(dataset_dir, f"{self.dataset_name}/sub")
        if not os.path.exists(dataset_path):
            self.prepare(dataset_dir=dataset_dir)
        gt_path = os.path.join(dataset_path, f"{self.dataset_name}_train_HR")
        lq_path = os.path.join(dataset_path, f"{self.dataset_name}_train_LR_bicubic/X{scale}")
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
        prepare_dataset(dataset_dir, cls.dataset_name, "_train")


class Flickr2K(PairedImageDataset):
    dataset_name = "Flickr2K"

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
        dataset_path = os.path.join(dataset_dir, f"{self.dataset_name}/sub")
        if not os.path.exists(dataset_path):
            self.prepare(dataset_dir=dataset_dir)
        gt_path = os.path.join(dataset_path, f"{self.dataset_name}_HR")
        lq_path = os.path.join(dataset_path, f"{self.dataset_name}_LR_bicubic/X{scale}")
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
        id = "1--pNeHQlsaIWPzSnnIPzmvPpimdIhN5C"
        gdown_and_extract(id=id, save_dir=dataset_dir)

    @classmethod
    def prepare(cls, dataset_dir: str) -> None:
        prepare_dataset(dataset_dir, cls.dataset_name)


class DF2K:
    def __init__(
        self,
        dataset_dir: str,
        size: int = 48,
        scale: int = 4,
        transform: bool = False,
        to_tensor: bool = False,
        download: bool = False,
    ):
        self.size = size
        self.scale = scale
        self.transform = transform
        self.to_tensor = to_tensor

        if download:
            DIV2K.download(dataset_dir=dataset_dir)
            Flickr2K.download(dataset_dir=dataset_dir)
        div2k_path = os.path.join(dataset_dir, "DIV2K/sub")
        flickr2k_path = os.path.join(dataset_dir, "Flickr2K/sub")
        if not os.path.exists(div2k_path):
            DIV2K.prepare(dataset_dir=dataset_dir)
        if not os.path.exists(flickr2k_path):
            Flickr2K.prepare(dataset_dir=dataset_dir)

        self.file_paths = []

        div2k_gt_path = os.path.join(div2k_path, "DIV2K_train_HR")
        div2k_lq_path = os.path.join(div2k_path, f"DIV2K_train_LR_bicubic/X{scale}")
        files = get_image_files(div2k_gt_path)
        for f in files:
            gt_file = os.path.join(div2k_gt_path, f)
            lq_file = os.path.join(div2k_lq_path, f)
            self.file_paths.append((lq_file, gt_file))

        flickr2k_gt_path = os.path.join(flickr2k_path, "Flickr2K_HR")
        flickr2k_lq_path = os.path.join(flickr2k_path, f"Flickr2K_LR_bicubic/X{scale}")
        files = get_image_files(flickr2k_gt_path)
        for f in files:
            gt_file = os.path.join(flickr2k_gt_path, f)
            lq_file = os.path.join(flickr2k_lq_path, f)
            self.file_paths.append((lq_file, gt_file))

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
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq, gt = self.get_image_pair(idx)
        if self.transform:
            lq, gt = self.transform(lq, gt)
        if self.to_tensor:
            lq, gt = self.to_tensor(lq, gt)
        return lq, gt

    def get_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        lq_path, gt_path = self.file_paths[idx]
        lq = imread(lq_path)
        gt = imread(gt_path)
        return lq, gt
