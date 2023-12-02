import os

import cv2
from torch.utils.data import Dataset

from studiosr.data import transforms as T

IMAGE_EXTENSIONS = [
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".avif",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".pfm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
    ".pic",
]


def get_image_files(root: str) -> list:
    image_files = []
    for (root, dirs, files) in os.walk(root):
        for f in files:
            extension = os.path.splitext(f)[1].lower()
            if extension in IMAGE_EXTENSIONS:
                image_files.append(f)
    return image_files


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
    ):
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        lq, gt = self.get_image_pair(idx)
        if self.transform:
            lq, gt = self.transform(lq, gt)
        if self.to_tensor:
            lq, gt = self.to_tensor(lq, gt)
        return lq, gt

    def get_image_pair(self, idx: int):
        file = self.files[idx]
        lq_path = os.path.join(self.lq_path, file)
        gt_path = os.path.join(self.gt_path, file)
        lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        return lq, gt


class DataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

    def __call__(self):
        return self.get_batch()

    def get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch
