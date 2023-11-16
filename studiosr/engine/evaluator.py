import os
from typing import Callable

import cv2
import numpy as np

from studiosr.data import PairedImageDataset
from studiosr.utils import compare, compute_psnr, compute_ssim


class Evaluator:
    """
    A class for evaluating the performance of super-resolution models on image datasets.

    Args:
        dataset (str, optional): The name of the evaluation dataset (default is "Set5").
        scale (int, optional): The scaling factor for super-resolution (default is 4).
        root (str, optional): The root directory where evaluation dataset is located (default is "data").

    Note:
        This class is designed for evaluating the performance of super-resolution models. It loads the
        evaluation dataset, calculates PSNR and SSIM values for the model's output, and optionally visualizes
        the results. The class can be used for various evaluation datasets and scaling factors.
    """

    def __init__(
        self,
        dataset: str = "Set5",
        scale: int = 4,
        root: str = "dataset",
    ):
        self.dataset = dataset
        self.scale = scale
        self.root = root
        root = self.download_dataset(self.root, self.dataset)
        gt_mod = 12 if scale in [2, 3, 4] else scale
        gt_path = os.path.join(root, f"GTmod{gt_mod}")
        lq_path = os.path.join(root, f"LRbicx{scale}")
        self.scale = scale
        self.testset = PairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable,
        y_only: bool = True,
        visualize: bool = False,
    ):
        return self.run(func, y_only, visualize)

    def run(
        self,
        func: Callable,
        y_only: bool = True,
        log: bool = True,
        visualize: bool = False,
    ):
        crop_border = self.scale
        psnrs, ssims = [], []
        for i, (lq, gt) in enumerate(self.testset):
            sr = func(lq)
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)
            if log:
                print(f" {self.dataset:>8} - {i+1:<3}  PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                compare([nn[:, :, ::-1], bc[:, :, ::-1], sr[:, :, ::-1], gt[:, :, ::-1]])
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        if log:
            print(f"\n {self.dataset:>8} - Avg. PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}\n")
        return psnr, ssim

    @staticmethod
    def download_dataset(root: str = "data", dataset: str = "Set5"):
        import zipfile

        import gdown

        dataset_id = {
            "Set5": "18bimJIcXV0nxYU9y64Liwo63afEZXlAY",
            "Set14": "1Wn8mJRFT7N4z0cGbqwGev4ltbLwi4Sg2",
            "BSD100": "1qoiBkwiUgv62MISQh4A4nibdmDfP5qzJ",
            "Urban100": "1YTYp0gVJj2gpIsL3N8NkEDKEPIZeyhnf",
            "Manga109": "1ZaUD3ZeaaI3zHlEI6HRSx0baBU2CeYe7",
        }
        benchmark_path = os.path.join(root, dataset)
        if not os.path.exists(benchmark_path):
            os.makedirs(root, exist_ok=True)
            id = dataset_id[dataset]
            zip_path = benchmark_path + ".zip"
            gdown.download(id=id, output=zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(root)
            os.remove(zip_path)
        return benchmark_path

    @staticmethod
    def benchmark(func: Callable, scale: int = 4, y_only: bool = True, log: bool = True) -> None:
        log_data = "| Metric |"
        log_line = "| ------ |"
        log_psnr = "|   PSNR |"
        log_ssim = "|   SSIM |"
        for dataset in ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]:
            evaluator = Evaluator(dataset, scale)
            psnr, ssim = evaluator.run(func, y_only, log=log)

            log_data += " %8s |" % dataset
            log_line += " -------- |"
            log_psnr += " %8.3f |" % psnr
            log_ssim += " %8.4f |" % ssim

        print(log_data)
        print(log_line)
        print(log_psnr)
        print(log_ssim)
        print()
