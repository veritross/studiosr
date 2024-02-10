import argparse
import os

from studiosr.utils import extract_subimages, gdown_and_extract


def download_div2k(save_dir: str):
    id = "1rhaiGcXoivv5pJKIf7Wy1QJHZ-tgiyB4"
    gdown_and_extract(id=id, save_dir=save_dir)


def main() -> None:

    parser = argparse.ArgumentParser(description="StudioSR Dataset")
    parser.add_argument("--dataset", type=str, default="DIV2K", help="dataset to be downloaded")
    parser.add_argument("--dir", type=str, default="./dataset", help="target directory")
    args = parser.parse_args()

    dataset = args.dataset
    save_dir = args.dir

    if dataset == "DIV2K":
        download_div2k(save_dir=save_dir)


if __name__ == "__main__":
    main()
