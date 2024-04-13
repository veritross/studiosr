import argparse
import os

from studiosr.models import EDSR, HAN, HAT, RCAN, VDSR, SwinIR
from studiosr.utils import get_device, get_image_files, imread, imwrite


def main() -> None:
    models = dict(
        vdsr=VDSR,
        edsr=EDSR,
        rcan=RCAN,
        han=HAN,
        swinir=SwinIR,
        hat=HAT,
    )

    parser = argparse.ArgumentParser(description="StudioSR")
    parser.add_argument("--image", type=str, default="./", help="image or directory to be upscaled")
    parser.add_argument("--scale", type=int, default=4, help="upscaling factor -> [2, 3, 4]")
    parser.add_argument("--model", type=str, default="swinir", help=f"model name -> {list(models.keys())}")
    parser.add_argument("--output", type=str, default="./studiosr", help="output directory")
    args = parser.parse_args()

    path = args.image
    scale = args.scale
    model_name = args.model
    output_dir = args.output

    paths = [path] if os.path.isfile(path) else [os.path.join(path, file) for file in get_image_files(path)]
    images = {os.path.basename(path): imread(path) for path in paths}
    model = models[model_name].from_pretrained(scale=scale)
    model = model.to(get_device())

    os.makedirs(output_dir, exist_ok=True)
    for file_name, image in images.items():
        image = model.inference(image)
        name = os.path.splitext(file_name)[0]
        save_path = os.path.join(output_dir, f"{name}.{model_name}_x{scale}.png")
        imwrite(save_path, image)
        print(" -> ", save_path)


if __name__ == "__main__":
    main()
