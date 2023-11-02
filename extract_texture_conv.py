import math
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

# scene_name = "0b16abb1-4a59-4ce3-85b5-8ec10440d9dd"
# short_prompt = "classic"  # used for output_dir
# kernel_size = 11
# sigma = 1


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--scene_name", type=str, default="0b16abb1-4a59-4ce3-85b5-8ec10440d9dd"
    )
    arg_parser.add_argument("--short_prompt", type=str, default="classic")
    arg_parser.add_argument("--kernel_size", type=int, default=11)
    arg_parser.add_argument("--sigma", type=float, default=1)
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    scene_name = args.scene_name
    short_prompt = args.short_prompt
    kernel_size = args.kernel_size
    sigma = args.sigma

    output_dir = Path("outputs") / "mvdiffusion" / scene_name / short_prompt / "images"

    pred_files = sorted(output_dir.glob("*_pred.png"))

    result_sum = np.zeros((4096, 4096, 3), dtype=np.float64)
    result_count = np.zeros((4096, 4096, 1), dtype=np.float64)
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * math.pi * sigma**2))
        * math.e
        ** (
            (-1 * ((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2))
            / (2 * sigma**2)
        ),
        (kernel_size, kernel_size),
    )
    # kernel /= np.sum(kernel)
    kernel = kernel.reshape(kernel_size, kernel_size, 1)

    for png_file in pred_files:
        view_id = int(png_file.stem.split("_")[0])
        pred_image = cv2.imread(str(png_file))
        uv_file = png_file.parent / f"{view_id}_uv.npy"
        uv_coord = np.load(uv_file)
        uv_pix = uv_coord.reshape(-1, 2)
        valid_uv = (uv_pix > 0).all(axis=1)
        uv_pix = np.floor(uv_pix * 4096).astype(int)
        uv_pix[:, 1] = 4096 - uv_pix[:, 1] - 1
        uv_pix = uv_pix[valid_uv]
        target_rgb = pred_image.reshape(-1, 3)[valid_uv]
        # result_sum[uv_pix[:, 1], uv_pix[:, 0]] += target_rgb
        # result_count[uv_pix[:, 1], uv_pix[:, 0]] += 1
        for i, (x, y) in enumerate(uv_pix):
            kernel_mid = (kernel_size - 1) // 2
            ym = min(y, kernel_mid)
            yp = min(4096 - y, kernel_mid + 1)
            xm = min(x, kernel_mid)
            xp = min(4096 - x, kernel_mid + 1)
            result_sum[y - ym : y + yp, x - xm : x + xp] += (
                kernel * target_rgb[i].reshape(1, 1, 3)
            )[kernel_mid - ym : kernel_mid + yp, kernel_mid - xm : kernel_mid + xp]
            result_count[y - ym : y + yp, x - xm : x + xp] += kernel[
                kernel_mid - ym : kernel_mid + yp, kernel_mid - xm : kernel_mid + xp
            ]

    result = result_sum / (result_count + 1e-8)
    result = result.astype(np.uint8)
    texture_file = (
        Path("outputs") / "mvdiffusion" / f"mvdiffusion_{scene_name}_{short_prompt}.png"
    )
    cv2.imwrite(str(texture_file), result)
    print("texture written to ", texture_file)
