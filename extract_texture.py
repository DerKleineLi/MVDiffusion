from pathlib import Path

import cv2
import numpy as np

scene_name = "LivingRoom-36282"
short_prompt = "classic"  # used for output_dir

if __name__ == "__main__":
    output_dir = Path("outputs") / scene_name / short_prompt / "images"

    pred_files = sorted(output_dir.glob("*_pred.png"))

    result_sum = np.zeros((4096, 4096, 3), dtype=np.int32)
    result_count = np.zeros((4096, 4096, 1), dtype=np.int32)

    for png_file in pred_files:
        view_id = int(png_file.stem.split("_")[0])
        pred_image = cv2.imread(str(png_file))
        uv_file = png_file.parent / f"{view_id}_uv.npy"
        uv_coord = np.load(uv_file)
        uv_pix = uv_coord.reshape(-1, 2)
        uv_pix = np.floor(uv_pix * 4096).astype(int)
        uv_pix[:, 1] = 4096 - uv_pix[:, 1] - 1
        result_sum[uv_pix[:, 1], uv_pix[:, 0]] += pred_image.reshape(-1, 3)
        result_count[uv_pix[:, 1], uv_pix[:, 0]] += 1
    result_count[result_count == 0] = 1
    result = result_sum / result_count

    cv2.imwrite(str(output_dir.parent / "texture.png"), result)
